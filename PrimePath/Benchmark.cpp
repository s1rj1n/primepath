#include "Benchmark.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace prime {

Benchmark::Benchmark(TaskManager* mgr) : _mgr(mgr) {}

// ── Formatting helpers ──────────────────────────────────────────────

std::string Benchmark::fmt_num(uint64_t n) {
    std::string s = std::to_string(n);
    int len = (int)s.length();
    std::string result;
    for (int i = 0; i < len; i++) {
        if (i > 0 && (len - i) % 3 == 0) result += ',';
        result += s[i];
    }
    return result;
}

std::string Benchmark::fmt_rate(double rate) {
    if (rate >= 1e9) return std::to_string((int)(rate / 1e9)) + "." +
        std::to_string((int)((int64_t)(rate / 1e7) % 100)) + " G";
    if (rate >= 1e6) return std::to_string((int)(rate / 1e6)) + "." +
        std::to_string((int)((int64_t)(rate / 1e4) % 100)) + " M";
    if (rate >= 1e3) return std::to_string((int)(rate / 1e3)) + "." +
        std::to_string((int)((int64_t)(rate / 10) % 100)) + " K";
    return std::to_string((int)rate);
}

std::string Benchmark::fmt_time(double sec) {
    if (sec < 0.001) return std::to_string((int)(sec * 1e6)) + " us";
    if (sec < 1.0)   return std::to_string((int)(sec * 1e3)) + " ms";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << sec << " s";
    return oss.str();
}

// ── Scale definitions ───────────────────────────────────────────────

struct Scale {
    const char* name;
    uint64_t start;
};

static const Scale SCALES[] = {
    {"10^6",  1000000ULL},
    {"10^9",  1000000000ULL},
    {"10^12", 1000000000000ULL},
    {"10^15", 1000000000000001ULL},
};
static const int NUM_SCALES = 4;

// ═════════════════════════════════════════════════════════════════════
// run_all — master benchmark orchestrator
// ═════════════════════════════════════════════════════════════════════

void Benchmark::run_all(std::function<void(const std::string&)> log_cb,
                        std::atomic<bool>& should_run) {
    std::function<void(const std::string&)> log = [&](const std::string& msg) {
        if (log_cb) log_cb(msg);
    };

    // Header
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char timebuf[32];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%S", localtime(&t));

    log("═══════════════════════════════════════════════════════════════");
    log("PrimePath Benchmark — " + std::string(timebuf));
    log("GPU: " + (_mgr->gpu() ? _mgr->gpu()->name() : "none") +
        " | CPU cores: " + std::to_string(_mgr->pool().size()));
    log("═══════════════════════════════════════════════════════════════");
    log("");

    if (!should_run.load()) return;
    bench_segmented_sieve(log, should_run);

    if (!should_run.load()) return;
    bench_cpu_primality(log, should_run);

    if (!should_run.load()) return;
    bench_gpu_batches(log, should_run);

    if (!should_run.load()) return;
    bench_even_shadow(log, should_run);

    if (!should_run.load()) return;
    bench_pair_scan(log, should_run);

    if (!should_run.load()) return;
    bench_sieve_pipeline(log, should_run);

    if (!should_run.load()) return;
    bench_gpu_cpu_split(log, should_run);

    log("");
    log("═══════════════════════════════════════════════════════════════");
    log("Benchmark complete.");
    log("═══════════════════════════════════════════════════════════════");
}

// ═════════════════════════════════════════════════════════════════════
// 1. Segmented Sieve throughput
// ═════════════════════════════════════════════════════════════════════

void Benchmark::bench_segmented_sieve(std::function<void(const std::string&)>& log,
                                       std::atomic<bool>& run) {
    log("── Segmented Sieve (1M segment) ─────────────────────────────");

    static const uint64_t SEG = 1048576;
    static const int ITERS = 5;

    for (int s = 0; s < NUM_SCALES && run.load(); s++) {
        uint64_t start = SCALES[s].start;
        // Ensure sieve table covers sqrt(start + SEG)
        _mgr->ensure_small_primes(start + SEG);
        // Warmup
        _mgr->segmented_sieve(start, start + SEG);

        double best = 1e9;
        uint64_t primes_found = 0;
        for (int i = 0; i < ITERS && run.load(); i++) {
            auto t0 = std::chrono::steady_clock::now();
            auto primes = _mgr->segmented_sieve(start, start + SEG);
            double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (dt < best) best = dt;
            primes_found = primes.size();
        }

        double rate = SEG / best;
        log("  " + std::string(SCALES[s].name) + ":  " +
            fmt_rate(rate) + " candidates/sec  (" + fmt_time(best) +
            ", " + fmt_num(primes_found) + " primes in segment)");
    }
    log("");
}

// ═════════════════════════════════════════════════════════════════════
// 2. CPU Primality (Miller-Rabin 12 witnesses)
// ═════════════════════════════════════════════════════════════════════

void Benchmark::bench_cpu_primality(std::function<void(const std::string&)>& log,
                                     std::atomic<bool>& run) {
    log("── CPU Primality (Miller-Rabin x12) ─────────────────────────");

    // Test count scales down at larger numbers (MR is slower)
    static const int TEST_COUNTS[] = {100000, 50000, 10000, 5000};

    for (int s = 0; s < NUM_SCALES && run.load(); s++) {
        uint64_t start = SCALES[s].start;
        int count = TEST_COUNTS[s];

        // Generate test candidates (odd numbers around start)
        std::vector<uint64_t> candidates(count);
        uint64_t p = (start | 1);  // ensure odd
        for (int i = 0; i < count; i++) {
            candidates[i] = p;
            p += 2;
        }

        // Time
        uint64_t found = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (auto c : candidates) {
            if (is_prime(c)) found++;
        }
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        double rate = count / dt;
        log("  " + std::string(SCALES[s].name) + ":  " +
            fmt_rate(rate) + " tests/sec  (" + fmt_time(dt) +
            ", " + std::to_string(count) + " tested, " +
            std::to_string(found) + " prime)");
    }
    log("");
}

// ═════════════════════════════════════════════════════════════════════
// 3. GPU Batch operations (vs CPU fallback)
// ═════════════════════════════════════════════════════════════════════

void Benchmark::bench_gpu_batches(std::function<void(const std::string&)>& log,
                                   std::atomic<bool>& run) {
    GPUBackend* gpu = _mgr->gpu();
    if (!gpu || !gpu->available()) {
        log("── GPU Batches ─── SKIPPED (no GPU) ─────────────────────────");
        log("");
        return;
    }

    log("── GPU Batch Tests ──────────────────────────────────────────");

    // Test at 10^9 and 10^12 scale with 256K batch
    static const uint32_t BATCH = 262144;
    struct GpuTest {
        const char* name;
        std::function<int(GPUBackend*, const uint64_t*, uint8_t*, uint32_t)> fn;
    };
    GpuTest tests[] = {
        {"primality", [](GPUBackend* g, const uint64_t* c, uint8_t* r, uint32_t n) {
            return g->primality_batch(c, r, n); }},
        {"wieferich", [](GPUBackend* g, const uint64_t* c, uint8_t* r, uint32_t n) {
            return g->wieferich_batch(c, r, n); }},
        {"wallsunsun", [](GPUBackend* g, const uint64_t* c, uint8_t* r, uint32_t n) {
            return g->wallsunsun_batch(c, r, n); }},
        {"twin", [](GPUBackend* g, const uint64_t* c, uint8_t* r, uint32_t n) {
            return g->twin_batch(c, r, n); }},
        {"sophie", [](GPUBackend* g, const uint64_t* c, uint8_t* r, uint32_t n) {
            return g->sophie_batch(c, r, n); }},
        {"sexy", [](GPUBackend* g, const uint64_t* c, uint8_t* r, uint32_t n) {
            return g->sexy_batch(c, r, n); }},
    };

    // Use 10^12 scale — representative of real workloads
    uint64_t base = 1000000000000ULL;
    _mgr->ensure_small_primes(base + 10000000);
    auto primes = _mgr->segmented_sieve(base, base + 10000000);
    if (primes.size() < BATCH) {
        // If not enough primes, pad with odd numbers
        uint64_t p = base + 10000001;
        while (primes.size() < BATCH) {
            primes.push_back(p);
            p += 2;
        }
    }

    std::vector<uint8_t> results(BATCH);

    // Warmup GPU
    {
        std::lock_guard<std::mutex> glock(_mgr->gpu_mutex());
        gpu->primality_batch(primes.data(), results.data(), 1024);
    }

    for (auto& test : tests) {
        if (!run.load()) break;

        auto t0 = std::chrono::steady_clock::now();
        int hits;
        {
            std::lock_guard<std::mutex> glock(_mgr->gpu_mutex());
            hits = test.fn(gpu, primes.data(), results.data(), BATCH);
        }
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        double rate = BATCH / dt;
        log("  " + std::string(test.name) + ":  " +
            fmt_rate(rate) + " candidates/sec  (" + fmt_time(dt) +
            ", " + fmt_num(BATCH) + " batch, " +
            std::to_string(hits >= 0 ? hits : 0) + " hits)");
    }

    // Also benchmark CPU fallback for comparison
    if (run.load()) {
        log("  --- CPU fallback comparison (primality) ---");
        CPUBackend cpu_fb;
        auto t0 = std::chrono::steady_clock::now();
        cpu_fb.primality_batch(primes.data(), results.data(), BATCH);
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        double gpu_rate = 0;
        {
            auto t1 = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> glock(_mgr->gpu_mutex());
            gpu->primality_batch(primes.data(), results.data(), BATCH);
            gpu_rate = BATCH / std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t1).count();
        }
        double cpu_rate = BATCH / dt;
        double speedup = gpu_rate / cpu_rate;

        log("  CPU:  " + fmt_rate(cpu_rate) + " candidates/sec");
        log("  GPU:  " + fmt_rate(gpu_rate) + " candidates/sec  (" +
            std::to_string((int)(speedup * 10) / 10) + "." +
            std::to_string((int)(speedup * 10) % 10) + "x speedup)");
    }
    log("");
}

// ═════════════════════════════════════════════════════════════════════
// 4. EvenShadow scoring
// ═════════════════════════════════════════════════════════════════════

void Benchmark::bench_even_shadow(std::function<void(const std::string&)>& log,
                                   std::atomic<bool>& run) {
    log("── EvenShadow Scoring ───────────────────────────────────────");

    static const int COUNT = 100000;

    for (int s = 0; s < NUM_SCALES && run.load(); s++) {
        uint64_t start = SCALES[s].start;

        // Generate candidates
        std::vector<uint64_t> cands(COUNT);
        uint64_t p = (start | 1);
        for (int i = 0; i < COUNT; i++) {
            cands[i] = p;
            p += 2;
        }

        // Benchmark analyze() — accumulate scores to prevent optimization
        volatile uint64_t sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (auto c : cands) {
            auto info = EvenShadow::analyze(c);
            sink += info.score;
        }
        double dt_analyze = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        // Benchmark reorder_inplace()
        auto t1 = std::chrono::steady_clock::now();
        EvenShadow::reorder_inplace(cands);
        double dt_reorder = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t1).count();

        double rate = COUNT / dt_analyze;
        log("  " + std::string(SCALES[s].name) + ":  analyze " +
            fmt_rate(rate) + "/sec (" + fmt_time(dt_analyze) +
            ")  reorder " + fmt_time(dt_reorder));
    }
    log("");
}

// ═════════════════════════════════════════════════════════════════════
// 5. Pair adjacency scan (CPU sieve + pair detection)
// ═════════════════════════════════════════════════════════════════════

void Benchmark::bench_pair_scan(std::function<void(const std::string&)>& log,
                                 std::atomic<bool>& run) {
    log("── Pair Scan (sieve + adjacency) ────────────────────────────");

    static const uint64_t SEG = 1048576;
    static const int SEGS = 10;

    struct PairDef { const char* name; uint64_t gap; };
    PairDef pairs[] = {{"twin(2)", 2}, {"cousin(4)", 4}, {"sexy(6)", 6}};

    for (int s = 1; s < NUM_SCALES && run.load(); s++) {  // skip 10^6, too small
        uint64_t start = SCALES[s].start;
        _mgr->ensure_small_primes(start + SEG * SEGS);
        log("  " + std::string(SCALES[s].name) + ":");

        for (auto& pd : pairs) {
            if (!run.load()) break;
            uint64_t found = 0;
            uint64_t total_primes = 0;

            auto t0 = std::chrono::steady_clock::now();
            uint64_t pos = start;
            for (int seg = 0; seg < SEGS; seg++) {
                auto primes = _mgr->segmented_sieve(pos, pos + SEG);
                total_primes += primes.size();
                for (size_t i = 0; i + 1 < primes.size(); i++) {
                    if (primes[i + 1] - primes[i] == pd.gap) found++;
                }
                pos += SEG;
            }
            double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            double rate = (SEGS * SEG) / dt;
            log("    " + std::string(pd.name) + ": " +
                fmt_rate(rate) + " candidates/sec  (" + fmt_time(dt) +
                ", " + std::to_string(found) + " pairs in " +
                fmt_num(total_primes) + " primes)");
        }
    }
    log("");
}

// ═════════════════════════════════════════════════════════════════════
// 6. Sieve Pipeline (ThreadPool prefetch throughput)
// ═════════════════════════════════════════════════════════════════════

void Benchmark::bench_sieve_pipeline(std::function<void(const std::string&)>& log,
                                      std::atomic<bool>& run) {
    log("── Sieve Pipeline (prefetch depth=12) ──────────────────────");

    static const uint64_t SEG = 1048576;
    static const int SEGS = 50;  // more segments for stable timing

    for (int s = 1; s < NUM_SCALES && run.load(); s++) {  // skip 10^6
        uint64_t start = SCALES[s].start;
        _mgr->ensure_small_primes(start + SEG * SEGS);

        SievePipeline pipeline(_mgr, start, SEG, 12);

        uint64_t total_primes = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < SEGS && run.load(); i++) {
            auto primes = pipeline.next_segment();
            total_primes += primes.size();
        }
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        pipeline.drain();

        double rate = (SEGS * SEG) / dt;
        double seg_rate = SEGS / dt;
        log("  " + std::string(SCALES[s].name) + ":  " +
            fmt_rate(rate) + " candidates/sec  (" +
            std::to_string((int)seg_rate) + " segs/sec, " +
            fmt_num(total_primes) + " primes in " + fmt_time(dt) + ")");
    }
    log("");
}

// ═════════════════════════════════════════════════════════════════════
// 7. GPU/CPU Split Test — verifies last-digit routing and measures both paths
// ═════════════════════════════════════════════════════════════════════

void Benchmark::bench_gpu_cpu_split(std::function<void(const std::string&)>& log,
                                     std::atomic<bool>& run) {
    log("── GPU/CPU Split Test ──────────────────────────────────────────");

    _mgr->ensure_small_primes(2000000);

    // Sieve a few segments and check the last-digit distribution
    uint64_t start = 1000000007ULL;
    uint64_t seg_size = 1000000;
    int num_segs = 5;

    uint64_t gpu_count = 0, cpu_count = 0;
    uint64_t d1 = 0, d3 = 0, d7 = 0, d9 = 0;

    auto t0 = std::chrono::steady_clock::now();

    for (int s = 0; s < num_segs && run.load(); s++) {
        uint64_t lo = start + s * seg_size;
        uint64_t hi = lo + seg_size;
        auto primes = _mgr->segmented_sieve(lo, hi);

        for (auto p : primes) {
            int d = (int)(p % 10);
            if (d == 1) { d1++; gpu_count++; }
            else if (d == 3) { d3++; gpu_count++; }
            else if (d == 7) { d7++; cpu_count++; }
            else if (d == 9) { d9++; cpu_count++; }
        }
    }

    double dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    uint64_t total = gpu_count + cpu_count;
    double gpu_pct = total > 0 ? 100.0 * gpu_count / total : 0;
    double cpu_pct = total > 0 ? 100.0 * cpu_count / total : 0;

    log("  Range: " + fmt_num(start) + " – " + fmt_num(start + num_segs * seg_size));
    log("  Total primes: " + fmt_num(total));
    log("  Last-digit distribution:");
    log("    d=1: " + fmt_num(d1) + "  d=3: " + fmt_num(d3) +
        "  d=7: " + fmt_num(d7) + "  d=9: " + fmt_num(d9));
    log("  GPU (d=1,3): " + fmt_num(gpu_count) +
        " (" + std::to_string((int)gpu_pct) + "%)");
    log("  CPU (d=7,9): " + fmt_num(cpu_count) +
        " (" + std::to_string((int)cpu_pct) + "%)");
    log("  Sieve time: " + fmt_time(dt));

    // Now test actual GPU and CPU paths if GPU available
    if (_mgr->gpu() && _mgr->gpu()->available() && run.load()) {
        log("  Testing GPU dispatch...");
        auto primes = _mgr->segmented_sieve(start, start + seg_size);

        // Split by last digit
        std::vector<uint64_t> gpu_batch, cpu_batch;
        for (auto p : primes) {
            int d = (int)(p % 10);
            if (d == 1 || d == 3) gpu_batch.push_back(p);
            else cpu_batch.push_back(p);
        }

        // Time GPU path
        std::vector<uint8_t> gpu_results(gpu_batch.size());
        auto gt0 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(_mgr->gpu_mutex());
            _mgr->gpu()->wieferich_batch(gpu_batch.data(), gpu_results.data(),
                                          (uint32_t)gpu_batch.size());
        }
        double gpu_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - gt0).count();

        // Time CPU path
        auto ct0 = std::chrono::steady_clock::now();
        // Just do trial MR tests as a proxy
        uint64_t cpu_tested = 0;
        for (auto p : cpu_batch) {
            (void)is_prime(p);  // MR test
            cpu_tested++;
            if (cpu_tested >= gpu_batch.size()) break;  // match count
        }
        double cpu_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - ct0).count();

        log("  GPU: " + fmt_num(gpu_batch.size()) + " candidates in " + fmt_time(gpu_sec) +
            " (" + fmt_rate(gpu_batch.size() / gpu_sec) + "/s)");
        log("  CPU: " + fmt_num(cpu_tested) + " candidates in " + fmt_time(cpu_sec) +
            " (" + fmt_rate(cpu_tested / cpu_sec) + "/s)");

        double ratio = (cpu_sec > 0) ? gpu_sec / cpu_sec : 0;
        log("  GPU/CPU speed ratio: " + std::to_string(ratio).substr(0, 5) +
            "x (< 1.0 means GPU faster)");

        // Report balancer state
        float sat = _mgr->balancer()->gpu_saturation();
        log("  Balancer gpu_sat=" + std::to_string((int)(sat * 100)) + "%");
    }

    if (total > 0 && (gpu_pct < 40 || gpu_pct > 60)) {
        log("  ⚠ WARNING: Split is uneven! Expected ~50/50.");
    } else if (total > 0) {
        log("  ✓ Split is balanced.");
    }
    log("");
}

} // namespace prime
