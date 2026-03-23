#pragma once
#include <cstdint>
#include <atomic>
#include <array>
#include <vector>

namespace prime {

// ═══════════════════════════════════════════════════════════════════════
// Three-Way Work Request Pool: GPU + CPU + NEON
//
// Three processor types compete for work through a shared request pool:
//
//   GPU  — massive parallelism, batch primality tests
//   CPU  — scalar, full verification, flexible
//   NEON — SIMD pre-filter: 2 candidates × trial division in parallel
//          eliminates composites before they reach GPU/CPU
//
// Each processor signals "I'm idle" after finishing work.
// Tasks call advise() which checks the pool and routes candidates
// to whichever processor has real spare capacity.
//
// NEON pre-filter pipeline:
//   candidates → NEON trial div (eliminates ~80% composites)
//             → survivors go to GPU batch or CPU scalar test
// ═══════════════════════════════════════════════════════════════════════

static constexpr int MAX_TASKS = 12;

struct alignas(16) TaskMetrics {
    uint32_t cand_rate;
    uint32_t gpu_queued;
    uint32_t cpu_queued;
    uint32_t sieve_density;
};

struct DispatchAdvice {
    float gpu_ratio;       // fraction of survivors → GPU (0.0–1.0)
    float neon_ratio;      // fraction of candidates → NEON pre-filter first (0.0–1.0)
    bool throttle_gpu;     // true = skip GPU entirely this batch
};

class LoadBalancer {
public:
    LoadBalancer();

    // ── GPU busy tracking ───────────────────────────────────────────
    void record_gpu_busy(int64_t busy_us);

    // ── Three-Way Work Request Pool ─────────────────────────────────
    void gpu_idle();
    void cpu_idle();
    void neon_idle();
    bool gpu_wants_work();
    bool cpu_wants_work();
    bool neon_wants_work();

    // ── NEON batch pre-filter ───────────────────────────────────────
    // Trial-divides candidates by small primes using NEON uint64x2_t.
    // Returns indices of survivors (probable primes) that need full test.
    // This IS the NEON work — it pulls from the pool and filters.
    static std::vector<uint32_t> neon_prefilter(const uint64_t* candidates,
                                                 uint32_t count);

    // ── Metrics + Advice ────────────────────────────────────────────
    void report(int task_id, uint32_t cand_rate, uint32_t gpu_q,
                uint32_t cpu_q, uint32_t density);
    DispatchAdvice advise(int task_id);

    float gpu_saturation() const;
    float cpu_pressure() const;
    float gpu_target = 0.50f;

private:
    alignas(64) std::array<TaskMetrics, MAX_TASKS> _metrics{};

    // Rolling GPU utilization window
    std::atomic<int64_t> _window_busy_us{0};
    std::atomic<int64_t> _window_start_us{0};
    static constexpr int64_t WINDOW_US = 200000;
    std::atomic<uint32_t> _dispatch_count{0};

    // Three-way request pool — lock-free atomic counters
    std::atomic<int32_t> _gpu_wants{0};
    std::atomic<int32_t> _cpu_wants{0};
    std::atomic<int32_t> _neon_wants{0};

    // NEON scan internals
    float neon_scan_gpu_pressure() const;
    float neon_scan_cpu_pressure() const;
};

} // namespace prime
