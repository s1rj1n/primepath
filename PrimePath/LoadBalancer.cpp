#include "LoadBalancer.hpp"
#include <arm_neon.h>
#include <chrono>
#include <algorithm>

namespace prime {

static int64_t now_us_lb() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

LoadBalancer::LoadBalancer() {
    for (auto& m : _metrics) {
        m.cand_rate = 0;
        m.gpu_queued = 0;
        m.cpu_queued = 0;
        m.sieve_density = 0;
    }
    _window_start_us.store(now_us_lb(), std::memory_order_relaxed);
}

// ── GPU busy time recording ─────────────────────────────────────────

void LoadBalancer::record_gpu_busy(int64_t busy_us) {
    int64_t now = now_us_lb();
    int64_t ws = _window_start_us.load(std::memory_order_relaxed);
    int64_t elapsed = now - ws;

    if (elapsed > WINDOW_US) {
        _window_busy_us.store(busy_us, std::memory_order_relaxed);
        _window_start_us.store(now, std::memory_order_relaxed);
        _dispatch_count.store(1, std::memory_order_relaxed);
    } else {
        _window_busy_us.fetch_add(busy_us, std::memory_order_relaxed);
        _dispatch_count.fetch_add(1, std::memory_order_relaxed);
    }
}

// ── Three-Way Work Request Pool ─────────────────────────────────────

void LoadBalancer::gpu_idle() {
    int32_t cur = _gpu_wants.load(std::memory_order_relaxed);
    if (cur < 4) _gpu_wants.fetch_add(1, std::memory_order_relaxed);
}

void LoadBalancer::cpu_idle() {
    int32_t cur = _cpu_wants.load(std::memory_order_relaxed);
    if (cur < 8) _cpu_wants.fetch_add(1, std::memory_order_relaxed);
}

void LoadBalancer::neon_idle() {
    int32_t cur = _neon_wants.load(std::memory_order_relaxed);
    if (cur < 8) _neon_wants.fetch_add(1, std::memory_order_relaxed);
}

bool LoadBalancer::gpu_wants_work() {
    int32_t cur = _gpu_wants.load(std::memory_order_relaxed);
    while (cur > 0) {
        if (_gpu_wants.compare_exchange_weak(cur, cur - 1,
                std::memory_order_relaxed)) return true;
    }
    return false;
}

bool LoadBalancer::cpu_wants_work() {
    int32_t cur = _cpu_wants.load(std::memory_order_relaxed);
    while (cur > 0) {
        if (_cpu_wants.compare_exchange_weak(cur, cur - 1,
                std::memory_order_relaxed)) return true;
    }
    return false;
}

bool LoadBalancer::neon_wants_work() {
    int32_t cur = _neon_wants.load(std::memory_order_relaxed);
    while (cur > 0) {
        if (_neon_wants.compare_exchange_weak(cur, cur - 1,
                std::memory_order_relaxed)) return true;
    }
    return false;
}

// ── NEON batch pre-filter ───────────────────────────────────────────
// Trial-divides each candidate by small primes {3,5,7,11,13,17,19,23}
// using NEON: processes 2 candidates at a time in uint64x2_t lanes.
// Returns indices of candidates that survive (not divisible by any).
// Eliminates ~75-80% of composites before they reach GPU/CPU.

std::vector<uint32_t> LoadBalancer::neon_prefilter(const uint64_t* cands,
                                                    uint32_t count) {
    // Small primes for trial division
    static const uint64_t sp[] = {3,5,7,11,13,17,19,23,29,31,37,41,43,47};
    static const int nsp = 14;

    std::vector<uint32_t> survivors;
    survivors.reserve(count / 3);  // ~30% survival expected

    uint32_t i = 0;

    // Process pairs with NEON
    for (; i + 2 <= count; i += 2) {
        uint64x2_t v = vld1q_u64(&cands[i]);
        bool alive0 = true, alive1 = true;

        for (int s = 0; s < nsp && (alive0 || alive1); s++) {
            uint64x2_t divisor = vdupq_n_u64(sp[s]);

            // Compute v / divisor (no NEON integer divide, use scalar)
            // But we can do modulo check: v - (v/d)*d == 0
            uint64_t c0 = vgetq_lane_u64(v, 0);
            uint64_t c1 = vgetq_lane_u64(v, 1);

            if (alive0 && c0 % sp[s] == 0 && c0 != sp[s]) alive0 = false;
            if (alive1 && c1 % sp[s] == 0 && c1 != sp[s]) alive1 = false;
        }

        if (alive0) survivors.push_back(i);
        if (alive1) survivors.push_back(i + 1);
    }

    // Scalar tail
    for (; i < count; i++) {
        bool alive = true;
        for (int s = 0; s < nsp && alive; s++) {
            if (cands[i] % sp[s] == 0 && cands[i] != sp[s]) alive = false;
        }
        if (alive) survivors.push_back(i);
    }

    return survivors;
}

// ── Metrics reporting ───────────────────────────────────────────────

void LoadBalancer::report(int task_id, uint32_t cand_rate, uint32_t gpu_q,
                          uint32_t cpu_q, uint32_t density) {
    if (task_id < 0 || task_id >= MAX_TASKS) return;
    auto& m = _metrics[task_id];
    m.cand_rate = (m.cand_rate * 3 + cand_rate) / 4;
    m.gpu_queued = gpu_q;
    m.cpu_queued = cpu_q;
    m.sieve_density = density;
}

// ── NEON-accelerated pressure scan ──────────────────────────────────

float LoadBalancer::neon_scan_gpu_pressure() const {
    uint32x4_t gpu_sum = vdupq_n_u32(0);
    const uint32_t* base = reinterpret_cast<const uint32_t*>(_metrics.data());
    for (int i = 0; i < MAX_TASKS; i++) {
        uint32x4_t v = vld1q_u32(base + i * 4);
        gpu_sum = vaddq_u32(gpu_sum, v);
    }
    uint32_t total_gpu_q = vgetq_lane_u32(gpu_sum, 1);
    uint32_t total_cpu_q = vgetq_lane_u32(gpu_sum, 2);
    uint32_t total = total_gpu_q + total_cpu_q;
    if (total == 0) return 0.0f;
    return (float)total_gpu_q / (float)total;
}

float LoadBalancer::neon_scan_cpu_pressure() const {
    uint32x4_t sum = vdupq_n_u32(0);
    const uint32_t* base = reinterpret_cast<const uint32_t*>(_metrics.data());
    for (int i = 0; i < MAX_TASKS; i++) {
        uint32x4_t v = vld1q_u32(base + i * 4);
        sum = vaddq_u32(sum, v);
    }
    uint32_t total_cpu_q = vgetq_lane_u32(sum, 2);
    uint32_t total_rate = vgetq_lane_u32(sum, 0);
    if (total_rate == 0) return 0.0f;
    return (float)total_cpu_q / (float)total_rate;
}

// ── Main advice function — simple: split in two ─────────────────────
//
// Base rule: 50/50 GPU/CPU split. Always.
// Pool signals only nudge ±10% — never dominate.
// NEON pre-filter always runs (it's cheap and helps both sides).

DispatchAdvice LoadBalancer::advise(int task_id) {
    DispatchAdvice advice;
    advice.gpu_ratio = 0.5f;       // DEFAULT: split in two
    advice.neon_ratio = 1.0f;      // ALWAYS pre-filter through NEON
    advice.throttle_gpu = false;

    // Check pool — only nudge, don't override
    bool gpu_hungry = gpu_wants_work();
    bool cpu_hungry = cpu_wants_work();
    (void)neon_wants_work();  // drain NEON tokens, it always runs

    if (gpu_hungry && !cpu_hungry) {
        advice.gpu_ratio = 0.60f;  // nudge 10% more to GPU
    } else if (cpu_hungry && !gpu_hungry) {
        advice.gpu_ratio = 0.40f;  // nudge 10% more to CPU
    }
    // both hungry or neither: stay at 0.5

    // Safety: if GPU is severely saturated, pull back
    float sat = gpu_saturation();
    if (sat > 0.85f) {
        advice.gpu_ratio = 0.25f;
        advice.throttle_gpu = true;
    } else if (sat > 0.70f) {
        advice.gpu_ratio = 0.35f;
    }

    return advice;
}

// ── Saturation readout ──────────────────────────────────────────────

float LoadBalancer::gpu_saturation() const {
    int64_t busy = _window_busy_us.load(std::memory_order_relaxed);
    int64_t ws = _window_start_us.load(std::memory_order_relaxed);
    int64_t elapsed = now_us_lb() - ws;
    if (elapsed <= 0) return 0.0f;
    float sat = (float)busy / (float)elapsed;
    if (sat < 0.0f) sat = 0.0f;
    if (sat > 1.0f) sat = 1.0f;
    return sat;
}

float LoadBalancer::cpu_pressure() const {
    return neon_scan_cpu_pressure();
}

} // namespace prime
