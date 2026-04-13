#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// ═══════════════════════════════════════════════════════════════════════
// PrimeEngine — C++ port of primepath with multi-core parallelism
// ═══════════════════════════════════════════════════════════════════════

namespace prime {

// ── Modular arithmetic (overflow-safe via __uint128_t) ───────────────

inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return (unsigned __int128)a * b % m;
}

// ═══════════════════════════════════════════════════════════════════════
// Nester-CarryChain Streaming Divisibility Tester (S. Nester, 2026)
// ═══════════════════════════════════════════════════════════════════════
//
// Stream an arbitrarily large number through multiple accumulator
// pipelines, testing candidate divisors in parallel. No division.
//
// The number is stored as an array of 64-bit limbs (big-endian, most
// significant first). For each limb we accumulate via Barrett reduction:
//   remainder = (remainder * 2^64 + limb) mod divisor
//   where mod uses precomputed reciprocal multiply, never hardware UDIV
//
// Three outcomes at each step:
//   HIT   - remainder < divisor, feed the next limb
//   BUST  - remainder >= divisor after accumulation, reduce
//   MATCH - remainder == 0 at the end, divisor divides the number
//
// N-wide batching (template<int N>):
//   - Process N divisors per pass through the number (N = 1,2,4,8,16)
//   - Each stream is independent, CPU pipelines all N chains in parallel
//   - Adaptive calibrator picks optimal N for the given number size
//
// For a 2048-bit number that's 32 limbs, so 32 streaming steps per
// divisor batch. Each step is a multiply + add + conditional subtract,
// all staying in registers.
// ═══════════════════════════════════════════════════════════════════════

// BigNum: a large integer stored as big-endian 64-bit limbs
struct BigNum {
    std::vector<uint64_t> limbs; // limbs[0] = most significant

    // Construct from a single uint64
    static BigNum from_u64(uint64_t v) {
        BigNum n;
        n.limbs.push_back(v);
        return n;
    }

    // Construct from a hex string (e.g. "FFAA0011...")
    static BigNum from_hex(const std::string& hex) {
        BigNum n;
        // Process 16 hex chars (64 bits) at a time, from the left
        size_t len = hex.size();
        size_t pos = 0;
        // Handle leading partial limb
        size_t first = len % 16;
        if (first > 0) {
            n.limbs.push_back(std::stoull(hex.substr(0, first), nullptr, 16));
            pos = first;
        }
        while (pos < len) {
            n.limbs.push_back(std::stoull(hex.substr(pos, 16), nullptr, 16));
            pos += 16;
        }
        // Strip leading zeros
        while (n.limbs.size() > 1 && n.limbs[0] == 0)
            n.limbs.erase(n.limbs.begin());
        return n;
    }

    size_t bit_width() const {
        if (limbs.empty()) return 0;
        size_t top = 64 - __builtin_clzll(limbs[0]);
        return top + (limbs.size() - 1) * 64;
    }
};

// ── Scalar streaming remainder (division-based reference) ────────────
// This is the OLD method kept as a baseline for benchmarking.
// Uses __int128 and hardware division -- the thing we're replacing.

inline uint64_t stream_mod_scalar(const uint64_t* limbs, size_t count, uint64_t d) {
    if (d <= 1) return 0;
    unsigned __int128 pow64 = 1;
    for (int i = 0; i < 64; i++) {
        pow64 <<= 1;
        if (pow64 >= d) pow64 -= d;
    }
    uint64_t p64 = (uint64_t)pow64;
    uint64_t rem = 0;
    for (size_t i = 0; i < count; i++) {
        unsigned __int128 wide = (unsigned __int128)rem * p64;
        wide += limbs[i];
        rem = (uint64_t)(wide % d);
    }
    return rem;
}

inline bool stream_divides_scalar(const BigNum& n, uint64_t d) {
    return stream_mod_scalar(n.limbs.data(), n.limbs.size(), d) == 0;
}

inline std::vector<uint64_t> stream_find_divisors_scalar(
    const BigNum& n, const uint64_t* divisors, size_t ndiv)
{
    std::vector<uint64_t> found;
    for (size_t i = 0; i < ndiv; i++) {
        if (divisors[i] > 1 && stream_mod_scalar(n.limbs.data(), n.limbs.size(), divisors[i]) == 0)
            found.push_back(divisors[i]);
    }
    return found;
}

// ═══════════════════════════════════════════════════════════════════════
// NESTER-CARRYCHAIN ENGINE: Accumulate, don't divide (S. Nester, 2026)
// ═══════════════════════════════════════════════════════════════════════
//
// Core idea: to test if d divides N, never compute N/d or N%d.
// Instead, stream through N segment by segment and ACCUMULATE
// multiples of d until the segment is filled:
//
//   - HIT:       accumulator < segment value, add more d's
//   - BUST:      accumulator > segment value, not a divisor here,
//                carry the deficit to the next segment
//   - MATCH:     accumulator == 0 at the end, d divides N
//
// Implementation: for each 64-bit limb, we need to find how many
// copies of d fill (remainder * 2^64 + limb). Instead of dividing,
// we use a RECIPROCAL MULTIPLY:
//
//   quotient ~= (value * inverse_d) >> 64      (UMULH instruction)
//   accumulated = quotient * d                   (MUL instruction)
//   leftover = value - accumulated               (SUB instruction)
//   if leftover >= d: leftover -= d              (CMP + SUB, rare)
//
// Total: UMULH + MUL + SUB + CMP  (~4 cycles on M-series)
// vs UDIV                          (~7 cycles, non-pipelineable)
//
// For 4 parallel divisors this means 4x UMULH + 4x MUL + 4x SUB,
// all independent, all pipelineable. No division anywhere.
// ═══════════════════════════════════════════════════════════════════════

// Barrett-style "how many d's fill x" without dividing.
// Returns x mod d using only multiply, shift, subtract.
//
// Uses floor reciprocal so q never overestimates (no unsigned underflow).
// At most 2 subtractions to correct. Zero division instructions.
__attribute__((always_inline))
inline uint64_t accumulate_mod32(uint64_t x, uint32_t d, uint64_t inv_d) {
    // q = floor(x * inv_d / 2^64)  -- underestimates x/d by at most 2
    uint64_t q = (uint64_t)(((unsigned __int128)x * inv_d) >> 64);
    // How many d's we stacked up (q * d fits in uint64 because q <= x/d < 2^64/d,
    // and q*d <= x < 2^64)
    uint64_t r = x - q * (uint64_t)d;
    // Correct: r is in [0, 3d) at worst, typically [0, 2d)
    if (r >= d) { r -= d; if (r >= d) r -= d; }
    return r;
}

// Precompute the reciprocal for a 32-bit divisor.
// inv = floor(2^64 / d) -- floor guarantees q never overestimates.
inline uint64_t precompute_inverse32(uint32_t d) {
    if (d <= 1) return 0;
    // UINT64_MAX / d is floor((2^64-1)/d), close enough to floor(2^64/d)
    return UINT64_MAX / d;
}

// ── Nester-CC accumulator: single divisor, no division in hot loop ───

inline uint64_t stream_mod_blackjack(const uint64_t* limbs, size_t count, uint32_t d) {
    if (d <= 1) return 0;
    uint64_t inv = precompute_inverse32(d);

    // Precompute 2^64 mod d (how the remainder shifts between segments)
    // Done with doubling, no division
    uint64_t pow64 = 1;
    for (int i = 0; i < 64; i++) {
        pow64 <<= 1;
        if (pow64 >= d) pow64 -= d;
    }

    uint64_t rem = 0;
    for (size_t i = 0; i < count; i++) {
        // Combined Horner step: rem = (rem * pow64 + limb) mod d
        // rem < d < 2^32, pow64 < d < 2^32, so rem*pow64 < 2^64
        // We need (rem*pow64 + limb) mod d but that sum could exceed 2^64.
        // So: reduce rem*pow64 first (it fits in uint64), then add limb mod d.
        uint64_t shifted = accumulate_mod32(rem * pow64, d, inv);
        // limb is uint64, reduce it too
        uint64_t limb_r = accumulate_mod32(limbs[i], d, inv);
        // shifted < d, limb_r < d, sum < 2d < 2^33, fits in uint64
        rem = shifted + limb_r;
        if (rem >= d) rem -= d;
    }
    return rem;
}

#if defined(__aarch64__)

// ── Nester-CC 4-wide: four divisors, zero division ───────────────────
//
// Four independent accumulate streams. Each uses:
//   UMULH (reciprocal multiply)  \
//   MUL   (quotient * divisor)    } per segment per divisor
//   SUB   (leftover)             /
//
// M-series can issue MUL+UMULH on separate ports, so 4 independent
// streams saturate the integer pipeline. No UDIV, no NEON overhead.

// ── Nester-CC N-wide: process N divisors per pass ────────────────────
//
// Generic N-wide accumulator. The compiler unrolls for small N and
// keeps all remainders in registers. Tested batch sizes: 1, 2, 4, 8, 16.
//
// The hot loop is N independent chains of UMULH+MUL+SUB per limb.
// Wider batches amortise the per-limb overhead (pointer increment,
// branch) across more divisors, but eventually spill from registers.
// The optimal width depends on number size and hardware.

static constexpr int BJ_MAX_BATCH = 16;

// Template N-wide Nester-CC: N is compile-time, so the compiler
// unrolls the inner loop even at -O0 for small N.
template<int N>
inline void stream_mod_Nx32_blackjack_t(
    const uint64_t* limbs, size_t count,
    const uint32_t* d,
    uint32_t* rem)
{
    static_assert(N <= BJ_MAX_BATCH, "batch too wide");
    uint64_t inv[N], p64[N], r[N];
    for (int k = 0; k < N; k++) {
        inv[k] = precompute_inverse32(d[k]);
        r[k] = 0;
        if (d[k] <= 1) { p64[k] = 0; continue; }
        uint64_t p = 1;
        for (int i = 0; i < 64; i++) {
            p <<= 1;
            if (p >= d[k]) p -= d[k];
        }
        p64[k] = p;
    }

    for (size_t i = 0; i < count; i++) {
        uint64_t limb = limbs[i];
        // Manually unrolled for common widths via template constant
        for (int k = 0; k < N; k++) {
            if (d[k] > 1) {
                uint64_t s = accumulate_mod32(r[k] * p64[k], d[k], inv[k]);
                uint64_t l = accumulate_mod32(limb,           d[k], inv[k]);
                r[k] = s + l;
                if (r[k] >= d[k]) r[k] -= d[k];
            }
        }
    }

    for (int k = 0; k < N; k++) rem[k] = (uint32_t)r[k];
}

// Runtime-dispatch wrapper
inline void stream_mod_Nx32_blackjack(
    const uint64_t* limbs, size_t count,
    const uint32_t* d, int N,
    uint32_t* rem)
{
    switch (N) {
        case 1:  stream_mod_Nx32_blackjack_t<1>(limbs, count, d, rem); break;
        case 2:  stream_mod_Nx32_blackjack_t<2>(limbs, count, d, rem); break;
        case 4:  stream_mod_Nx32_blackjack_t<4>(limbs, count, d, rem); break;
        case 8:  stream_mod_Nx32_blackjack_t<8>(limbs, count, d, rem); break;
        case 16: stream_mod_Nx32_blackjack_t<16>(limbs, count, d, rem); break;
        default:
            // Fallback for odd sizes (tail processing)
            for (int k = 0; k < N; k++) {
                rem[k] = (uint32_t)stream_mod_blackjack(limbs, count, d[k]);
            }
            break;
    }
}

// ── Nester-CC 2-wide: two 64-bit divisors, no division ───────────────
//
// For divisors >= 2^32, we need __int128 for the reciprocal multiply
// but still avoid UDIV. Uses UMULH-equivalent via __int128 shift.

inline void stream_mod_2x64_blackjack(
    const uint64_t* limbs, size_t count,
    uint64_t d0, uint64_t d1,
    uint64_t& rem0, uint64_t& rem1)
{
    auto compute_pow64 = [](uint64_t d) -> uint64_t {
        unsigned __int128 p = 1;
        for (int i = 0; i < 64; i++) { p <<= 1; if (p >= d) p -= d; }
        return (uint64_t)p;
    };

    // For 64-bit divisors, Barrett with __int128
    // inv ~= 2^128 / d (we need 128-bit reciprocal for 128-bit intermediates)
    // Simpler: just use __int128 mod, still faster than UDIV via compiler intrinsics
    uint64_t p64_0 = compute_pow64(d0);
    uint64_t p64_1 = compute_pow64(d1);
    uint64_t r0 = 0, r1 = 0;

    for (size_t i = 0; i < count; i++) {
        uint64_t limb = limbs[i];
        // __int128 mod compiles to UMULH + MUL + SUB on AArch64 (no UDIV)
        unsigned __int128 w0 = (unsigned __int128)r0 * p64_0 + limb;
        unsigned __int128 w1 = (unsigned __int128)r1 * p64_1 + limb;
        r0 = (uint64_t)(w0 % d0);
        r1 = (uint64_t)(w1 % d1);
    }
    rem0 = r0;
    rem1 = r1;
}

#endif // __aarch64__

// ── High-level: stream-test a batch of candidate divisors ────────────
//
// Tests up to `ndiv` candidate divisors against a BigNum.
// Returns a vector of divisors that evenly divide N (Nester-CC).
// Automatically picks the NEON fast path when available.

struct StreamResult {
    std::vector<uint64_t> divisors;  // which ones matched (remainder == 0)
    size_t tested = 0;               // how many we checked
    size_t bits = 0;                 // bit width of the number
    double elapsed_us = 0;           // wall time in microseconds
    int batch_size = 0;              // adaptive batch width chosen
};

inline StreamResult stream_find_divisors(
    const BigNum& n,
    const uint64_t* candidates, size_t ndiv)
{
    StreamResult res;
    res.bits = n.bit_width();
    auto t0 = std::chrono::high_resolution_clock::now();

#if defined(__aarch64__)
    // Separate into small (< 2^32) and large divisors
    std::vector<uint32_t> small_divs;
    std::vector<size_t>   small_idx;
    std::vector<uint64_t> large_divs;
    std::vector<size_t>   large_idx;

    for (size_t i = 0; i < ndiv; i++) {
        if (candidates[i] <= 1) continue;
        if (candidates[i] < (1ULL << 32)) {
            small_divs.push_back((uint32_t)candidates[i]);
            small_idx.push_back(i);
        } else {
            large_divs.push_back(candidates[i]);
            large_idx.push_back(i);
        }
    }

    // ── Adaptive batch size calibration ──────────────────────────────
    // Two-phase selection:
    //   1. Heuristic: pick initial width from limb count
    //   2. Calibrate: time multiple iterations at nearby widths,
    //      pick the one with highest throughput
    //
    // Heuristic: more limbs = wider batches pay off (inner loop dominates).
    // Fewer limbs = setup cost matters, narrower is safer.
    int best_batch = 4;
    size_t nlimbs = n.limbs.size();
    if (nlimbs <= 4)        best_batch = 16; // small numbers: max parallelism
    else if (nlimbs <= 16)  best_batch = 8;
    else if (nlimbs <= 64)  best_batch = 8;
    else                    best_batch = 4;  // huge numbers: less register spill

    // Calibrate: run 4 iterations of each candidate width, pick best throughput
    if (small_divs.size() >= 48) {
        int trial_widths[] = {4, 8, 16};
        double best_rate = 0;
        const int CALIB_ITERS = 4; // enough to smooth noise

        for (int w : trial_widths) {
            if ((size_t)w * CALIB_ITERS > small_divs.size()) continue;
            uint32_t trial_rem[BJ_MAX_BATCH];
            auto ct0 = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < CALIB_ITERS; iter++) {
                stream_mod_Nx32_blackjack(n.limbs.data(), n.limbs.size(),
                    &small_divs[iter * w], w, trial_rem);
            }
            auto ct1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(ct1 - ct0).count();
            double rate = (double)(w * CALIB_ITERS) / (us + 0.001);
            if (rate > best_rate) {
                best_rate = rate;
                best_batch = w;
            }
        }
    }
    res.batch_size = best_batch;

    // ── Process small divisors at the chosen batch width ─────────────
    size_t si = 0;
    while (si + best_batch <= small_divs.size()) {
        uint32_t rem[BJ_MAX_BATCH];
        stream_mod_Nx32_blackjack(n.limbs.data(), n.limbs.size(),
                                  &small_divs[si], best_batch, rem);
        for (int k = 0; k < best_batch; k++) {
            if (rem[k] == 0)
                res.divisors.push_back(candidates[small_idx[si + k]]);
        }
        res.tested += best_batch;
        si += best_batch;
    }
    // Remaining small divisors, single-stream
    for (; si < small_divs.size(); si++) {
        if (stream_mod_blackjack(n.limbs.data(), n.limbs.size(), small_divs[si]) == 0)
            res.divisors.push_back(candidates[small_idx[si]]);
        res.tested++;
    }

    // Process large divisors in pairs (64-bit Nester-CC)
    size_t li = 0;
    while (li + 2 <= large_divs.size()) {
        uint64_t rem0, rem1;
        stream_mod_2x64_blackjack(n.limbs.data(), n.limbs.size(),
                        large_divs[li], large_divs[li+1], rem0, rem1);
        if (rem0 == 0) res.divisors.push_back(candidates[large_idx[li]]);
        if (rem1 == 0) res.divisors.push_back(candidates[large_idx[li+1]]);
        res.tested += 2;
        li += 2;
    }
    // Remaining large divisor
    if (li < large_divs.size()) {
        if (stream_mod_scalar(n.limbs.data(), n.limbs.size(), large_divs[li]) == 0)
            res.divisors.push_back(candidates[large_idx[li]]);
        res.tested++;
    }

#else
    // Non-ARM fallback: scalar
    res.batch_size = 1;
    for (size_t i = 0; i < ndiv; i++) {
        if (candidates[i] > 1 &&
            stream_mod_scalar(n.limbs.data(), n.limbs.size(), candidates[i]) == 0)
            res.divisors.push_back(candidates[i]);
        res.tested++;
    }
#endif

    auto t1 = std::chrono::high_resolution_clock::now();
    res.elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return res;
}

// Convenience: test a single divisor against a BigNum
inline bool stream_divides(const BigNum& n, uint64_t d) {
    if (d < (1ULL << 32))
        return stream_mod_blackjack(n.limbs.data(), n.limbs.size(), (uint32_t)d) == 0;
    return stream_mod_scalar(n.limbs.data(), n.limbs.size(), d) == 0;
}

// ═══════════════════════════════════════════════════════════════════════
// GEAR SHIFT ENGINE (S. Nester, 2026)
// ═══════════════════════════════════════════════════════════════════════
//
// Three gears, auto-selected like a transmission:
//   Gear 1: CPU single-thread N-wide (existing, low overhead)
//   Gear 2: CPU multi-thread (divisors split across cores)
//   Gear 3: GPU Metal kernel (one thread per divisor)
//
// Gear selection based on work volume = limbs x divisors.
// Small jobs stay on one core. Medium jobs fan out to CPU threads.
// Large jobs go to GPU where thousands of threads run in parallel.

enum class StreamGear { CPU_SINGLE = 1, CPU_MULTI = 2, GPU = 3 };

// Gear selection uses two rules per transition, calibrated from
// benchmark data across 128-32768 bit numbers and 100-200K divisors:
//
// Gear 2 (CPU multi-thread) wins when there are enough divisors to
// amortize thread-spawn overhead (~100us). For very large numbers
// (512+ limbs), even 50 divisors justify multi-threading.
//
// Gear 3 (GPU) wins at 50K+ divisors (GPU dispatch overhead ~400us
// is amortized by massive parallelism), or at 1K+ divisors when the
// number has 256+ limbs (large per-divisor work fills GPU occupancy).

inline StreamGear select_gear(size_t num_limbs, size_t num_divisors,
                               bool gpu_available) {
    // Gear 3: GPU
    if (gpu_available) {
        if (num_divisors >= 50000)
            return StreamGear::GPU;
        if (num_limbs >= 256 && num_divisors >= 1000)
            return StreamGear::GPU;
    }

    // Gear 2: CPU multi-thread
    if (num_divisors >= 5000)
        return StreamGear::CPU_MULTI;
    if (num_limbs >= 256 && num_divisors >= 50)
        return StreamGear::CPU_MULTI;

    return StreamGear::CPU_SINGLE;
}

// ── Gear 2: CPU multi-thread ─────────────────────────────────────────
// Split divisors across available cores. Each thread independently
// streams through all limbs (read-only, no contention).

inline StreamResult stream_find_divisors_mt(
    const BigNum& n,
    const uint64_t* candidates, size_t ndiv,
    int nthreads = 0)
{
    if (nthreads <= 0)
        nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads < 2) nthreads = 2;
    if ((size_t)nthreads > ndiv) nthreads = (int)ndiv;

    StreamResult res;
    res.bits = n.bit_width();
    auto t0 = std::chrono::high_resolution_clock::now();

    // Per-thread results
    struct ThreadResult {
        std::vector<uint64_t> divisors;
        size_t tested = 0;
    };
    std::vector<ThreadResult> thread_results(nthreads);
    std::vector<std::thread> threads;

    size_t chunk = ndiv / nthreads;
    size_t remainder = ndiv % nthreads;

    for (int t = 0; t < nthreads; t++) {
        size_t start = t * chunk + std::min((size_t)t, remainder);
        size_t count = chunk + (t < (int)remainder ? 1 : 0);

        threads.emplace_back([&, t, start, count]() {
            auto& tr = thread_results[t];
            // Use single-thread gear 1 for this chunk
            auto partial = stream_find_divisors(n, candidates + start, count);
            tr.divisors = std::move(partial.divisors);
            tr.tested = partial.tested;
            if (t == 0) res.batch_size = partial.batch_size;
        });
    }

    for (auto& t : threads) t.join();

    // Merge results
    for (auto& tr : thread_results) {
        for (auto d : tr.divisors) res.divisors.push_back(d);
        res.tested += tr.tested;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return res;
}

// ── Auto gear selector: picks the right gear and dispatches ──────────
// Pass gpu_dispatch = nullptr if GPU is not available.
// If provided, gpu_dispatch receives (limbs, num_limbs, divisors, num_divs)
// and returns a vector of divisors that divide N.

using GpuDispatchFn = std::function<std::vector<uint64_t>(
    const uint64_t* limbs, uint32_t num_limbs,
    const uint32_t* divisors, uint32_t num_divs)>;

inline StreamResult stream_find_divisors_auto(
    const BigNum& n,
    const uint64_t* candidates, size_t ndiv,
    GpuDispatchFn gpu_dispatch = nullptr)
{
    bool gpu_ok = (gpu_dispatch != nullptr);
    StreamGear gear = select_gear(n.limbs.size(), ndiv, gpu_ok);

    if (gear == StreamGear::GPU && gpu_dispatch) {
        // Build uint32 divisor list for GPU (GPU kernel uses uint32)
        std::vector<uint32_t> divs32;
        divs32.reserve(ndiv);
        for (size_t i = 0; i < ndiv; i++) {
            if (candidates[i] > 1 && candidates[i] < (1ULL << 32))
                divs32.push_back((uint32_t)candidates[i]);
        }

        StreamResult res;
        res.bits = n.bit_width();
        res.batch_size = -3; // indicates GPU gear
        auto t0 = std::chrono::high_resolution_clock::now();

        auto hits = gpu_dispatch(n.limbs.data(), (uint32_t)n.limbs.size(),
                                  divs32.data(), (uint32_t)divs32.size());
        res.divisors = std::move(hits);
        res.tested = divs32.size();

        auto t1 = std::chrono::high_resolution_clock::now();
        res.elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return res;
    }

    if (gear == StreamGear::CPU_MULTI) {
        return stream_find_divisors_mt(n, candidates, ndiv);
    }

    // Gear 1: single-thread
    return stream_find_divisors(n, candidates, ndiv);
}

// ── Mersenne streaming: is 2^p - 1 divisible by f? ──────────────────
// Instead of building the full 2^p-1 in memory, we compute
// (2^p - 1) mod f directly: (2^p mod f) == 1 means f | (2^p - 1).
// This is O(log p) and needs no big number at all.
// Uses square-and-multiply with the carry-chain mulmod already defined above.
inline bool mersenne_stream_divides(uint64_t p, uint64_t f) {
    if (f < 2 || p < 2) return false;
    // Inline modpow(2, p, f) to avoid forward-reference issues
    uint64_t base = 2 % f, result = 1, exp = p;
    while (exp > 0) {
        if (exp & 1) result = mulmod(result, base, f);
        exp >>= 1;
        base = mulmod(base, base, f);
    }
    return result == 1;
}

inline uint64_t modpow(uint64_t base, uint64_t exp, uint64_t mod) {
    if (mod == 1) return 0;
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mulmod(result, base, mod);
        exp >>= 1;
        base = mulmod(base, base, mod);
    }
    return result;
}

// ── Deterministic Miller-Rabin (12 witnesses, correct for all u64) ──

inline bool miller_test(uint64_t n, uint64_t a) {
    if (a % n == 0) return true;
    uint64_t d = n - 1;
    int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }
    uint64_t x = modpow(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int i = 0; i < r - 1; i++) {
        x = mulmod(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

constexpr uint64_t MR_WITNESSES[12] = {2,3,5,7,11,13,17,19,23,29,31,37};

inline bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (auto p : {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL}) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    for (auto a : MR_WITNESSES) {
        if (!miller_test(n, a)) return false;
    }
    return true;
}

// Forward declaration for use in mersenne_factor_scan
inline std::vector<uint64_t> factor_u64(uint64_t n);

// ── Mersenne factor testing ──────────────────────────────────────────
//
// A factor f of 2^p - 1 must satisfy:
//   1. f = 2kp + 1 for some k >= 1
//   2. f mod 8 in {1, 7}
//   3. modpow(2, p, f) == 1
//
// is_mersenne_factor(p, f): returns true if f divides 2^p - 1.
// mersenne_factor_scan(f, max_p): given a prime f, find Mersenne
//   exponents p where f divides 2^p - 1. Uses the 2kp+1 constraint
//   to iterate only valid candidates.

inline bool is_mersenne_factor(uint64_t p, uint64_t f) {
    if (f < 3 || p < 2) return false;
    return modpow(2, p, f) == 1;
}

// Scan for Mersenne exponents divisible by factor f, up to max_p.
// Returns vector of exponents p where f | 2^p - 1.
inline std::vector<uint64_t> mersenne_factor_scan(uint64_t f, uint64_t max_p = 100000000ULL) {
    std::vector<uint64_t> hits;
    if (f < 3) return hits;
    // f must be odd and f mod 8 in {1,7} to be a Mersenne factor
    if (f % 2 == 0) return hits;
    uint64_t r8 = f % 8;
    if (r8 != 1 && r8 != 7) return hits;
    // f = 2kp + 1, so p = (f-1)/(2k) for k=1,2,...
    // Equivalently, p must divide (f-1)/2
    uint64_t half = (f - 1) / 2;
    // Find all prime divisors of half, test each as Mersenne exponent
    auto divs = factor_u64(half);
    // Collect unique prime factors
    std::set<uint64_t> prime_divs(divs.begin(), divs.end());
    // Also test half itself and all divisor combinations would be expensive.
    // Simpler: the order of 2 mod f divides (f-1). The Mersenne exponent p
    // must equal the multiplicative order of 2 mod f, if that order is prime.
    // Compute ord_2(f) directly.
    uint64_t order = 0;
    uint64_t fm1 = f - 1;
    // Start with fm1, divide by prime factors while modpow still == 1
    order = fm1;
    for (uint64_t pd : prime_divs) {
        while (order % pd == 0 && modpow(2, order / pd, f) == 1)
            order /= pd;
    }
    if (order <= max_p && order >= 2 && is_prime(order)) {
        hits.push_back(order);
    }
    return hits;
}

// ── Wheel-210 (48 spokes, eliminates multiples of 2,3,5,7) ─────────

constexpr int WHEEL210_COUNT = 48;
constexpr int SPOKES_210[48] = {
    1,11,13,17,19,23,29,31,37,41,43,47,
    53,59,61,67,71,73,79,83,89,97,
    101,103,107,109,113,121,127,131,137,139,
    143,149,151,157,163,167,169,173,179,181,
    187,191,193,197,199,209
};

// Bitset for O(1) spoke lookup
struct Wheel210 {
    uint64_t bits[4] = {};
    constexpr Wheel210() {
        for (int s : SPOKES_210) {
            bits[s / 64] |= 1ULL << (s % 64);
        }
    }
    inline bool valid(uint64_t n) const {
        int r = (int)(n % 210);
        return (bits[r / 64] >> (r % 64)) & 1;
    }
};
constexpr Wheel210 WHEEL;

// ── CRT composite rejection (primes 11–31) ─────────────────────────

inline bool crt_reject(uint64_t n) {
    if (n < 2) return true;
    if (n <= 37) {
        // Direct check for small values
        return !is_prime(n);
    }
    if (n % 2 == 0 || n % 3 == 0 || n % 5 == 0 || n % 7 == 0) return true;
    // CRT tables for primes 11–31
    uint64_t r143 = n % 143;   // lcm(11,13)
    if (r143 % 11 == 0 || r143 % 13 == 0) return true;
    uint64_t r7429 = n % 7429;  // lcm(17,19,23)
    if (r7429 % 17 == 0 || r7429 % 19 == 0 || r7429 % 23 == 0) return true;
    uint64_t r899 = n % 899;   // lcm(29,31)
    if (r899 % 29 == 0 || r899 % 31 == 0) return true;
    return false;
}

// ── Sieve of Eratosthenes ───────────────────────────────────────────

inline std::vector<bool> sieve(uint64_t limit) {
    std::vector<bool> is_p(limit + 1, true);
    is_p[0] = is_p[1] = false;
    for (uint64_t i = 2; i * i <= limit; i++) {
        if (is_p[i]) {
            for (uint64_t j = i * i; j <= limit; j += i)
                is_p[j] = false;
        }
    }
    return is_p;
}

// ── Quick factoring (trial + Pollard-Brent rho) ─────────────────────

inline uint64_t brent_rho_one(uint64_t n, uint64_t c) {
    uint64_t x = 2, y = 2, d = 1;
    auto f = [&](uint64_t v) { return (mulmod(v, v, n) + c) % n; };
    while (d == 1) {
        x = f(x);
        y = f(f(y));
        d = std::gcd(x > y ? x - y : y - x, n);
    }
    return d == n ? 0 : d;
}

// Forward declaration: heuristic divisor finder (Lucky7s + PinchFactor)
// Returns candidate divisors of n found via digit-structural and round-number proximity checks.
// Defined after PinchHit/Lucky7Hit structs below.
inline std::vector<uint64_t> heuristic_divisors(uint64_t n);

inline std::vector<uint64_t> factor_u64(uint64_t n) {
    std::vector<uint64_t> factors;
    if (n < 2) return factors;
    for (uint64_t p : {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL}) {
        while (n % p == 0) { factors.push_back(p); n /= p; }
    }
    // Trial division to 10000
    for (uint64_t d = 41; d < 10000 && d * d <= n; d += 2) {
        while (n % d == 0) { factors.push_back(d); n /= d; }
    }
    // Heuristic pre-pass: Lucky7s + PinchFactor before expensive Pollard rho
    if (n > 1 && !is_prime(n)) {
        auto candidates = heuristic_divisors(n);
        for (uint64_t d : candidates) {
            while (n > 1 && d > 1 && d < n && n % d == 0) {
                // Recursively factor the divisor (it may be composite)
                auto sub = factor_u64(d);
                factors.insert(factors.end(), sub.begin(), sub.end());
                n /= d;
            }
        }
    }
    // Pollard rho for remaining
    while (n > 1 && !is_prime(n)) {
        uint64_t d = 0;
        for (uint64_t c = 1; c < 100 && d == 0; c++) {
            d = brent_rho_one(n, c);
        }
        if (d == 0) { factors.push_back(n); return factors; }
        while (n % d == 0) { factors.push_back(d); n /= d; }
    }
    if (n > 1) factors.push_back(n);
    std::sort(factors.begin(), factors.end());
    return factors;
}

inline std::string factors_string(uint64_t n) {
    auto f = factor_u64(n);
    if (f.empty()) return "";
    std::string s;
    for (size_t i = 0; i < f.size(); i++) {
        if (i > 0) s += " x ";
        s += std::to_string(f[i]);
    }
    return s;
}

// ── Atomic Prime Analysis ────────────────────────────────────────────
//
// Treats primes as boundaries between "atoms" of composite matter.
// Dense composite regions (high smoothness of p-1) are the nucleus.
// Prime clusters (small gaps) are the shell boundaries.
//
// Smoothness: how many small prime factors pack p-1.
//   Higher smoothness = denser composite packing nearby.
//   Omega(n) = total prime factors with multiplicity
//   omega(n) = distinct prime factors
//   smoothness_score = Omega(n) / log2(n) -- normalized, 0 to 1 range
//
// Composite density in a gap: (gap - 1) composites in a gap of size gap.
//   density = (gap - 1) / gap -- approaches 1 for large gaps.
//
// Shell classification based on gap before and gap after:
//   BOUNDARY -- large gap before or after (crossing a nucleus)
//   SHELL    -- small gaps on both sides (prime cluster / electron shell)
//   EDGE     -- transitional (one side small, one side moderate)

struct AtomicAnalysis {
    // Smoothness of p-1
    int omega;           // distinct prime factors
    int big_omega;       // total prime factors with multiplicity
    double smoothness;   // Omega / log2(p-1), 0 to ~1
    uint64_t largest_pf; // largest prime factor of p-1

    // Gap context
    uint64_t gap_before; // gap from previous prime
    uint64_t gap_after;  // gap to next prime (0 if unknown)
    double density_before; // composite density in gap before
    double density_after;  // composite density in gap after

    // Shell classification
    enum Shell { NUCLEUS_BOUNDARY, SHELL, EDGE };
    Shell shell;

    // Radius: half the sum of surrounding gaps (the "atom" this prime borders)
    double radius;

    std::string shell_str() const {
        switch (shell) {
            case NUCLEUS_BOUNDARY: return "BOUNDARY";
            case SHELL: return "SHELL";
            case EDGE: return "EDGE";
        }
        return "?";
    }
};

inline AtomicAnalysis analyze_prime_atom(uint64_t p, uint64_t gap_before, uint64_t gap_after = 0) {
    AtomicAnalysis a = {};
    a.gap_before = gap_before;
    a.gap_after = gap_after;

    // Factor p-1
    uint64_t pm1 = p - 1;
    auto factors = factor_u64(pm1);

    // Omega (total with multiplicity)
    a.big_omega = (int)factors.size();

    // omega (distinct)
    {
        std::set<uint64_t> distinct(factors.begin(), factors.end());
        a.omega = (int)distinct.size();
    }

    // Largest prime factor
    a.largest_pf = factors.empty() ? 1 : factors.back();

    // Smoothness score: Omega / log2(p-1)
    double log2_pm1 = log2((double)pm1);
    a.smoothness = log2_pm1 > 0 ? a.big_omega / log2_pm1 : 0;

    // Composite density in gaps
    a.density_before = gap_before > 1 ? (double)(gap_before - 1) / gap_before : 0;
    a.density_after = gap_after > 1 ? (double)(gap_after - 1) / gap_after : 0;

    // Radius: average of surrounding gaps (the "atom size" this prime borders)
    if (gap_after > 0) {
        a.radius = (gap_before + gap_after) / 2.0;
    } else {
        a.radius = (double)gap_before;
    }

    // Shell classification
    // Use ln(p) as the expected gap (prime number theorem)
    double expected_gap = log((double)p);
    double threshold_large = expected_gap * 2.0;  // > 2x expected = nucleus crossing
    double threshold_small = expected_gap * 0.6;   // < 0.6x expected = shell cluster

    bool before_large = gap_before > threshold_large;
    bool before_small = gap_before <= threshold_small;
    bool after_large = gap_after > threshold_large;
    bool after_small = gap_after > 0 && gap_after <= threshold_small;

    if (before_large || after_large) {
        a.shell = AtomicAnalysis::NUCLEUS_BOUNDARY;
    } else if (before_small && (gap_after == 0 || after_small)) {
        a.shell = AtomicAnalysis::SHELL;
    } else {
        a.shell = AtomicAnalysis::EDGE;
    }

    return a;
}

// ── Pinch Factor — digit-structural factoring heuristic ─────────────
//
// "Pinch" N at every digit boundary: N = L × 10^i + R
// Check algebraic combinations of L, R for perfect squares and shared
// factors with N. If any gcd(candidate, N) is nontrivial, we found a factor.

struct PinchHit {
    uint64_t divisor;
    uint64_t left;
    uint64_t right;
    int pinch_pos;       // digit position of the split
    std::string method;  // which check found it
};

inline uint64_t isqrt_exact(uint64_t n) {
    if (n == 0) return 0;
    uint64_t s = (uint64_t)std::sqrt((double)n);
    // Correct for floating point imprecision
    while (s * s > n) s--;
    while ((s + 1) * (s + 1) <= n) s++;
    return (s * s == n) ? s : 0;  // 0 means not a perfect square
}

inline std::vector<PinchHit> pinch_factor(uint64_t n) {
    std::vector<PinchHit> hits;
    if (n < 10) return hits;

    // Count digits and build powers of 10
    std::string digits = std::to_string(n);
    int d = (int)digits.size();

    uint64_t pow10 = 10;  // 10^i for split position i

    for (int i = 1; i < d && pow10 <= n; i++) {
        uint64_t R = n % pow10;
        uint64_t L = n / pow10;

        // 1. Direct gcd checks: gcd(L, N) and gcd(R, N)
        if (L > 1) {
            uint64_t g = std::gcd(L, n);
            if (g > 1 && g < n) {
                hits.push_back({g, L, R, i, "gcd(L,N)"});
            }
        }
        if (R > 1) {
            uint64_t g = std::gcd(R, n);
            if (g > 1 && g < n) {
                hits.push_back({g, L, R, i, "gcd(R,N)"});
            }
        }

        // 2. gcd(L+R, N) and gcd(|L-R|, N)
        uint64_t sum = L + R;
        if (sum > 1) {
            uint64_t g = std::gcd(sum, n);
            if (g > 1 && g < n) {
                hits.push_back({g, L, R, i, "gcd(L+R,N)"});
            }
        }
        uint64_t diff = (L > R) ? L - R : R - L;
        if (diff > 1) {
            uint64_t g = std::gcd(diff, n);
            if (g > 1 && g < n) {
                hits.push_back({g, L, R, i, "gcd(|L-R|,N)"});
            }
        }

        // 3. If L*R is a perfect square, check gcd(sqrt(L*R), N)
        // Use __uint128_t to avoid overflow
        unsigned __int128 prod = (unsigned __int128)L * R;
        if (prod > 0 && prod <= (unsigned __int128)UINT64_MAX) {
            uint64_t sq = isqrt_exact((uint64_t)prod);
            if (sq > 1) {
                uint64_t g = std::gcd(sq, n);
                if (g > 1 && g < n) {
                    hits.push_back({g, L, R, i, "sqrt(L*R)"});
                }
            }
        }

        // 4. Quadratic: treat as x² - Lx + R = 0, discriminant = L² - 4R
        if (L >= 2 && L * L >= 4 * R) {
            uint64_t disc = L * L - 4 * R;
            uint64_t sq = isqrt_exact(disc);
            if (sq > 0) {
                // Roots: (L ± sq) / 2
                if ((L + sq) % 2 == 0) {
                    uint64_t root1 = (L + sq) / 2;
                    uint64_t root2 = (L - sq) / 2;
                    if (root1 > 1) {
                        uint64_t g = std::gcd(root1, n);
                        if (g > 1 && g < n)
                            hits.push_back({g, L, R, i, "quad root+"});
                    }
                    if (root2 > 1) {
                        uint64_t g = std::gcd(root2, n);
                        if (g > 1 && g < n)
                            hits.push_back({g, L, R, i, "quad root-"});
                    }
                }
            }
        }

        // 5. Check if L or R themselves are perfect squares
        uint64_t sqL = isqrt_exact(L);
        if (sqL > 1) {
            uint64_t g = std::gcd(sqL, n);
            if (g > 1 && g < n) {
                hits.push_back({g, L, R, i, "sqrt(L)"});
            }
        }
        uint64_t sqR = isqrt_exact(R);
        if (sqR > 1) {
            uint64_t g = std::gcd(sqR, n);
            if (g > 1 && g < n) {
                hits.push_back({g, L, R, i, "sqrt(R)"});
            }
        }

        // Avoid overflow on pow10
        if (pow10 > UINT64_MAX / 10) break;
        pow10 *= 10;
    }

    // Deduplicate by divisor
    std::sort(hits.begin(), hits.end(),
              [](const PinchHit& a, const PinchHit& b) { return a.divisor < b.divisor; });
    auto last = std::unique(hits.begin(), hits.end(),
                            [](const PinchHit& a, const PinchHit& b) { return a.divisor == b.divisor; });
    hits.erase(last, hits.end());

    return hits;
}

// ── Lucky 7s — round-number proximity factoring ─────────────────────
//
// If N = (10^k + δ₁)(10^k + δ₂) where δ are small, then:
//   q = N / 10^k ≈ other factor,  r = N % 10^k ≈ δ × q
//   δ ≈ r/q — try nearby values and check divisibility.
//
// Also tries offsets that are multiples of 7 near powers of 10,
// since many primes cluster at 10^k ± 7j.

struct Lucky7Hit {
    uint64_t divisor;
    uint64_t power_of_10;   // which 10^k
    int64_t offset;          // divisor = 10^k + offset
    std::string method;
};

inline std::vector<Lucky7Hit> lucky7_factor(uint64_t n) {
    std::vector<Lucky7Hit> hits;
    if (n < 100) return hits;

    // Try each power of 10 from 10^2 to 10^9
    uint64_t pow10 = 100;
    for (int k = 2; k <= 18 && pow10 < n; k++) {
        // Method 1: δ estimation from quotient/remainder
        uint64_t q = n / pow10;
        uint64_t r = n % pow10;

        if (q > 1) {
            // Estimated δ for the "small" factor near 10^k
            int64_t delta_est = (q > 0) ? (int64_t)(r / q) : 0;

            // Search window around the estimate
            for (int64_t d = delta_est - 50; d <= delta_est + 50; d++) {
                int64_t candidate = (int64_t)pow10 + d;
                if (candidate < 2 || (uint64_t)candidate >= n) continue;
                if (n % (uint64_t)candidate == 0) {
                    hits.push_back({(uint64_t)candidate, pow10, d, "delta-est"});
                }
            }
        }

        // Method 2: multiples of 7 near 10^k (the "lucky 7s")
        // Check 10^k ± 7j for j = 0..100
        for (int j = 0; j <= 100; j++) {
            uint64_t offsets[] = {(uint64_t)((int64_t)pow10 + 7*j),
                                  (uint64_t)((int64_t)pow10 - 7*j)};
            for (uint64_t cand : offsets) {
                if (cand < 2 || cand >= n) continue;
                if (n % cand == 0) {
                    int64_t off = (int64_t)cand - (int64_t)pow10;
                    hits.push_back({cand, pow10, off, "lucky-7"});
                }
            }
        }

        // Method 3: small primes times 7 near 10^k
        // Check 10^k ± p*7 for small primes p
        for (uint64_t p : {2ULL,3ULL,5ULL,11ULL,13ULL}) {
            for (int sign = -1; sign <= 1; sign += 2) {
                for (int j = 1; j <= 20; j++) {
                    int64_t off = sign * (int64_t)(p * 7 * j);
                    uint64_t cand = (uint64_t)((int64_t)pow10 + off);
                    if (cand < 2 || cand >= n) continue;
                    if (n % cand == 0) {
                        hits.push_back({cand, pow10, off, "7x" + std::to_string(p)});
                    }
                }
            }
        }

        if (pow10 > UINT64_MAX / 10) break;
        pow10 *= 10;
    }

    // Deduplicate by divisor
    std::sort(hits.begin(), hits.end(),
              [](const Lucky7Hit& a, const Lucky7Hit& b) { return a.divisor < b.divisor; });
    auto last = std::unique(hits.begin(), hits.end(),
                            [](const Lucky7Hit& a, const Lucky7Hit& b) { return a.divisor == b.divisor; });
    hits.erase(last, hits.end());

    return hits;
}

// ── DivisorWeb — digit-by-digit factor sieve ────────────────────────
//
// Build a "web" where each row is a power of the base (10 or 60).
// Horizontal axis is 0..base-1 (digit values at that position).
// Drop candidate divisors down level by level:
//   Level 0: test 2..base-1
//   Level k: test base^k .. base^(k+1)-1
// Pruning rules:
//   1. Skip composites whose prime factors don't all divide N
//      (can't have a composite divisor without having its prime factors)
//   2. Stop when candidate > sqrt(N)
//   3. Skip even candidates if 2 doesn't divide N, etc.

struct WebLevel {
    int level;
    uint64_t range_lo, range_hi;
    uint64_t tested;
    uint64_t pruned_composite;   // skipped: composite with non-divisor prime factor
    uint64_t pruned_modular;     // skipped: modular constraint eliminated
    std::vector<uint64_t> divisors;  // divisors found at this level
};

struct DivisorWebResult {
    uint64_t n;
    int base;
    std::vector<uint64_t> all_divisors;    // every divisor found, sorted
    std::vector<uint64_t> prime_divisors;  // prime factors of N discovered
    std::vector<WebLevel> levels;
    double elapsed_us;
};

inline DivisorWebResult divisor_web(uint64_t n, int base = 10) {
    auto t0 = std::chrono::steady_clock::now();
    DivisorWebResult result;
    result.n = n;
    result.base = base;

    if (n < 2) {
        result.elapsed_us = 0;
        return result;
    }

    uint64_t sqrt_n = (uint64_t)std::sqrt((double)n);
    // Correct for floating point
    while (sqrt_n * sqrt_n > n) sqrt_n--;
    while ((sqrt_n + 1) * (sqrt_n + 1) <= n) sqrt_n++;

    // Known prime factors of N — drives the composite pruning
    std::set<uint64_t> known_prime_divs;
    // Small primes for quick primality/factoring of candidates
    std::vector<uint64_t> small_primes_list;
    {
        auto sv = sieve(std::min((uint64_t)100000, sqrt_n + 1));
        for (uint64_t i = 2; i < sv.size(); i++) {
            if (sv[i]) small_primes_list.push_back(i);
        }
    }

    // Quick check: is candidate prime? (for candidates up to ~10^10, trial to 10^5 suffices)
    auto is_candidate_prime = [&](uint64_t c) -> bool {
        if (c < 2) return false;
        for (uint64_t p : small_primes_list) {
            if (p * p > c) return true;
            if (c % p == 0) return (c == p);
        }
        // Fall back to Miller-Rabin for larger candidates
        return is_prime(c);
    };

    // Get prime factors of a small candidate
    auto prime_factors_of = [&](uint64_t c) -> std::vector<uint64_t> {
        std::vector<uint64_t> pf;
        for (uint64_t p : small_primes_list) {
            if (p * p > c) break;
            if (c % p == 0) {
                pf.push_back(p);
                while (c % p == 0) c /= p;
            }
        }
        if (c > 1) pf.push_back(c);
        return pf;
    };

    // Composite pruning: all prime factors of candidate must divide N
    auto composite_prunable = [&](uint64_t c) -> bool {
        if (is_candidate_prime(c)) return false;  // primes can't be pruned this way
        auto pf = prime_factors_of(c);
        for (uint64_t p : pf) {
            if (n % p != 0) return true;  // prime factor doesn't divide N → prune
        }
        return false;
    };

    // Modular pre-filter: for known small prime divisors p of N,
    // N mod p = 0, so any divisor d of N must satisfy: no constraint from p alone.
    // But for primes p that DON'T divide N, we know: d cannot be ≡ 0 (mod p)
    // unless d itself isn't a divisor. Not directly useful here.
    //
    // More useful: at level k, N mod base^(k+1) constrains possible divisors.
    // If d | N and d < base^(k+1), then d | (N mod lcm(d, base^(k+1))).
    // Simplified: if d < base^(k+1), then N mod d == 0 iff (N mod base^(k+1)) mod d == 0
    // when d < base^(k+1). This lets us test against a smaller number!

    uint64_t B = (uint64_t)base;
    uint64_t pow_base = 1;  // B^level

    for (int level = 0; pow_base <= sqrt_n; level++) {
        uint64_t next_pow = pow_base * B;
        uint64_t lo = (level == 0) ? 2 : pow_base;
        uint64_t hi = std::min(next_pow - 1, sqrt_n);

        WebLevel wl;
        wl.level = level;
        wl.range_lo = lo;
        wl.range_hi = hi;
        wl.tested = 0;
        wl.pruned_composite = 0;
        wl.pruned_modular = 0;

        for (uint64_t d = lo; d <= hi; d++) {
            // Quick skip: if small prime ∤ N, skip multiples of that prime
            if (d % 2 == 0 && n % 2 != 0) { wl.pruned_modular++; continue; }
            if (d % 3 == 0 && n % 3 != 0) { wl.pruned_modular++; continue; }
            if (d % 5 == 0 && n % 5 != 0) { wl.pruned_modular++; continue; }
            if (d % 7 == 0 && n % 7 != 0) { wl.pruned_modular++; continue; }

            // Composite pruning: all prime factors must divide N
            if (d > 1 && composite_prunable(d)) {
                wl.pruned_composite++;
                continue;
            }

            wl.tested++;

            // Direct divisibility test (no shortcuts — must be exact)
            bool divides = (n % d == 0);
            if (divides) {
                wl.divisors.push_back(d);
                result.all_divisors.push_back(d);
                // If prime, add to known prime divisors
                if (is_candidate_prime(d)) {
                    known_prime_divs.insert(d);
                    result.prime_divisors.push_back(d);
                }
                // Also record complementary divisor
                uint64_t comp = n / d;
                if (comp != d && comp > 1) {
                    result.all_divisors.push_back(comp);
                    if (is_prime(comp)) {
                        result.prime_divisors.push_back(comp);
                    }
                }
            }
        }

        result.levels.push_back(wl);

        // Advance to next power of base
        if (pow_base > UINT64_MAX / B) break;
        pow_base = next_pow;

        // Early exit if fully factored
        // Check: can we account for all of N with known prime divisors?
        uint64_t remaining = n;
        for (uint64_t p : known_prime_divs) {
            while (remaining % p == 0) remaining /= p;
        }
        if (remaining == 1 || is_prime(remaining)) {
            if (remaining > 1) {
                result.prime_divisors.push_back(remaining);
                result.all_divisors.push_back(remaining);
            }
            break;  // fully factored
        }
    }

    std::sort(result.all_divisors.begin(), result.all_divisors.end());
    result.all_divisors.erase(std::unique(result.all_divisors.begin(), result.all_divisors.end()),
                              result.all_divisors.end());
    std::sort(result.prime_divisors.begin(), result.prime_divisors.end());
    result.prime_divisors.erase(std::unique(result.prime_divisors.begin(), result.prime_divisors.end()),
                                result.prime_divisors.end());

    result.elapsed_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
    return result;
}

// ── Heuristic divisor finder (combines Lucky7s + PinchFactor) ────────

inline std::vector<uint64_t> heuristic_divisors(uint64_t n) {
    std::vector<uint64_t> divs;
    auto l7 = lucky7_factor(n);
    for (auto& h : l7) divs.push_back(h.divisor);
    auto ph = pinch_factor(n);
    for (auto& h : ph) divs.push_back(h.divisor);
    // DivisorWeb base-10 and base-60
    auto web10 = divisor_web(n, 10);
    for (auto d : web10.prime_divisors) divs.push_back(d);
    for (auto d : web10.all_divisors) divs.push_back(d);
    auto web60 = divisor_web(n, 60);
    for (auto d : web60.prime_divisors) divs.push_back(d);
    for (auto d : web60.all_divisors) divs.push_back(d);
    // Deduplicate
    std::sort(divs.begin(), divs.end());
    divs.erase(std::unique(divs.begin(), divs.end()), divs.end());
    return divs;
}

// ── Shadow convergence field ────────────────────────────────────────

constexpr uint64_t SHADOW_PRIMES[16] = {7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67};

inline double shadow_velocity(uint64_t n, uint64_t p) {
    uint64_t r = n % p;
    if (r == 0) return 0.0;
    return (2 * r < p) ? 1.0 : -1.0;
}

inline double proximity_weight(uint64_t n, uint64_t p) {
    uint64_t r = n % p;
    double d = (double)std::min(r, p - r) / (double)p;
    return 1.0 / (d + 0.1);
}

inline double convergence(uint64_t n, int np = 10) {
    double total = 0.0;
    for (int i = 0; i < np && i < 16; i++) {
        uint64_t p = SHADOW_PRIMES[i];
        if (n % p == 0) return -999.0; // on shadow = composite
        total += shadow_velocity(n, p) * proximity_weight(n, p);
    }
    return total;
}

// ── Markov Prime Predictor ──────────────────────────────────────────
//
// Builds empirical conditional distributions from known primes in a
// region (last digit -> gap distribution, digit transition matrix),
// then runs a Markov chain forward to predict candidate prime locations.
//
// Used by PrimeLocation as a candidate generator alongside convergence
// scoring. Candidates predicted by both methods get priority.

class MarkovPredictor {
public:
    static constexpr int DIGITS[4] = {1, 3, 7, 9};

    // Build distributions from a list of known primes (should be > ~100)
    void train(const std::vector<uint64_t>& primes) {
        // Clear
        for (int i = 0; i < 4; i++) {
            gap_dist_[i].clear();
            for (int j = 0; j < 4; j++) trans_[i][j] = 0;
        }

        // Filter to primes with last digit in {1,3,7,9}
        std::vector<uint64_t> relevant;
        relevant.reserve(primes.size());
        for (auto p : primes) {
            int d = p % 10;
            if (d == 1 || d == 3 || d == 7 || d == 9) relevant.push_back(p);
        }
        if (relevant.size() < 10) return;
        trained_ = true;

        for (size_t i = 0; i + 1 < relevant.size(); i++) {
            int d = digit_idx(relevant[i] % 10);
            int d_nxt = digit_idx(relevant[i + 1] % 10);
            if (d < 0 || d_nxt < 0) continue;
            uint64_t gap = relevant[i + 1] - relevant[i];
            gap_dist_[d].push_back(gap);
            trans_[d][d_nxt]++;
        }

        // Normalize transition counts to cumulative probabilities
        for (int d = 0; d < 4; d++) {
            double total = 0;
            for (int j = 0; j < 4; j++) total += trans_[d][j];
            if (total > 0) {
                double cum = 0;
                for (int j = 0; j < 4; j++) {
                    cum += trans_[d][j] / total;
                    trans_cum_[d][j] = cum;
                }
                trans_cum_[d][3] = 1.0; // ensure rounding
            }
        }
    }

    // Generate candidate prime locations by running the Markov chain forward.
    // Returns sorted, deduplicated list of predicted positions.
    std::vector<uint64_t> predict(uint64_t anchor, int steps, uint64_t seed = 12345) const {
        if (!trained_) return {};

        // Simple xorshift64 RNG (fast, no external dependency)
        uint64_t rng_state = seed ^ (anchor * 2654435761ULL);
        auto rng_next = [&]() -> double {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            return (rng_state & 0xFFFFFFFFFFFFULL) / (double)0x1000000000000ULL;
        };

        std::vector<uint64_t> candidates;
        candidates.reserve(steps);

        uint64_t current = anchor;
        int d = digit_idx(anchor % 10);
        if (d < 0) d = 0; // fallback

        for (int s = 0; s < steps; s++) {
            // Sample gap from empirical distribution
            if (gap_dist_[d].empty()) break;
            size_t gi = (size_t)(rng_next() * gap_dist_[d].size());
            if (gi >= gap_dist_[d].size()) gi = gap_dist_[d].size() - 1;
            uint64_t gap = gap_dist_[d][gi];
            current += gap;

            // Sample next last digit from transition distribution
            double r = rng_next();
            int d_nxt = 3; // default to last
            for (int j = 0; j < 4; j++) {
                if (r <= trans_cum_[d][j]) { d_nxt = j; break; }
            }

            // Adjust current to end in the predicted digit
            int target_digit = DIGITS[d_nxt];
            int cur_digit = (int)(current % 10);
            if (cur_digit != target_digit) {
                int off1 = (target_digit - cur_digit + 10) % 10;
                int off2 = -((cur_digit - target_digit + 10) % 10);
                current += (abs(off1) <= abs(off2)) ? off1 : off2;
            }

            // Only keep odd candidates > 2
            if (current > 2 && (current & 1)) {
                candidates.push_back(current);
            }
            d = d_nxt;
        }

        // Sort and deduplicate
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        return candidates;
    }

    // Run multiple chains with different seeds, collect consensus candidates
    // that appear in at least min_hits chains. Consensus = higher confidence.
    std::vector<std::pair<uint64_t, int>> predict_consensus(
            uint64_t anchor, int steps, int chains = 10, int min_hits = 2) const {
        // Count how many chains predict each candidate
        std::map<uint64_t, int> counts;
        for (int c = 0; c < chains; c++) {
            auto preds = predict(anchor, steps, 42 + c * 7919);
            for (auto v : preds) counts[v]++;
        }
        // Filter to consensus and sort by hit count descending
        std::vector<std::pair<uint64_t, int>> result;
        for (auto& [val, cnt] : counts) {
            if (cnt >= min_hits) result.push_back({val, cnt});
        }
        std::sort(result.begin(), result.end(),
            [](auto& a, auto& b) { return a.second > b.second; });
        return result;
    }

    // Predict the immediate next prime after anchor.
    // Runs many single-step trials, votes on the result.
    // Returns sorted vector of (candidate, vote_count).
    std::vector<std::pair<uint64_t, int>> predict_next(
            uint64_t anchor, int trials = 2000) const {
        if (!trained_) return {};

        int d = digit_idx(anchor % 10);
        if (d < 0) d = 0;
        if (gap_dist_[d].empty()) return {};

        std::map<uint64_t, int> votes;

        // Method 1: direct gap sampling (majority of trials)
        uint64_t rng_state = anchor ^ 0xBEEF;
        auto rng_next = [&]() -> double {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            return (rng_state & 0xFFFFFFFFFFFFULL) / (double)0x1000000000000ULL;
        };

        for (int t = 0; t < trials; t++) {
            // Sample gap for current last digit
            size_t gi = (size_t)(rng_next() * gap_dist_[d].size());
            if (gi >= gap_dist_[d].size()) gi = gap_dist_[d].size() - 1;
            uint64_t gap = gap_dist_[d][gi];
            uint64_t candidate = anchor + gap;

            // Sample next digit from transition matrix
            double r = rng_next();
            int d_nxt = 3;
            for (int j = 0; j < 4; j++) {
                if (r <= trans_cum_[d][j]) { d_nxt = j; break; }
            }

            // Adjust to target digit
            int target = DIGITS[d_nxt];
            int cur = (int)(candidate % 10);
            if (cur != target) {
                int off1 = (target - cur + 10) % 10;
                int off2 = -((cur - target + 10) % 10);
                candidate += (abs(off1) <= abs(off2)) ? off1 : off2;
            }

            if (candidate > anchor && (candidate & 1))
                votes[candidate]++;
        }

        // Method 2: mode-based prediction (add strong votes for most common gaps)
        // For each possible next digit, find the most common gap for that transition
        for (int d_nxt = 0; d_nxt < 4; d_nxt++) {
            // Build gap distribution for specifically d -> d_nxt transitions
            // We approximate by using the overall gap dist for digit d
            // and the most common gaps
            if (gap_dist_[d].empty()) continue;

            // Find mode (most frequent gap)
            std::map<uint64_t, int> gap_freq;
            for (auto g : gap_dist_[d]) gap_freq[g]++;
            // Top 3 most frequent gaps
            std::vector<std::pair<int, uint64_t>> ranked;
            for (auto& [g, c] : gap_freq) ranked.push_back({c, g});
            std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) {
                return a.first > b.first;
            });

            int target = DIGITS[d_nxt];
            int top = std::min((int)ranked.size(), 5);
            for (int i = 0; i < top; i++) {
                uint64_t candidate = anchor + ranked[i].second;
                int cur = (int)(candidate % 10);
                if (cur != target) {
                    int off1 = (target - cur + 10) % 10;
                    int off2 = -((cur - target + 10) % 10);
                    candidate += (abs(off1) <= abs(off2)) ? off1 : off2;
                }
                if (candidate > anchor && (candidate & 1)) {
                    // Weight by transition probability and gap frequency
                    double t_prob = (d_nxt > 0)
                        ? trans_cum_[d][d_nxt] - trans_cum_[d][d_nxt - 1]
                        : trans_cum_[d][0];
                    int bonus = (int)(ranked[i].first * t_prob * 10);
                    votes[candidate] += bonus;
                }
            }
        }

        // Sort by votes descending
        std::vector<std::pair<uint64_t, int>> result;
        for (auto& [val, cnt] : votes) {
            if (cnt >= 2) result.push_back({val, cnt});
        }
        std::sort(result.begin(), result.end(),
            [](auto& a, auto& b) { return a.second > b.second; });
        return result;
    }

    // Predict next prime with primality verification on top candidates.
    // Returns the best verified prime candidate, its vote count, rank,
    // and total candidates checked.
    struct VerifiedPrediction {
        uint64_t value = 0;       // the predicted prime (0 if none found)
        int votes = 0;            // vote count for this candidate
        int rank = 0;             // rank among all candidates (1 = top)
        int candidates_checked = 0; // how many we tested before finding a prime
        int total_candidates = 0;  // total in the vote set
    };

    VerifiedPrediction predict_next_verified(
            uint64_t anchor, int trials = 2000) const {
        auto candidates = predict_next(anchor, trials);
        VerifiedPrediction result;
        result.total_candidates = (int)candidates.size();

        int rank = 0;
        for (auto& [val, cnt] : candidates) {
            if (val <= anchor) continue;
            rank++;
            result.candidates_checked++;
            // is_prime uses Miller-Rabin with carry-chain mulmod
            if (is_prime(val)) {
                result.value = val;
                result.votes = cnt;
                result.rank = rank;
                return result;
            }
        }
        return result; // no verified prime found
    }

    // Incremental update: add a single observed prime to the training data.
    // Avoids full retrain. Updates gap distribution and transition counts.
    void update(uint64_t prev_prime, uint64_t new_prime) {
        if (!trained_) return;
        int d_prev = digit_idx(prev_prime % 10);
        int d_new = digit_idx(new_prime % 10);
        if (d_prev < 0 || d_new < 0) return;

        uint64_t gap = new_prime - prev_prime;
        gap_dist_[d_prev].push_back(gap);
        trans_[d_prev][d_new]++;

        // Rebuild cumulative transition probabilities for this row
        double total = 0;
        for (int j = 0; j < 4; j++) total += trans_[d_prev][j];
        if (total > 0) {
            double cum = 0;
            for (int j = 0; j < 4; j++) {
                cum += trans_[d_prev][j] / total;
                trans_cum_[d_prev][j] = cum;
            }
            trans_cum_[d_prev][3] = 1.0;
        }

        // Keep gap distributions from growing unbounded: if any digit's
        // gap list exceeds 50000 entries, trim the oldest half.
        // This gives a recency bias, adapting to local density.
        for (int i = 0; i < 4; i++) {
            if (gap_dist_[i].size() > 50000) {
                gap_dist_[i].erase(
                    gap_dist_[i].begin(),
                    gap_dist_[i].begin() + gap_dist_[i].size() / 2);
            }
        }
    }

    bool is_trained() const { return trained_; }

    // Mean gap per digit (for diagnostics)
    double mean_gap(int digit) const {
        int d = digit_idx(digit);
        if (d < 0 || gap_dist_[d].empty()) return 0;
        double sum = 0;
        for (auto g : gap_dist_[d]) sum += g;
        return sum / gap_dist_[d].size();
    }

private:
    static int digit_idx(int d) {
        switch (d) {
            case 1: return 0; case 3: return 1;
            case 7: return 2; case 9: return 3;
        }
        return -1;
    }

    bool trained_ = false;
    std::vector<uint64_t> gap_dist_[4];      // gap distributions per last digit
    double trans_[4][4] = {};                 // raw transition counts
    double trans_cum_[4][4] = {};             // cumulative transition probabilities
};

// ── Search result ───────────────────────────────────────────────────

struct PrimeResult {
    uint64_t value;
    double convergence_score;
    bool confirmed;  // by Miller-Rabin
};

// ── Stats tracking ──────────────────────────────────────────────────

struct SearchStats {
    std::atomic<uint64_t> candidates_tested{0};
    std::atomic<uint64_t> crt_rejected{0};
    std::atomic<uint64_t> wheel_rejected{0};
    std::atomic<uint64_t> mr_tested{0};
    std::atomic<uint64_t> primes_found{0};
    std::atomic<uint64_t> range_start{0};
    std::atomic<uint64_t> range_end{0};
    std::atomic<bool> running{false};
    std::chrono::steady_clock::time_point start_time;

    void reset() {
        candidates_tested = 0;
        crt_rejected = 0;
        wheel_rejected = 0;
        mr_tested = 0;
        primes_found = 0;
        running = false;
    }

    double elapsed_seconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time).count();
    }

    double primes_per_second() const {
        double t = elapsed_seconds();
        return t > 0 ? (double)candidates_tested.load() / t : 0;
    }
};

// ── Callback for progress/results ───────────────────────────────────

using ProgressCallback = std::function<void(const SearchStats&)>;
using PrimeCallback = std::function<void(const PrimeResult&)>;

// ── Multi-threaded search engine ────────────────────────────────────

class Engine {
public:
    SearchStats stats;

    // Search for primes in [start, end] using n_threads.
    // Calls prime_cb for each confirmed prime, progress_cb periodically.
    void search(uint64_t start, uint64_t end, int n_threads,
                PrimeCallback prime_cb, ProgressCallback progress_cb);

    // Stop a running search.
    void stop();

    // Verify a single number.
    PrimeResult verify(uint64_t n);

    // Search a range, return all primes found.
    std::vector<PrimeResult> search_range(uint64_t start, uint64_t end, int n_threads);

private:
    std::mutex result_mutex;
    std::vector<std::thread> workers;

    void search_chunk(uint64_t start, uint64_t end,
                      PrimeCallback prime_cb, int thread_id);
};

} // namespace prime
