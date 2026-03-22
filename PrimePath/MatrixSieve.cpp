#include "MatrixSieve.hpp"
#include <Accelerate/Accelerate.h>
#include <cstring>
#include <algorithm>
#include <arm_neon.h>

namespace prime {

MatrixSieve::MatrixSieve() {
    build_patterns();
    build_combined_pattern();
}

void MatrixSieve::build_patterns() {
    _patterns.resize(NUM_MATRIX_PRIMES);
    for (int pi = 0; pi < NUM_MATRIX_PRIMES; pi++) {
        uint64_t p = MATRIX_PRIMES[pi];
        _patterns[pi].resize(p);
        for (uint64_t i = 0; i < p; i++) {
            _patterns[pi][i] = (i == 0) ? 0.0f : 1.0f;
        }
    }
}

void MatrixSieve::build_combined_pattern() {
    _combined_pattern.resize(PATTERN_PERIOD_SMALL);
    for (uint64_t i = 0; i < PATTERN_PERIOD_SMALL; i++) {
        bool passes = true;
        if (i % 3 == 0 || i % 5 == 0 || i % 7 == 0 || i % 11 == 0 || i % 13 == 0)
            passes = false;
        _combined_pattern[i] = passes ? 1 : 0;
    }
}

// NEON-accelerated sieve: uses 128-bit SIMD for bulk AND operations
// and vectorized even-number clearing.
void MatrixSieve::sieve_block(uint64_t start, uint32_t count, uint8_t *result) const {
    _total_tested += count;

    // Phase 1: Initialize all candidates as potentially prime
    // NEON: fill 16 bytes at a time with 1s
    {
        uint8x16_t ones = vdupq_n_u8(1);
        uint32_t i = 0;
        for (; i + 16 <= count; i += 16) {
            vst1q_u8(&result[i], ones);
        }
        for (; i < count; i++) {
            result[i] = 1;
        }
    }

    // Phase 2: Marker-stride for ALL primes including 2
    // For each prime p: compute first multiple in [start, start+count),
    // then stride by p, marking composites with result[j] = 0.
    // Only touches O(count/p) positions per prime.
    //
    // Total work: count/2 + count/3 + count/5 + ... ≈ count × 1.67
    // vs old approach: count × 10 (per-element modulo) + count NEON even pass
    //
    // Visualization for p=7, start=100, count=20:
    //   r = 100 % 7 = 2  →  marker = 7 - 2 = 5
    //   Mark: j=5, j=12, j=19  (only 3 writes for 20 elements)
    //
    static constexpr uint64_t ALL_PRIMES[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
    static constexpr int NUM_ALL = 11;
    for (int pi = 0; pi < NUM_ALL; pi++) {
        uint64_t p = ALL_PRIMES[pi];
        uint64_t r = start % p;
        uint32_t marker = (r == 0) ? 0 : (uint32_t)(p - r);
        for (uint32_t j = marker; j < count; j += (uint32_t)p) {
            result[j] = 0;
        }
    }

    // Count rejections (zero bytes) using NEON
    uint64_t rejected = 0;
    {
        uint32_t j = 0;
        uint32x4_t acc = vdupq_n_u32(0);
        for (; j + 16 <= count; j += 16) {
            uint8x16_t v = vld1q_u8(&result[j]);
            // vceqq_u8 gives 0xFF (255) for zero bytes, 0 for non-zero.
            // Negate to get 1 per zero byte: ~0xFF = 0x00, ~0x00 = 0xFF → wrong.
            // Instead, use the fact that -0xFF = 1 in u8 (wrapping): 0 - 0xFF = 1.
            // Simpler: right-shift by 7 to convert 0xFF→1, 0x00→0, then sum.
            uint8x16_t zeros = vceqq_u8(v, vdupq_n_u8(0));
            uint8x16_t ones = vshrq_n_u8(zeros, 7);  // 0xFF>>7 = 1, 0>>7 = 0
            uint16x8_t sum16 = vpaddlq_u8(ones);
            uint32x4_t sum32 = vpaddlq_u16(sum16);
            acc = vaddq_u32(acc, sum32);
        }
        uint32_t lane_sum[4];
        vst1q_u32(lane_sum, acc);
        for (int k = 0; k < 4; k++) rejected += lane_sum[k];

        // Scalar remainder
        for (; j < count; j++) {
            if (!result[j]) rejected++;
        }
    }
    _total_rejected += rejected;
}

double MatrixSieve::score_range(uint64_t lo, uint64_t hi) const {
    if (hi <= lo) return 0.0;
    uint64_t range = hi - lo;

    double expected_ratio = 1.0;
    for (int i = 0; i < NUM_MATRIX_PRIMES; i++) {
        expected_ratio *= (1.0 - 1.0 / MATRIX_PRIMES[i]);
    }
    expected_ratio *= 0.5;

    const uint32_t sample_size = 1000;
    uint8_t sample[1000];
    uint64_t step = range / 10;
    if (step < sample_size) step = sample_size;

    double total_score = 0.0;
    int samples = 0;
    for (uint64_t s = lo; s + sample_size <= hi && samples < 10; s += step, samples++) {
        sieve_block(s, sample_size, sample);
        uint32_t survivors = 0;
        for (uint32_t i = 0; i < sample_size; i++) survivors += sample[i];
        double actual_ratio = (double)survivors / sample_size;
        double deviation = fabs(actual_ratio - expected_ratio) / expected_ratio;
        total_score += 1.0 / (1.0 + deviation);
    }

    if (samples == 0) return 0.5;
    return total_score / samples;
}

uint64_t MatrixSieve::best_subrange(uint64_t lo, uint64_t hi, uint64_t subrange_size) const {
    if (hi - lo <= subrange_size) return lo;

    uint64_t best_start = lo;
    double best_score = -1.0;
    uint64_t stride = (hi - lo) / 20;
    if (stride < subrange_size) stride = subrange_size;

    for (uint64_t s = lo; s + subrange_size <= hi; s += stride) {
        double score = score_range(s, s + subrange_size);
        if (score > best_score) {
            best_score = score;
            best_start = s;
        }
    }
    return best_start;
}

double MatrixSieve::rejection_ratio() const {
    if (_total_tested == 0) return 0.0;
    return (double)_total_rejected / _total_tested;
}

} // namespace prime
