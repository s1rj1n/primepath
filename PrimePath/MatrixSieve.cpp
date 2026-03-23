#include "MatrixSieve.hpp"
#include <Accelerate/Accelerate.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <arm_neon.h>

namespace prime {

MatrixSieve::MatrixSieve() {
    build_patterns();
    build_combined_pattern();
    build_prefilled_tile();

    // Allocate 128-byte aligned working buffer for sieve_block_fast
    posix_memalign((void **)&_work_buf, 128, SIEVE_BUF_SIZE);
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
    // Combined pattern: marks composites for primes {2,3,5,7,11,13}
    // Period = lcm(2,3,5,7,11,13) = 30030
    _combined_pattern.resize(COMBINED_PERIOD);
    for (uint64_t i = 0; i < COMBINED_PERIOD; i++) {
        bool composite = (i % 2 == 0) || (i % 3 == 0) || (i % 5 == 0) ||
                          (i % 7 == 0) || (i % 11 == 0) || (i % 13 == 0);
        _combined_pattern[i] = composite ? 0 : 1;
    }
    // Mark 0 and 1 as non-prime
    _combined_pattern[0] = 0;
    _combined_pattern[1] = 0;
}

void MatrixSieve::build_prefilled_tile() {
    // Allocate 128-byte aligned buffer and tile the combined pattern across it.
    // This creates a "production line" template: any sieve_block call just
    // copies the relevant portion and applies the remaining few primes (17-31).
    posix_memalign((void **)&_prefilled_tile, 128, SIEVE_BUF_SIZE);

    // Tile the pattern using NEON bulk copy
    const uint8_t *pat = _combined_pattern.data();
    uint64_t filled = 0;

    // First: write one full period
    memcpy(_prefilled_tile, pat, COMBINED_PERIOD);
    filled = COMBINED_PERIOD;

    // Double-up: copy what we have to fill the rest (exponential fill)
    while (filled < SIEVE_BUF_SIZE) {
        uint64_t chunk = std::min(filled, (uint64_t)SIEVE_BUF_SIZE - filled);
        memcpy(_prefilled_tile + filled, _prefilled_tile, chunk);
        filled += chunk;
    }
}

// NEON-accelerated sieve using pre-filled pattern tile
void MatrixSieve::sieve_block(uint64_t start, uint32_t count, uint8_t *result) const {
    _total_tested += count;

    // Phase 1: Copy pre-filled pattern tile with correct alignment.
    // The tile has the combined pattern for primes {2,3,5,7,11,13} baked in.
    // We just need to align it to `start mod COMBINED_PERIOD`.
    uint64_t pat_offset = start % COMBINED_PERIOD;

    // Fast NEON copy from the pre-filled tile
    uint32_t copied = 0;

    // First chunk: from pat_offset to end of first period
    uint32_t first_chunk = (uint32_t)std::min((uint64_t)(COMBINED_PERIOD - pat_offset), (uint64_t)count);
    memcpy(result, _combined_pattern.data() + pat_offset, first_chunk);
    copied = first_chunk;

    // Remaining: tile from the start of the pattern
    while (copied < count) {
        uint32_t chunk = std::min((uint32_t)COMBINED_PERIOD, count - copied);
        memcpy(result + copied, _combined_pattern.data(), chunk);
        copied += chunk;
    }

    // Phase 2: Apply remaining primes {17, 19, 23, 29, 31} via marker-stride.
    // These have larger strides so fewer marks per prime — fast scalar loop.
    static constexpr uint64_t REMAINING_PRIMES[] = {17, 19, 23, 29, 31};
    static constexpr int NUM_REMAINING = 5;
    for (int pi = 0; pi < NUM_REMAINING; pi++) {
        uint64_t p = REMAINING_PRIMES[pi];
        uint64_t r = start % p;
        uint32_t marker = (r == 0) ? 0 : (uint32_t)(p - r);
        // Don't mark the prime itself
        if (start + marker == p) marker += (uint32_t)p;
        for (uint32_t j = marker; j < count; j += (uint32_t)p) {
            result[j] = 0;
        }
    }

    // Count rejections using NEON
    uint64_t rejected = 0;
    {
        uint32_t j = 0;
        uint32x4_t acc = vdupq_n_u32(0);
        for (; j + 16 <= count; j += 16) {
            uint8x16_t v = vld1q_u8(&result[j]);
            uint8x16_t zeros = vceqq_u8(v, vdupq_n_u8(0));
            uint8x16_t ones = vshrq_n_u8(zeros, 7);
            uint16x8_t sum16 = vpaddlq_u8(ones);
            uint32x4_t sum32 = vpaddlq_u16(sum16);
            acc = vaddq_u32(acc, sum32);
        }
        uint32_t lane_sum[4];
        vst1q_u32(lane_sum, acc);
        for (int k = 0; k < 4; k++) rejected += lane_sum[k];
        for (; j < count; j++) {
            if (!result[j]) rejected++;
        }
    }
    _total_rejected += rejected;
}

// Fast path: uses pre-allocated aligned buffer, avoids caller allocation
const uint8_t* MatrixSieve::sieve_block_fast(uint64_t start, uint32_t count) const {
    if (count > SIEVE_BUF_SIZE) count = SIEVE_BUF_SIZE;
    sieve_block(start, count, _work_buf);
    return _work_buf;
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
