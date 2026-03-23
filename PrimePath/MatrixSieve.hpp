#pragma once
#include <cstdint>
#include <vector>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════
// MatrixSieve — uses matrix operations (Accelerate/ANE) for fast
// multi-prime composite rejection across candidate blocks.
//
// Concept: For each small prime p, create a p-element pattern vector
// where entry[i] = 0 if i ≡ 0 (mod p), else 1.
// For a block of N candidates starting at offset s, compute:
//   mask[j] = pattern_p[(s+j) mod p] for each prime p
// Then multiply all masks element-wise: result[j] = 0 → composite.
//
// The matrix form: reshape the block into a k×k grid, apply
// convolution-like patterns. Accelerate framework auto-dispatches
// to ANE/GPU for the matrix ops.
//
// Additionally: hill-climbing search guidance using residue scoring
// to prioritize ranges with higher expected prime density.
// ═══════════════════════════════════════════════════════════════════════

namespace prime {

// Small primes used for matrix sieve patterns
constexpr uint64_t MATRIX_PRIMES[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
constexpr int NUM_MATRIX_PRIMES = 10;

// Pre-computed LCM of matrix primes for pattern repetition
// lcm(3,5,7,11,13) = 15015, lcm(all 10) = 223092870
constexpr uint64_t PATTERN_PERIOD_SMALL = 15015;       // primes 3,5,7,11,13
constexpr uint64_t PATTERN_PERIOD_FULL  = 223092870;   // all 10 primes

// Pre-allocated sieve buffer size (reused across calls, avoids heap allocation)
constexpr uint32_t SIEVE_BUF_SIZE = 2 * 1024 * 1024;  // 2M — fits in L2 cache

class MatrixSieve {
public:
    MatrixSieve();

    // Generate a composite mask for candidates [start, start+count).
    // result[i] = 1 if start+i passes all small-prime tests (likely prime candidate)
    // result[i] = 0 if start+i is divisible by any matrix prime (definitely composite)
    // Uses NEON + pre-filled pattern tile for zero-setup sieve blocks.
    void sieve_block(uint64_t start, uint32_t count, uint8_t *result) const;

    // Fast sieve using pre-allocated aligned buffer (avoids caller allocation).
    // Returns pointer to internal buffer valid until next call. count <= SIEVE_BUF_SIZE.
    const uint8_t* sieve_block_fast(uint64_t start, uint32_t count) const;

    // Score a range [lo, hi] by expected prime density and residue quality.
    double score_range(uint64_t lo, uint64_t hi) const;

    // Find the most promising sub-range within [lo, hi] of given size.
    uint64_t best_subrange(uint64_t lo, uint64_t hi, uint64_t subrange_size) const;

    // Statistics
    double rejection_ratio() const;
    uint64_t total_tested() const { return _total_tested; }
    uint64_t total_rejected() const { return _total_rejected; }

private:
    // Pre-computed pattern for each prime
    std::vector<std::vector<float>> _patterns;

    // Combined pattern: lcm(2,3,5,7,11,13) = 30030 entries, pre-marks all composites
    // for primes 2..13. Tiled at init time into a large pre-filled buffer.
    std::vector<uint8_t> _combined_pattern;
    static constexpr uint64_t COMBINED_PERIOD = 30030; // lcm(2,3,5,7,11,13)

    // Pre-filled tile: SIEVE_BUF_SIZE bytes, pre-tiled with the combined pattern.
    // sieve_block copies this then applies remaining primes (17,19,23,29,31).
    uint8_t *_prefilled_tile = nullptr; // 128-byte aligned for NEON

    // Pre-allocated working buffer for sieve_block_fast (128-byte aligned)
    mutable uint8_t *_work_buf = nullptr;

    mutable uint64_t _total_tested = 0;
    mutable uint64_t _total_rejected = 0;

    void build_patterns();
    void build_combined_pattern();
    void build_prefilled_tile();
};

} // namespace prime
