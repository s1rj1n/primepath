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

class MatrixSieve {
public:
    MatrixSieve();

    // Generate a composite mask for candidates [start, start+count).
    // result[i] = 1 if start+i passes all small-prime tests (likely prime candidate)
    // result[i] = 0 if start+i is divisible by any matrix prime (definitely composite)
    // Uses Accelerate framework for vectorized computation.
    void sieve_block(uint64_t start, uint32_t count, uint8_t *result) const;

    // Score a range [lo, hi] by expected prime density and residue quality.
    // Returns a score 0.0-1.0 where higher = more promising range.
    // Uses hill-climbing heuristic based on prime residue distribution.
    double score_range(uint64_t lo, uint64_t hi) const;

    // Find the most promising sub-range within [lo, hi] of given size.
    // Returns the start offset of the best sub-range.
    // Uses the scoring function with stride-based sampling.
    uint64_t best_subrange(uint64_t lo, uint64_t hi, uint64_t subrange_size) const;

    // Statistics
    double rejection_ratio() const; // fraction rejected by matrix sieve
    uint64_t total_tested() const { return _total_tested; }
    uint64_t total_rejected() const { return _total_rejected; }

private:
    // Pre-computed pattern for each prime: pattern_p[i] = (i % p != 0) ? 1 : 0
    std::vector<std::vector<float>> _patterns;

    // Combined pattern for full period (PATTERN_PERIOD_SMALL)
    std::vector<uint8_t> _combined_pattern;

    mutable uint64_t _total_tested = 0;
    mutable uint64_t _total_rejected = 0;

    void build_patterns();
    void build_combined_pattern();
};

} // namespace prime
