#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

// ═══════════════════════════════════════════════════════════════════════
// EvenShadow — divisor interference pre-filter
//
// For any odd candidate p, both p-1 and p+1 are even. Their
// factorizations cast a "shadow" that reveals structural information:
//
// 1. Smoothness of p-1 controls Miller-Rabin behavior:
//    - High v₂(p-1): more room for strong pseudoprimes base 2
//    - B-smooth (p-1): Fermat test is definitive (Pocklington)
//
// 2. Smoothness of p+1 feeds Lucas-type tests:
//    - If p+1 has known factorization, Lucas test is definitive
//
// 3. Divisor interference pattern:
//    The overlap of divisors from p-1 and p+1 constrains p's
//    properties. Numbers where both neighbors are very smooth
//    (many small factors) are "well-constrained" — if they pass
//    primality tests, the result is more trustworthy.
//    Numbers where neighbors have large prime factors are
//    "loosely constrained" — pseudoprimes are more likely.
//
// 4. Carmichael susceptibility:
//    Carmichael numbers n require (p-1)|(n-1) for each factor p.
//    If n-1 is highly smooth, more factor combinations satisfy
//    Korselt's criterion → higher pseudoprime risk.
//
// Usage: score a batch of sieved candidates, then:
//   - GPU-test high-scoring (well-constrained) candidates first
//   - Flag low-scoring (loosely constrained) for extra verification
//   - Feed smoothness data to the pseudoprime predictor
// ═══════════════════════════════════════════════════════════════════════

namespace prime {

struct ShadowInfo {
    uint8_t  num_factors_minus;  // count of small prime factors in p-1
    uint8_t  num_factors_plus;   // count of small prime factors in p+1
    uint8_t  two_valuation;      // v₂(p-1) = largest k where 2^k | (p-1)
    uint8_t  score;              // composite score 0-255 (higher = more constrained)
    uint64_t cofactor_minus;     // p-1 after dividing out all small factors
    uint64_t cofactor_plus;      // p+1 after dividing out all small factors
};

class EvenShadow {
    static constexpr int NUM_SHADOW_PRIMES = 25;
    static constexpr uint64_t SHADOW_PRIMES[NUM_SHADOW_PRIMES] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    };

public:
    // Analyze a single candidate
    static ShadowInfo analyze(uint64_t p) {
        ShadowInfo info = {};
        if (p < 3) return info;

        uint64_t pm1 = p - 1;  // even
        uint64_t pp1 = p + 1;  // even

        // Factor p-1: extract powers of 2 first
        uint64_t m = pm1;
        while ((m & 1) == 0) {
            info.two_valuation++;
            m >>= 1;
        }
        info.num_factors_minus = info.two_valuation;

        // Trial divide p-1 by odd small primes
        for (int i = 1; i < NUM_SHADOW_PRIMES; i++) {  // skip 2, already done
            uint64_t sp = SHADOW_PRIMES[i];
            while (m % sp == 0) {
                m /= sp;
                info.num_factors_minus++;
            }
        }
        info.cofactor_minus = m;

        // Factor p+1
        m = pp1;
        while ((m & 1) == 0) {
            info.num_factors_plus++;
            m >>= 1;
        }
        for (int i = 1; i < NUM_SHADOW_PRIMES; i++) {
            uint64_t sp = SHADOW_PRIMES[i];
            while (m % sp == 0) {
                m /= sp;
                info.num_factors_plus++;
            }
        }
        info.cofactor_plus = m;

        // Score: higher = more constrained = more trustworthy if passes test
        //
        // Components:
        // - Total small factors (more = more constrained): 0-30 typical
        // - Cofactor smallness (smaller cofactor = more fully factored)
        // - Low two_valuation bonus (v₂=1 means p≡3 mod 4, fewer SPRP)
        // - Penalty for both cofactors being large (loosely constrained)

        int total_factors = info.num_factors_minus + info.num_factors_plus;

        // Cofactor score: log2(cofactor) penalty
        // Fully factored (cofactor=1): 64 points. Large cofactor: 0 points.
        int cf_score_m = (info.cofactor_minus <= 1) ? 64 :
                         64 - (int)std::min(63.0, log2((double)info.cofactor_minus));
        int cf_score_p = (info.cofactor_plus <= 1) ? 64 :
                         64 - (int)std::min(63.0, log2((double)info.cofactor_plus));

        // Two-valuation: v₂=1 is best (p≡3 mod 4), high v₂ is worst
        int v2_score = (info.two_valuation <= 1) ? 20 :
                       (info.two_valuation <= 3) ? 10 : 0;

        int raw = total_factors * 3 + cf_score_m + cf_score_p + v2_score;
        info.score = (uint8_t)std::min(255, std::max(0, raw));

        return info;
    }

    // Batch scoring: compute scores for all candidates
    // Returns scores[i] for candidates[i], higher = more constrained
    static void score_batch(const uint64_t* candidates, uint32_t count,
                           uint8_t* scores) {
        for (uint32_t i = 0; i < count; i++) {
            scores[i] = analyze(candidates[i]).score;
        }
    }

    // Batch scoring with full info
    static void analyze_batch(const uint64_t* candidates, uint32_t count,
                             ShadowInfo* infos) {
        for (uint32_t i = 0; i < count; i++) {
            infos[i] = analyze(candidates[i]);
        }
    }

    // Reorder candidates by score (descending — highest priority first)
    // Returns a permutation vector: result[0] is the index of the
    // highest-scoring candidate.
    static std::vector<uint32_t> priority_order(const uint64_t* candidates,
                                                 uint32_t count) {
        std::vector<uint8_t> scores(count);
        score_batch(candidates, count, scores.data());

        std::vector<uint32_t> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&scores](uint32_t a, uint32_t b) {
                      return scores[a] > scores[b];  // descending
                  });
        return indices;
    }

    // Reorder candidates in-place by score (highest priority first)
    // Also returns the score threshold: candidates below this score
    // should get extra pseudoprime verification.
    static uint8_t reorder_inplace(std::vector<uint64_t>& candidates) {
        if (candidates.empty()) return 0;

        uint32_t n = (uint32_t)candidates.size();
        std::vector<uint8_t> scores(n);
        score_batch(candidates.data(), n, scores.data());

        // Build (score, value) pairs and sort
        std::vector<std::pair<uint8_t, uint64_t>> scored(n);
        for (uint32_t i = 0; i < n; i++) {
            scored[i] = {scores[i], candidates[i]};
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) {
                      return a.first > b.first;
                  });

        // Compute suspicion threshold: bottom 10% by score
        uint8_t threshold = scored[n * 9 / 10].first;

        // Write back sorted
        for (uint32_t i = 0; i < n; i++) {
            candidates[i] = scored[i].second;
        }
        return threshold;
    }

    // Quick batch stats for logging
    struct BatchStats {
        uint32_t count;
        double avg_score;
        uint32_t fully_factored;   // both cofactors == 1
        uint32_t suspicious;       // score below threshold
        uint8_t  min_score;
        uint8_t  max_score;
    };

    static BatchStats compute_stats(const uint64_t* candidates, uint32_t count,
                                     uint8_t threshold = 50) {
        BatchStats s = {count, 0.0, 0, 0, 255, 0};
        if (count == 0) return s;

        uint64_t total = 0;
        for (uint32_t i = 0; i < count; i++) {
            auto info = analyze(candidates[i]);
            total += info.score;
            if (info.cofactor_minus == 1 && info.cofactor_plus == 1)
                s.fully_factored++;
            if (info.score < threshold) s.suspicious++;
            if (info.score < s.min_score) s.min_score = info.score;
            if (info.score > s.max_score) s.max_score = info.score;
        }
        s.avg_score = (double)total / count;
        return s;
    }
};

} // namespace prime
