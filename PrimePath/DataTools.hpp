#pragma once
#include "PrimeEngine.hpp"
#include <vector>
#include <cmath>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <utility>
#include <string>

namespace prime {

// ═══════════════════════════════════════════════════════════════════════
// Data Tools — Large-scale computational tools beyond prime search
// ═══════════════════════════════════════════════════════════════════════

// ── GCD / LCM ────────────────────────────────────────────────────────

inline uint64_t gcd(uint64_t a, uint64_t b) {
    while (b) { uint64_t t = b; b = a % b; a = t; }
    return a;
}

inline uint64_t lcm(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0;
    return a / gcd(a, b) * b;
}

// ── Extended GCD ─────────────────────────────────────────────────────

inline int64_t extended_gcd(int64_t a, int64_t b, int64_t& x, int64_t& y) {
    if (a == 0) { x = 0; y = 1; return b; }
    int64_t x1, y1;
    int64_t g = extended_gcd(b % a, a, x1, y1);
    x = y1 - (b / a) * x1;
    y = x1;
    return g;
}

// ── Modular Inverse ──────────────────────────────────────────────────

inline uint64_t mod_inverse(uint64_t a, uint64_t m) {
    int64_t x, y;
    int64_t g = extended_gcd((int64_t)a, (int64_t)m, x, y);
    if (g != 1) return 0;
    return (uint64_t)(((x % (int64_t)m) + (int64_t)m) % (int64_t)m);
}

// ── Euler Totient ────────────────────────────────────────────────────

inline uint64_t euler_totient(uint64_t n) {
    if (n <= 1) return n;
    auto factors = factor_u64(n);
    uint64_t result = n;
    uint64_t prev = 0;
    for (auto p : factors) {
        if (p != prev) {
            result = result / p * (p - 1);
            prev = p;
        }
    }
    return result;
}

// ── Multiplicative Order ─────────────────────────────────────────────
// Smallest k > 0 such that a^k ≡ 1 (mod m)

inline uint64_t multiplicative_order(uint64_t a, uint64_t m) {
    if (m <= 1) return 0;
    if (gcd(a, m) != 1) return 0;
    uint64_t phi = euler_totient(m);
    auto phi_factors = factor_u64(phi);
    std::vector<uint64_t> distinct;
    uint64_t prev = 0;
    for (auto f : phi_factors) {
        if (f != prev) { distinct.push_back(f); prev = f; }
    }
    uint64_t order = phi;
    for (auto p : distinct) {
        while (order % p == 0 && modpow(a, order / p, m) == 1) {
            order /= p;
        }
    }
    return order;
}

// ── Primitive Root ───────────────────────────────────────────────────
// Smallest primitive root modulo p (p must be prime)

inline uint64_t primitive_root(uint64_t p) {
    if (p <= 1) return 0;
    if (p == 2) return 1;
    uint64_t phi = p - 1;
    auto factors = factor_u64(phi);
    std::vector<uint64_t> distinct;
    uint64_t prev = 0;
    for (auto f : factors) {
        if (f != prev) { distinct.push_back(f); prev = f; }
    }
    for (uint64_t g = 2; g < p; g++) {
        bool is_root = true;
        for (auto q : distinct) {
            if (modpow(g, phi / q, p) == 1) { is_root = false; break; }
        }
        if (is_root) return g;
    }
    return 0;
}

// ── Quadratic Residue (Euler criterion) ──────────────────────────────

inline bool is_quadratic_residue(uint64_t a, uint64_t p) {
    if (a % p == 0) return true;
    return modpow(a, (p - 1) / 2, p) == 1;
}

// ── Tonelli-Shanks Square Root mod p ─────────────────────────────────

inline uint64_t sqrt_mod(uint64_t a, uint64_t p) {
    if (a == 0) return 0;
    if (p <= 1) return 0;
    if (p == 2) return a & 1;
    a = a % p;
    if (a == 0) return 0;
    if (!is_quadratic_residue(a, p)) return 0;

    uint64_t q = p - 1, s = 0;
    while (q % 2 == 0) { q /= 2; s++; }
    if (s == 1) return modpow(a, (p + 1) / 4, p);

    uint64_t z = 2;
    while (z < p && is_quadratic_residue(z, p)) z++;
    if (z >= p) return 0;

    uint64_t M = s;
    uint64_t c = modpow(z, q, p);
    uint64_t t = modpow(a, q, p);
    uint64_t R = modpow(a, (q + 1) / 2, p);

    for (int iter = 0; iter < 200; iter++) {
        if (t == 1) return R;
        uint64_t i = 0, temp = t;
        while (temp != 1 && i < M) { temp = mulmod(temp, temp, p); i++; }
        if (i >= M) return 0; // should not happen for valid inputs
        uint64_t b = c;
        for (uint64_t j = 0; j + i + 1 < M; j++) b = mulmod(b, b, p);
        M = i;
        c = mulmod(b, b, p);
        t = mulmod(t, c, p);
        R = mulmod(R, b, p);
    }
    return 0; // failsafe
}

// ── Baby-Step Giant-Step Discrete Logarithm ──────────────────────────
// Finds x such that g^x ≡ h (mod p), returns -1 if not found

inline int64_t baby_step_giant_step(uint64_t g, uint64_t h, uint64_t p) {
    if (p <= 1) return -1;
    uint64_t m = (uint64_t)std::ceil(std::sqrt((double)p));
    if (m == 0) m = 1;

    std::unordered_map<uint64_t, uint64_t> table;
    uint64_t val = 1;
    for (uint64_t j = 0; j < m; j++) {
        table[val] = j;
        val = mulmod(val, g, p);
    }

    uint64_t g_inv = modpow(g, p - 2, p);
    uint64_t factor = modpow(g_inv, m, p);

    val = h;
    for (uint64_t i = 0; i < m; i++) {
        auto it = table.find(val);
        if (it != table.end()) {
            return (int64_t)(i * m + it->second);
        }
        val = mulmod(val, factor, p);
    }
    return -1;
}

// ── B-Smooth Number Testing ──────────────────────────────────────────

inline bool is_b_smooth(uint64_t n, uint64_t B) {
    if (n <= 1) return true;
    auto factors = factor_u64(n);
    for (auto f : factors) {
        if (f > B) return false;
    }
    return true;
}

inline std::vector<uint64_t> enumerate_smooth(uint64_t lo, uint64_t hi, uint64_t B,
                                               uint64_t max_results = 100000) {
    std::vector<uint64_t> result;
    for (uint64_t n = lo; n <= hi && result.size() < max_results; n++) {
        if (is_b_smooth(n, B)) result.push_back(n);
    }
    return result;
}

// ── Batch GCD (Weak Key Detection) ───────────────────────────────────

struct WeakKeyResult {
    uint64_t modulus;
    uint64_t shared_factor;
    uint64_t other_modulus;
};

inline std::vector<WeakKeyResult> batch_gcd_audit(const std::vector<uint64_t>& moduli) {
    std::vector<WeakKeyResult> weak;
    for (size_t i = 0; i < moduli.size(); i++) {
        for (size_t j = i + 1; j < moduli.size(); j++) {
            uint64_t g = gcd(moduli[i], moduli[j]);
            if (g > 1 && g < moduli[i] && g < moduli[j]) {
                weak.push_back({moduli[i], g, moduli[j]});
                weak.push_back({moduli[j], g, moduli[i]});
            }
        }
    }
    return weak;
}

// ── Chinese Remainder Theorem ────────────────────────────────────────

inline std::pair<uint64_t, uint64_t> chinese_remainder(
    const std::vector<uint64_t>& remainders,
    const std::vector<uint64_t>& moduli)
{
    if (remainders.empty()) return {0, 0};
    int64_t r = (int64_t)remainders[0];
    int64_t m = (int64_t)moduli[0];

    for (size_t i = 1; i < remainders.size(); i++) {
        int64_t r2 = (int64_t)remainders[i];
        int64_t m2 = (int64_t)moduli[i];
        int64_t x, y;
        int64_t g = extended_gcd(m, m2, x, y);
        if ((r2 - r) % g != 0) return {0, 0};
        int64_t lcm_val = m / g * m2;
        int64_t diff = (r2 - r) / g;
        __int128 step = (__int128)diff * x % (m2 / g);
        r = (int64_t)(r + m * (int64_t)step);
        m = lcm_val;
        r = ((r % m) + m) % m;
    }
    return {(uint64_t)r, (uint64_t)m};
}

// ── LCG Period Detection (Floyd's Cycle) ─────────────────────────────

struct LCGAnalysis {
    uint64_t a, c, m;
    uint64_t period;
    uint64_t tail_length;
    bool full_period;
    std::vector<uint64_t> factors_of_m;
};

inline LCGAnalysis analyze_lcg(uint64_t a, uint64_t c, uint64_t m, uint64_t seed = 0) {
    LCGAnalysis result;
    result.a = a; result.c = c; result.m = m;
    result.factors_of_m = factor_u64(m);

    auto step = [&](uint64_t x) -> uint64_t {
        return (mulmod(a, x, m) + c) % m;
    };

    uint64_t tortoise = step(seed);
    uint64_t hare = step(step(seed));
    uint64_t max_iter = (m < 100000000ULL) ? m : 100000000ULL;
    uint64_t iter = 0;

    while (tortoise != hare && iter < max_iter) {
        tortoise = step(tortoise);
        hare = step(step(hare));
        iter++;
    }

    result.tail_length = 0;
    tortoise = seed;
    while (tortoise != hare && result.tail_length < max_iter) {
        tortoise = step(tortoise);
        hare = step(hare);
        result.tail_length++;
    }

    result.period = 1;
    hare = step(tortoise);
    while (tortoise != hare && result.period < max_iter) {
        hare = step(hare);
        result.period++;
    }

    result.full_period = (result.period == m);
    return result;
}

// ── Cornacchia's Algorithm ───────────────────────────────────────────
// Find x, y such that x² + d·y² = p

inline std::pair<uint64_t, uint64_t> cornacchia(uint64_t d, uint64_t p) {
    if (d >= p) return {0, 0};
    uint64_t r = sqrt_mod(p - d, p);
    if (r == 0) return {0, 0};
    if (2 * r < p) r = p - r;

    uint64_t a = p, b = r;
    uint64_t limit = (uint64_t)std::sqrt((double)p);
    while (b > limit) {
        uint64_t t = a % b;
        a = b; b = t;
    }

    uint64_t rem = p - b * b;
    if (rem % d != 0) return {0, 0};
    uint64_t y2 = rem / d;
    uint64_t y = (uint64_t)std::sqrt((double)y2);
    if (y * y != y2) return {0, 0};
    return {b, y};
}

// ── Legendre Symbol ──────────────────────────────────────────────────

inline int legendre_symbol(uint64_t a, uint64_t p) {
    uint64_t val = modpow(a % p, (p - 1) / 2, p);
    if (val == 0) return 0;
    if (val == 1) return 1;
    return -1;
}

// ── Jacobi Symbol ────────────────────────────────────────────────────

inline int jacobi_symbol(int64_t a, int64_t n) {
    if (n <= 0 || n % 2 == 0) return 0;
    a = ((a % n) + n) % n;
    int result = 1;
    while (a != 0) {
        while (a % 2 == 0) {
            a /= 2;
            if (n % 8 == 3 || n % 8 == 5) result = -result;
        }
        std::swap(a, n);
        if (a % 4 == 3 && n % 4 == 3) result = -result;
        a %= n;
    }
    return (n == 1) ? result : 0;
}

// ── Sum of Two Squares ───────────────────────────────────────────────
// Find a, b such that a² + b² = n (Fermat's theorem for primes ≡ 1 mod 4)

inline std::pair<uint64_t, uint64_t> sum_two_squares(uint64_t n) {
    if (n == 0) return {0, 0};
    if (n == 1) return {1, 0};
    if (n == 2) return {1, 1};
    if (is_prime(n)) {
        if (n % 4 != 1) return {0, 0};
        return cornacchia(1, n);
    }
    // For composites, try direct search up to sqrt(n)
    uint64_t limit = (uint64_t)std::sqrt((double)n);
    for (uint64_t a = 0; a <= limit; a++) {
        uint64_t rem = n - a * a;
        uint64_t b = (uint64_t)std::sqrt((double)rem);
        if (b * b == rem) return {a, b};
    }
    return {0, 0};
}

// ── Pollard p-1 Factoring ────────────────────────────────────────────
// Finds factor of n if n has a factor p where p-1 is B-smooth

inline uint64_t pollard_p_minus_1(uint64_t n, uint64_t B = 100000) {
    if (n <= 1) return 0;
    uint64_t a = 2;
    auto small_primes = sieve(B);
    for (uint64_t p = 2; p <= B; p++) {
        if (!small_primes[p]) continue;
        uint64_t pk = p;
        while (pk <= B) {
            a = modpow(a, p, n);
            pk *= p;
        }
        uint64_t g = gcd((a > 0 ? a - 1 : n - 1), n);
        if (g > 1 && g < n) return g;
    }
    return 0;
}

// ── Perfect Power Detection ──────────────────────────────────────────
// Check if n = a^k for some a, k ≥ 2. Returns {a, k} or {n, 1}.

inline std::pair<uint64_t, uint64_t> perfect_power(uint64_t n) {
    if (n <= 1) return {n, 1};
    for (uint64_t k = 63; k >= 2; k--) {
        double root = std::pow((double)n, 1.0 / k);
        uint64_t a = (uint64_t)root;
        for (uint64_t test = (a > 0 ? a - 1 : 0); test <= a + 1; test++) {
            uint64_t power = 1;
            bool overflow = false;
            for (uint64_t i = 0; i < k; i++) {
                if (power > UINT64_MAX / (test + 1)) { overflow = true; break; }
                power *= test;
            }
            if (!overflow && power == n) return {test, k};
        }
    }
    return {n, 1};
}

// =====================================================================
// Factor Seed Prediction Tests
// Treating factors as "seeds" that generate composite families,
// looking for commonality between seed types to predict primes
// =====================================================================

// ── Seed Classification ─────────────────────────────────────────────
// Classify a factor into a seed type based on its properties

enum class SeedType {
    TinyPrime,      // 2, 3, 5, 7
    SmallPrime,     // primes < 100
    TwinFactor,     // factor is part of a twin prime pair
    SophieFactor,   // factor is a Sophie Germain prime (2p+1 also prime)
    MersenneFactor, // factor is 2^k - 1
    PowerFactor,    // factor is near a perfect power
    DigitFactor,    // factor has repeating/palindromic digits
    LargePrime,     // prime >= 100
    Composite       // factor is itself composite (shouldn't happen with full factoring)
};

inline SeedType classify_seed(uint64_t f) {
    if (f <= 7 && is_prime(f)) return SeedType::TinyPrime;
    if (f < 100 && is_prime(f)) return SeedType::SmallPrime;
    if (is_prime(f)) {
        // Twin prime factor?
        if (is_prime(f - 2) || is_prime(f + 2)) return SeedType::TwinFactor;
        // Sophie Germain?
        if (is_prime(2 * f + 1)) return SeedType::SophieFactor;
        // Mersenne-like? Check if f = 2^k - 1
        uint64_t t = f + 1;
        if ((t & (t - 1)) == 0) return SeedType::MersenneFactor;
        // Near a power?
        for (uint64_t base = 2; base * base <= f + 2; base++) {
            uint64_t p = base * base;
            while (p < f - 1 && p < UINT64_MAX / base) p *= base;
            if (p >= f - 1 && p <= f + 1) return SeedType::PowerFactor;
        }
        return f >= 100 ? SeedType::LargePrime : SeedType::SmallPrime;
    }
    return SeedType::Composite;
}

inline const char* seed_type_name(SeedType t) {
    switch (t) {
        case SeedType::TinyPrime: return "TinyPrime";
        case SeedType::SmallPrime: return "SmallPrime";
        case SeedType::TwinFactor: return "TwinFactor";
        case SeedType::SophieFactor: return "SophieFactor";
        case SeedType::MersenneFactor: return "MersenneFactor";
        case SeedType::PowerFactor: return "PowerFactor";
        case SeedType::DigitFactor: return "DigitFactor";
        case SeedType::LargePrime: return "LargePrime";
        case SeedType::Composite: return "Composite";
    }
    return "Unknown";
}

// ── Ring Beacon Test ────────────────────────────────────────────────
// Maps numbers onto a mod-210 wheel (ring/spoke). For each composite,
// its factor seeds act as "beacons" marking that spoke as composite.
// Spokes with no beacons across multiple rings predict prime locations.

struct RingBeaconResult {
    uint64_t range_start, range_end;
    int total_primes;
    int predicted_primes;   // spokes with no beacon hits
    int correct_predictions;
    double precision;       // correct / predicted
    double recall;          // correct / total_primes
    std::vector<std::pair<int, int>> spoke_beacon_counts; // spoke -> beacon count
};

inline RingBeaconResult ring_beacon_test(uint64_t lo, uint64_t hi) {
    const int WHEEL = 210;
    RingBeaconResult result;
    result.range_start = lo;
    result.range_end = hi;
    result.total_primes = 0;
    result.predicted_primes = 0;
    result.correct_predictions = 0;

    // Count beacon hits per spoke
    std::vector<int> beacon_hits(WHEEL, 0);
    std::vector<bool> is_prime_at(hi - lo + 1, false);

    for (uint64_t n = lo; n <= hi; n++) {
        if (is_prime(n)) {
            is_prime_at[n - lo] = true;
            result.total_primes++;
        } else if (n > 1) {
            auto factors = factor_u64(n);
            int spoke = (int)(n % WHEEL);
            for (auto f : factors) {
                beacon_hits[spoke]++;
            }
        }
    }

    // Low beacon count spokes are predicted prime locations
    // Find threshold: spokes with beacon count below median
    std::vector<int> sorted_hits;
    for (int s = 0; s < WHEEL; s++) {
        if (gcd((uint64_t)s, (uint64_t)WHEEL) == 1 || s == 1) // only coprime spokes can have primes
            sorted_hits.push_back(beacon_hits[s]);
    }
    if (sorted_hits.empty()) {
        result.precision = 0; result.recall = 0;
        return result;
    }
    std::sort(sorted_hits.begin(), sorted_hits.end());
    int threshold = sorted_hits[sorted_hits.size() / 4]; // bottom quartile

    for (uint64_t n = lo; n <= hi; n++) {
        int spoke = (int)(n % WHEEL);
        if (beacon_hits[spoke] <= threshold) {
            result.predicted_primes++;
            if (is_prime_at[n - lo]) result.correct_predictions++;
        }
    }

    result.precision = result.predicted_primes > 0 ?
        (double)result.correct_predictions / result.predicted_primes : 0;
    result.recall = result.total_primes > 0 ?
        (double)result.correct_predictions / result.total_primes : 0;

    result.spoke_beacon_counts.clear();
    for (int s = 0; s < WHEEL; s++) {
        if (beacon_hits[s] > 0)
            result.spoke_beacon_counts.push_back({s, beacon_hits[s]});
    }
    std::sort(result.spoke_beacon_counts.begin(), result.spoke_beacon_counts.end(),
        [](auto& a, auto& b) { return a.second < b.second; });

    return result;
}

// ── Topography Test ─────────────────────────────────────────────────
// Treats factor count as "elevation". Composites with many factors are
// deep valleys; primes are peaks. Analyzes the terrain to find if
// peak spacing (prime gaps) correlates with valley depth patterns.

struct TopographyResult {
    uint64_t range_start, range_end;
    int num_primes;
    double avg_gap;
    double avg_valley_depth;     // avg factor count of composites between primes
    double depth_gap_correlation; // correlation between valley depth and next gap
    std::vector<std::pair<uint64_t, int>> deepest_valleys; // position, depth
    std::vector<std::pair<uint64_t, int>> prime_gaps;      // prime, gap to next
};

inline TopographyResult topography_test(uint64_t lo, uint64_t hi) {
    TopographyResult result;
    result.range_start = lo;
    result.range_end = hi;
    result.num_primes = 0;

    // Build elevation map: prime = 0 (peak marker), composite = factor count
    struct Point { uint64_t n; int elevation; bool prime; };
    std::vector<Point> terrain;
    for (uint64_t n = lo; n <= hi; n++) {
        Point p;
        p.n = n;
        p.prime = is_prime(n);
        if (p.prime) {
            p.elevation = 0;
            result.num_primes++;
        } else if (n <= 1) {
            p.elevation = 0;
        } else {
            auto factors = factor_u64(n);
            p.elevation = (int)factors.size();
        }
        terrain.push_back(p);
    }

    // Find prime gaps and valley depths between consecutive primes
    std::vector<uint64_t> primes_found;
    for (auto& pt : terrain) {
        if (pt.prime) primes_found.push_back(pt.n);
    }

    double sum_gap = 0, sum_depth = 0;
    std::vector<double> gaps, depths;
    for (size_t i = 0; i + 1 < primes_found.size(); i++) {
        uint64_t gap = primes_found[i + 1] - primes_found[i];
        result.prime_gaps.push_back({primes_found[i], (int)gap});
        gaps.push_back((double)gap);
        sum_gap += gap;

        // Average valley depth between these two primes
        double valley_sum = 0;
        int valley_count = 0;
        int max_depth = 0;
        for (uint64_t n = primes_found[i] + 1; n < primes_found[i + 1]; n++) {
            int elev = terrain[n - lo].elevation;
            valley_sum += elev;
            valley_count++;
            if (elev > max_depth) max_depth = elev;
        }
        double avg_d = valley_count > 0 ? valley_sum / valley_count : 0;
        depths.push_back(avg_d);
        sum_depth += avg_d;

        if (max_depth >= 4) {
            // Find the deepest point
            for (uint64_t n = primes_found[i] + 1; n < primes_found[i + 1]; n++) {
                if (terrain[n - lo].elevation == max_depth) {
                    result.deepest_valleys.push_back({n, max_depth});
                    break;
                }
            }
        }
    }

    result.avg_gap = gaps.empty() ? 0 : sum_gap / gaps.size();
    result.avg_valley_depth = depths.empty() ? 0 : sum_depth / depths.size();

    // Pearson correlation between valley depth and gap size
    if (gaps.size() >= 3) {
        double mean_g = sum_gap / gaps.size();
        double mean_d = sum_depth / depths.size();
        double cov = 0, var_g = 0, var_d = 0;
        for (size_t i = 0; i < gaps.size(); i++) {
            double dg = gaps[i] - mean_g;
            double dd = depths[i] - mean_d;
            cov += dg * dd;
            var_g += dg * dg;
            var_d += dd * dd;
        }
        double denom = std::sqrt(var_g * var_d);
        result.depth_gap_correlation = denom > 0 ? cov / denom : 0;
    } else {
        result.depth_gap_correlation = 0;
    }

    // Sort deepest valleys
    std::sort(result.deepest_valleys.begin(), result.deepest_valleys.end(),
        [](auto& a, auto& b) { return a.second > b.second; });
    if (result.deepest_valleys.size() > 20)
        result.deepest_valleys.resize(20);

    return result;
}

// ── Web Test (Factor Web / Divisor Graph) ───────────────────────────
// Builds a graph where composites sharing a factor seed type are
// connected. Clusters of composites with common seed types leave
// predictable gaps where primes appear.

struct WebTestResult {
    uint64_t range_start, range_end;
    int num_primes;
    int num_composites;
    // Seed type distribution
    std::map<SeedType, int> seed_counts;
    // How many composites share each seed type pair
    int shared_tiny;      // composites with TinyPrime seeds
    int shared_twin;      // composites with TwinFactor seeds
    int shared_sophie;    // composites with SophieFactor seeds
    // Prediction: numbers not "webbed" (no shared seed neighbors) -> likely prime
    int predicted_primes;
    int correct_predictions;
    double precision, recall;
};

inline WebTestResult web_test(uint64_t lo, uint64_t hi) {
    WebTestResult result;
    result.range_start = lo;
    result.range_end = hi;
    result.num_primes = 0;
    result.num_composites = 0;
    result.shared_tiny = 0;
    result.shared_twin = 0;
    result.shared_sophie = 0;
    result.predicted_primes = 0;
    result.correct_predictions = 0;

    // For each number, compute its seed signature
    struct NumInfo {
        uint64_t n;
        bool prime;
        std::vector<SeedType> seeds;
        int web_connections; // how many neighbors share a seed type
    };

    std::vector<NumInfo> nums;
    for (uint64_t n = lo; n <= hi; n++) {
        NumInfo info;
        info.n = n;
        info.prime = is_prime(n);
        info.web_connections = 0;
        if (info.prime) {
            result.num_primes++;
        } else if (n > 1) {
            result.num_composites++;
            auto factors = factor_u64(n);
            uint64_t prev = 0;
            for (auto f : factors) {
                if (f != prev) {
                    SeedType st = classify_seed(f);
                    info.seeds.push_back(st);
                    result.seed_counts[st]++;
                    prev = f;
                }
            }
        }
        nums.push_back(info);
    }

    // Count web connections: for each composite, count how many of its
    // immediate neighbors (+/-1..5) share a seed type
    for (size_t i = 0; i < nums.size(); i++) {
        if (nums[i].prime || nums[i].seeds.empty()) continue;
        for (int d = 1; d <= 5 && i + d < nums.size(); d++) {
            if (nums[i + d].prime || nums[i + d].seeds.empty()) continue;
            for (auto s1 : nums[i].seeds) {
                for (auto s2 : nums[i + d].seeds) {
                    if (s1 == s2) {
                        nums[i].web_connections++;
                        nums[i + d].web_connections++;
                        if (s1 == SeedType::TinyPrime) result.shared_tiny++;
                        else if (s1 == SeedType::TwinFactor) result.shared_twin++;
                        else if (s1 == SeedType::SophieFactor) result.shared_sophie++;
                        goto next_neighbor;
                    }
                }
            }
            next_neighbor:;
        }
    }

    // Predict: numbers with zero web connections in neighborhood -> prime candidate
    for (size_t i = 0; i < nums.size(); i++) {
        if (nums[i].seeds.empty() && !nums[i].prime && nums[i].n <= 1) continue;
        // Check if this position has low web density
        int local_web = 0;
        for (int d = -3; d <= 3; d++) {
            int j = (int)i + d;
            if (j >= 0 && j < (int)nums.size())
                local_web += nums[j].web_connections;
        }
        if (local_web == 0 && nums[i].n > 1) {
            result.predicted_primes++;
            if (nums[i].prime) result.correct_predictions++;
        }
    }

    result.precision = result.predicted_primes > 0 ?
        (double)result.correct_predictions / result.predicted_primes : 0;
    result.recall = result.num_primes > 0 ?
        (double)result.correct_predictions / result.num_primes : 0;
    return result;
}

// ── Audio Test (Harmonic Factor Resonance) ──────────────────────────
// Treats each prime factor as a frequency. Composites produce "chords".
// When multiple composites in sequence share harmonic relationships
// (factors with small ratios like 2:3, 3:5), they form resonance zones.
// Gaps in resonance -> prime candidates.

struct AudioTestResult {
    uint64_t range_start, range_end;
    int num_primes;
    // Resonance analysis
    double avg_resonance;        // average harmonic score across range
    int resonance_gaps;          // count of low-resonance zones
    int primes_in_gaps;          // primes found in low-resonance zones
    double gap_prime_rate;       // primes per position in gaps vs overall
    // Harmonic ratios found
    std::map<std::string, int> common_harmonics; // "2:3" -> count
};

inline double harmonic_score(const std::vector<uint64_t>& factors) {
    if (factors.size() < 2) return 0;
    double score = 0;
    std::vector<uint64_t> distinct;
    uint64_t prev = 0;
    for (auto f : factors) {
        if (f != prev) { distinct.push_back(f); prev = f; }
    }
    // Score based on how "harmonically related" the factors are
    for (size_t i = 0; i < distinct.size(); i++) {
        for (size_t j = i + 1; j < distinct.size(); j++) {
            uint64_t a = distinct[i], b = distinct[j];
            uint64_t g = gcd(a, b);
            double ratio = (double)b / a;
            // Simple integer ratios score high (like musical intervals)
            double r_int = std::round(ratio);
            double deviation = std::abs(ratio - r_int);
            if (deviation < 0.01 && r_int > 0) {
                score += 1.0 / r_int; // octave (2:1) = 0.5, fifth (3:2) = 0.67
            }
            // Small ratios (like 2:3, 3:5) score higher
            if (g > 1) score += 0.1;
        }
    }
    return score;
}

inline AudioTestResult audio_test(uint64_t lo, uint64_t hi) {
    AudioTestResult result;
    result.range_start = lo;
    result.range_end = hi;
    result.num_primes = 0;
    result.resonance_gaps = 0;
    result.primes_in_gaps = 0;

    std::vector<double> resonance;
    std::vector<bool> is_p;

    for (uint64_t n = lo; n <= hi; n++) {
        bool p = is_prime(n);
        is_p.push_back(p);
        if (p) {
            result.num_primes++;
            resonance.push_back(0);
        } else if (n <= 1) {
            resonance.push_back(0);
        } else {
            auto factors = factor_u64(n);
            double h = harmonic_score(factors);
            resonance.push_back(h);

            // Track common harmonic ratios
            std::vector<uint64_t> distinct;
            uint64_t prev = 0;
            for (auto f : factors) {
                if (f != prev) { distinct.push_back(f); prev = f; }
            }
            for (size_t i = 0; i + 1 < distinct.size(); i++) {
                uint64_t a = distinct[i], b = distinct[i + 1];
                uint64_t g = gcd(a, b);
                std::string ratio = std::to_string(a / g) + ":" + std::to_string(b / g);
                result.common_harmonics[ratio]++;
            }
        }
    }

    // Compute windowed resonance (window of 5)
    double total_res = 0;
    std::vector<double> windowed;
    for (size_t i = 0; i < resonance.size(); i++) {
        double sum = 0;
        int count = 0;
        for (int d = -2; d <= 2; d++) {
            int j = (int)i + d;
            if (j >= 0 && j < (int)resonance.size()) {
                sum += resonance[j];
                count++;
            }
        }
        double w = count > 0 ? sum / count : 0;
        windowed.push_back(w);
        total_res += w;
    }
    result.avg_resonance = resonance.empty() ? 0 : total_res / resonance.size();

    // Find low-resonance gaps (below 25th percentile)
    std::vector<double> sorted_w = windowed;
    std::sort(sorted_w.begin(), sorted_w.end());
    double threshold = sorted_w.size() > 4 ? sorted_w[sorted_w.size() / 4] : 0;

    bool in_gap = false;
    for (size_t i = 0; i < windowed.size(); i++) {
        if (windowed[i] <= threshold && windowed[i] < 0.01) {
            if (!in_gap) { result.resonance_gaps++; in_gap = true; }
            if (is_p[i]) result.primes_in_gaps++;
        } else {
            in_gap = false;
        }
    }

    double total_positions = (double)(hi - lo + 1);
    double gap_positions = 0;
    for (size_t i = 0; i < windowed.size(); i++) {
        if (windowed[i] <= threshold && windowed[i] < 0.01) gap_positions++;
    }
    double overall_rate = result.num_primes / total_positions;
    double gap_rate = gap_positions > 0 ? result.primes_in_gaps / gap_positions : 0;
    result.gap_prime_rate = overall_rate > 0 ? gap_rate / overall_rate : 0;

    return result;
}

// ── Twisting Tree Test ──────────────────────────────────────────────
// Builds factor trees for composites. Classifies each tree by its
// "twist" -- the shape signature (depth, branching pattern).
// Looks for patterns: do certain tree shapes cluster? Do transitions
// between tree shapes predict primes?

struct TreeShape {
    int depth;          // depth of factor tree
    int width;          // number of distinct prime factors
    int total_factors;  // total factors with multiplicity
    bool balanced;      // all prime factors appear same number of times
    uint64_t smallest;  // smallest prime factor
    uint64_t largest;   // largest prime factor
};

inline TreeShape compute_tree_shape(uint64_t n) {
    TreeShape t;
    t.depth = 0; t.width = 0; t.total_factors = 0;
    t.balanced = true; t.smallest = n; t.largest = 0;

    if (n <= 1 || is_prime(n)) {
        t.depth = 0; t.width = 1; t.total_factors = 1;
        t.smallest = n; t.largest = n;
        return t;
    }

    auto factors = factor_u64(n);
    t.total_factors = (int)factors.size();

    // Count distinct factors and multiplicities
    std::map<uint64_t, int> mult;
    for (auto f : factors) mult[f]++;
    t.width = (int)mult.size();
    t.smallest = mult.begin()->first;
    t.largest = mult.rbegin()->first;

    // Depth = max multiplicity (depth of repeated factoring)
    int max_mult = 0, first_mult = -1;
    for (auto& [p, m] : mult) {
        if (m > max_mult) max_mult = m;
        if (first_mult < 0) first_mult = m;
        if (m != first_mult) t.balanced = false;
    }
    t.depth = max_mult;

    return t;
}

struct TwistingTreeResult {
    uint64_t range_start, range_end;
    int num_primes;
    // Tree shape distribution
    std::map<std::string, int> shape_counts; // "d2w3" (depth 2, width 3) -> count
    // Transition analysis
    int shape_changes;        // how many times tree shape changes between consecutive composites
    int primes_after_change;  // how many primes appear right after a shape change
    double change_prime_rate; // rate of primes after changes vs overall
    // Twist patterns
    int balanced_trees;
    int unbalanced_trees;
    double balanced_gap_avg;   // avg gap after balanced tree composite
    double unbalanced_gap_avg; // avg gap after unbalanced tree composite
};

inline TwistingTreeResult twisting_tree_test(uint64_t lo, uint64_t hi) {
    TwistingTreeResult result;
    result.range_start = lo;
    result.range_end = hi;
    result.num_primes = 0;
    result.shape_changes = 0;
    result.primes_after_change = 0;
    result.balanced_trees = 0;
    result.unbalanced_trees = 0;
    result.balanced_gap_avg = 0;
    result.unbalanced_gap_avg = 0;

    struct Entry {
        uint64_t n;
        bool prime;
        TreeShape shape;
        std::string shape_key;
    };

    std::vector<Entry> entries;
    for (uint64_t n = lo; n <= hi; n++) {
        Entry e;
        e.n = n;
        e.prime = is_prime(n);
        if (e.prime) result.num_primes++;
        if (!e.prime && n > 1) {
            e.shape = compute_tree_shape(n);
            e.shape_key = "d" + std::to_string(e.shape.depth) +
                          "w" + std::to_string(e.shape.width);
            result.shape_counts[e.shape_key]++;
            if (e.shape.balanced) result.balanced_trees++;
            else result.unbalanced_trees++;
        } else {
            e.shape_key = "prime";
        }
        entries.push_back(e);
    }

    // Analyze transitions between tree shapes
    std::string last_shape = "";
    bool last_was_change = false;
    double bal_gap_sum = 0, unbal_gap_sum = 0;
    int bal_gap_count = 0, unbal_gap_count = 0;

    for (size_t i = 0; i < entries.size(); i++) {
        if (entries[i].prime) {
            if (last_was_change) result.primes_after_change++;
            last_was_change = false;
            continue;
        }
        if (entries[i].shape_key == "prime") continue;

        if (!last_shape.empty() && entries[i].shape_key != last_shape) {
            result.shape_changes++;
            last_was_change = true;
        } else {
            last_was_change = false;
        }
        last_shape = entries[i].shape_key;

        // Measure gap to next prime after this composite
        for (size_t j = i + 1; j < entries.size(); j++) {
            if (entries[j].prime) {
                double gap = (double)(entries[j].n - entries[i].n);
                if (entries[i].shape.balanced) {
                    bal_gap_sum += gap;
                    bal_gap_count++;
                } else {
                    unbal_gap_sum += gap;
                    unbal_gap_count++;
                }
                break;
            }
        }
    }

    result.balanced_gap_avg = bal_gap_count > 0 ? bal_gap_sum / bal_gap_count : 0;
    result.unbalanced_gap_avg = unbal_gap_count > 0 ? unbal_gap_sum / unbal_gap_count : 0;

    double overall_prime_rate = (double)result.num_primes / (hi - lo + 1);
    double change_prime_rate = result.shape_changes > 0 ?
        (double)result.primes_after_change / result.shape_changes : 0;
    result.change_prime_rate = overall_prime_rate > 0 ?
        change_prime_rate / overall_prime_rate : 0;

    return result;
}

} // namespace prime
