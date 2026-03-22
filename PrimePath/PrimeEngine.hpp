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

// ═══════════════════════════════════════════════════════════════════════
// PrimeEngine — C++ port of primepath with multi-core parallelism
// ═══════════════════════════════════════════════════════════════════════

namespace prime {

// ── Modular arithmetic (overflow-safe via __uint128_t) ───────────────

inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return (unsigned __int128)a * b % m;
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
