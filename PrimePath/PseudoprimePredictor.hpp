#pragma once
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <functional>
#include <sys/sysctl.h>
#include <mach/mach.h>

namespace prime {

// ═══════════════════════════════════════════════════════════════════════
// PseudoprimePredictor — generates Carmichael numbers and strong
// pseudoprimes ahead of the search frontier using seed-pair construction.
//
// Key insight from analysis:
//   - Carmichaels are p×q×r where (p-1)|(n-1) for each factor (Korselt)
//   - 71% of factors are ≡ 1 (mod 6), 95% have 120-smooth (p-1)
//   - Locked-ratio pairs (p,q) with (p-1)|(q-1) generate families
//   - SPRP-2: 84% are 2-factor, 71% have (p-1)|(q-1)
//
// Runs on spare CPU cycles to pre-populate a lookup set.
// Workers check this set to avoid false-positive discoveries.
// ═══════════════════════════════════════════════════════════════════════

// ── Global memory budget: 2/3 of physical RAM ──────────────────────
// Shared with GPU (unified memory on Apple Silicon), so we leave 1/3 free.

inline uint64_t system_memory_budget() {
    static uint64_t budget = 0;
    if (budget == 0) {
        int64_t phys = 0;
        size_t len = sizeof(phys);
        if (sysctlbyname("hw.memsize", &phys, &len, nullptr, 0) == 0 && phys > 0)
            budget = (uint64_t)phys * 2 / 3;
        else
            budget = 8ULL * 1024 * 1024 * 1024 * 2 / 3;  // fallback: assume 8 GB
    }
    return budget;
}

inline uint64_t current_memory_usage() {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return info.resident_size;
    return 0;
}

inline bool memory_pressure_ok() {
    return current_memory_usage() < system_memory_budget();
}

class PseudoprimePredictor {
public:
    PseudoprimePredictor() = default;

    // Generate all 3-factor Carmichael numbers in [lo, hi]
    // Uses Korselt construction: enumerate seed pairs, solve for third factor.
    // Designed to run on thread pool — each call is self-contained.
    void generate_carmichaels(uint64_t lo, uint64_t hi);

    // Generate SPRP base-2 candidates in [lo, hi] using (p-1)|(q-1) construction
    void generate_sprp2(uint64_t lo, uint64_t hi);

    // Check if n is a predicted pseudoprime
    bool is_predicted(uint64_t n) const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _predicted.count(n) > 0;
    }

    // Get count of predicted pseudoprimes
    size_t count() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _predicted.size();
    }

    // Get the current high-water mark (how far ahead we've predicted)
    uint64_t frontier() const { return _frontier.load(std::memory_order_relaxed); }

    // Evict all predicted values below the given position (they're behind
    // the search frontier and will never be checked again).
    // Call this periodically from workers when advancing the search position.
    size_t evict_below(uint64_t pos) {
        std::lock_guard<std::mutex> lock(_mutex);
        size_t before = _predicted.size();
        for (auto it = _predicted.begin(); it != _predicted.end(); ) {
            if (*it < pos)
                it = _predicted.erase(it);
            else
                ++it;
        }
        _evict_watermark = pos;
        return before - _predicted.size();
    }

    uint64_t evict_watermark() const { return _evict_watermark.load(std::memory_order_relaxed); }

    // Get all predicted values in a range (for display/export)
    std::vector<uint64_t> get_in_range(uint64_t lo, uint64_t hi) const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<uint64_t> result;
        for (auto v : _predicted) {
            if (v >= lo && v <= hi) result.push_back(v);
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    size_t carmichael_count() const { return _n_carmichaels.load(std::memory_order_relaxed); }
    size_t sprp2_count() const { return _n_sprp2.load(std::memory_order_relaxed); }

private:
    mutable std::mutex _mutex;
    std::unordered_set<uint64_t> _predicted;
    std::atomic<uint64_t> _frontier{0};
    std::atomic<uint64_t> _evict_watermark{0};
    std::atomic<size_t> _n_carmichaels{0};
    std::atomic<size_t> _n_sprp2{0};

    // Hard cap: 2M entries ≈ ~96 MB (unordered_set overhead).
    // Beyond this, skip insertions — eviction will make room.
    static constexpr size_t MAX_PREDICTED = 2'000'000;

    void add(uint64_t n) {
        // Don't store values already behind the eviction watermark
        if (n < _evict_watermark.load(std::memory_order_relaxed)) return;
        std::lock_guard<std::mutex> lock(_mutex);
        if (_predicted.size() >= MAX_PREDICTED) return;  // hard cap
        _predicted.insert(n);
    }

    // Small prime sieve for seed generation
    static std::vector<uint64_t> sieve_primes(uint64_t limit) {
        std::vector<bool> is_p(limit + 1, true);
        is_p[0] = is_p[1] = false;
        for (uint64_t i = 2; i * i <= limit; i++)
            if (is_p[i]) for (uint64_t j = i*i; j <= limit; j += i) is_p[j] = false;
        std::vector<uint64_t> primes;
        for (uint64_t i = 3; i <= limit; i += 2)
            if (is_p[i]) primes.push_back(i);
        return primes;
    }

    // Modular inverse of a mod m (extended GCD). Returns 0 if no inverse.
    static uint64_t mod_inverse(uint64_t a, uint64_t m) {
        if (m <= 1) return 0;
        a %= m;
        if (a == 0) return 0;
        // Extended Euclidean via iterative method
        int64_t old_r = (int64_t)a, r = (int64_t)m;
        int64_t old_s = 1, s = 0;
        while (r != 0) {
            int64_t q = old_r / r;
            int64_t tmp = r;
            r = old_r - q * r;
            old_r = tmp;
            tmp = s;
            s = old_s - q * s;
            old_s = tmp;
        }
        if (old_r != 1) return 0; // no inverse
        return old_s < 0 ? (uint64_t)(old_s + (int64_t)m) : (uint64_t)old_s;
    }

    // Miller-Rabin with specific base
    static uint64_t modpow(uint64_t base, uint64_t exp, uint64_t mod) {
        unsigned __int128 result = 1, b = base % mod;
        while (exp > 0) {
            if (exp & 1) result = result * b % mod;
            b = b * b % mod;
            exp >>= 1;
        }
        return (uint64_t)result;
    }

    static bool miller_rabin(uint64_t n, uint64_t a) {
        if (n < 2) return false;
        if (n == a) return true;
        if (n % 2 == 0) return false;
        uint64_t d = n - 1;
        int r = 0;
        while (d % 2 == 0) { d /= 2; r++; }
        uint64_t x = modpow(a, d, n);
        if (x == 1 || x == n - 1) return true;
        for (int i = 0; i < r - 1; i++) {
            x = (unsigned __int128)x * x % n;
            if (x == n - 1) return true;
        }
        return false;
    }

    static bool is_prime(uint64_t n) {
        if (n < 2) return false;
        if (n < 4) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        for (uint64_t a : {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL}) {
            if (n == a) return true;
            if (!miller_rabin(n, a)) return false;
        }
        return true;
    }

    // Check if n passes Miller-Rabin for base 2 (i.e. is SPRP-2)
    static bool is_sprp_base2(uint64_t n) {
        return miller_rabin(n, 2);
    }
};

// ── Carmichael generation via Korselt construction ──────────────────
//
// For each pair of odd primes (p, q) with p < q:
//   L = lcm(p-1, q-1)
//   r ≡ (p*q)^{-1} (mod L), r > q, r prime
//   n = p*q*r is Carmichael iff (r-1) | (n-1)
//
// Complexity: O(π(sqrt(hi))² × hi/L) — dominated by seed pair enumeration

inline void PseudoprimePredictor::generate_carmichaels(uint64_t lo, uint64_t hi) {
    if (hi < 561) return; // smallest Carmichael

    // Seed primes up to cube root of hi (p is smallest factor, p³ ≤ hi)
    // Cap aggressively to avoid runaway computation at large frontiers (>10^12)
    uint64_t seed_limit = 1;
    for (uint64_t s = 1; s * s * s <= hi; s++) seed_limit = s;
    seed_limit += 100;
    if (seed_limit > 10000) seed_limit = 10000;

    auto primes = sieve_primes(seed_limit);

    size_t found = 0;
    for (size_t pi = 0; pi < primes.size(); pi++) {
        uint64_t p = primes[pi];
        if (p * p * (p + 2) > hi) break;

        for (size_t qi = pi + 1; qi < primes.size(); qi++) {
            uint64_t q = primes[qi];
            if (p * q * (q + 2) > hi) break;

            uint64_t pm1 = p - 1, qm1 = q - 1;
            uint64_t g = std::__gcd(pm1, qm1);
            uint64_t L = pm1 / g * qm1;

            if (L == 0 || L > hi / (p * q)) continue;

            uint64_t pq = p * q;
            uint64_t pq_mod_L = pq % L;
            uint64_t inv = mod_inverse(pq_mod_L, L);
            if (inv == 0) continue;

            // Enumerate valid r values
            for (uint64_t r = inv; ; r += L) {
                if (r <= q) continue;
                if (r > hi / pq + 1) break;
                if (r > UINT64_MAX / pq) break;  // prevent overflow

                uint64_t n = pq * r;
                if (n < lo || n > hi) { if (n > hi) break; continue; }

                if (!is_prime(r)) continue;

                // Verify full Korselt criterion for r
                if ((n - 1) % (r - 1) == 0) {
                    add(n);
                    found++;
                }
            }
        }
    }

    _n_carmichaels.fetch_add(found, std::memory_order_relaxed);
    uint64_t cur = _frontier.load(std::memory_order_relaxed);
    while (hi > cur && !_frontier.compare_exchange_weak(cur, hi, std::memory_order_relaxed));
}

// ── SPRP-2 generation via divisibility construction ─────────────────
//
// For 2-factor SPRP-2: n = p*q where p < q, both prime
//   Condition: n passes Miller-Rabin base 2 despite being composite
//   Key pattern: (p-1) | (q-1) in 71% of cases
//
// Strategy: for each small prime p, enumerate q where:
//   q ≡ 1 (mod p-1), q prime, and p*q passes MR base 2

inline void PseudoprimePredictor::generate_sprp2(uint64_t lo, uint64_t hi) {
    if (hi < 2047) return; // smallest SPRP-2

    // Seed primes up to sqrt(hi) — cap aggressively to bound computation.
    // At 10^15, sqrt ≈ 31.6M; we cap at 10K to keep runtime < 1s.
    uint64_t seed_limit = 1;
    for (uint64_t s = 1; s * s <= hi; s++) seed_limit = s;
    seed_limit += 100;
    if (seed_limit > 10000) seed_limit = 10000;

    auto primes = sieve_primes(seed_limit);

    size_t found = 0;
    static const size_t MAX_Q_ITERS = 100000; // cap per-prime q iteration
    for (size_t pi = 0; pi < primes.size(); pi++) {
        uint64_t p = primes[pi];
        if (p * (p + 2) > hi) break;

        uint64_t pm1 = p - 1;
        // q ≡ 1 (mod p-1) is the dominant pattern
        // Start q from max(p+2, lo/p) stepping by pm1
        uint64_t q_start = p + 2;
        if (lo > 0 && lo / p > q_start) q_start = lo / p;

        // Align q_start to 1 (mod pm1)
        uint64_t r = q_start % pm1;
        if (r != 1) q_start += (r == 0) ? 1 : (pm1 + 1 - r);

        size_t q_iters = 0;
        for (uint64_t q = q_start; q <= hi / p; q += pm1) {
            if (++q_iters > MAX_Q_ITERS) break;
            if (q <= p) continue;
            if (!is_prime(q)) continue;

            uint64_t n = p * q;
            if (n < lo || n > hi) continue;

            // Must be composite (it is, since n = p*q)
            // Check if it's SPRP base 2
            if (is_sprp_base2(n)) {
                add(n);
                found++;
            }
        }
    }

    _n_sprp2.fetch_add(found, std::memory_order_relaxed);
    uint64_t cur = _frontier.load(std::memory_order_relaxed);
    while (hi > cur && !_frontier.compare_exchange_weak(cur, hi, std::memory_order_relaxed));
}

} // namespace prime
