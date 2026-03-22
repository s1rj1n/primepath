#include "GPUBackend.hpp"
#include "PrimeEngine.hpp"
#include <thread>

namespace prime {

// ═══════════════════════════════════════════════════════════════════════
// CPUBackend — pure C++ fallback, works on any platform
// Uses __uint128_t (GCC/Clang) or _mul128 (MSVC) for u128 arithmetic
// ═══════════════════════════════════════════════════════════════════════

#ifdef _MSC_VER
// Windows: use _umul128 intrinsic
#include <intrin.h>
static uint64_t mulmod128_cpu(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t hi;
    uint64_t lo = _umul128(a, b, &hi);
    uint64_t result;
    _udiv128(hi, lo, m, &result);
    return result;
}
#else
// Unix/macOS: use __uint128_t
static uint64_t mulmod128_cpu(uint64_t a, uint64_t b, uint64_t m) {
    return (unsigned __int128)a * b % m;
}
#endif

// Wieferich test: 2^(p-1) ≡ 1 (mod p²)
static bool cpu_wieferich_test(uint64_t p) {
    if (p < 3) return false;
    unsigned __int128 p_sq = (unsigned __int128)p * p;
    unsigned __int128 base = 2, exp_val = p - 1, result = 1;
    base %= p_sq;
    while (exp_val > 0) {
        if (exp_val & 1) result = result * base % p_sq;
        exp_val >>= 1;
        if (exp_val > 0) base = base * base % p_sq;
    }
    return result == 1;
}

// Wall-Sun-Sun: p² | F(p - legendre(p,5))
static bool cpu_wallsunsun_test(uint64_t p) {
    if (p < 7) return false;
    unsigned __int128 p_sq = (unsigned __int128)p * p;
    int r5 = (int)(p % 5);
    uint64_t fib_idx = (r5 == 1 || r5 == 4) ? p - 1 : p + 1;

    // Matrix exponentiation: [[1,1],[1,0]]^n mod p²
    unsigned __int128 m00=1, m01=0, m10=0, m11=1;
    unsigned __int128 b00=1, b01=1, b10=1, b11=0;
    uint64_t e = fib_idx;
    while (e > 0) {
        if (e & 1) {
            unsigned __int128 t00 = (m00*b00 + m01*b10) % p_sq;
            unsigned __int128 t01 = (m00*b01 + m01*b11) % p_sq;
            unsigned __int128 t10 = (m10*b00 + m11*b10) % p_sq;
            unsigned __int128 t11 = (m10*b01 + m11*b11) % p_sq;
            m00=t00; m01=t01; m10=t10; m11=t11;
        }
        e >>= 1;
        if (e > 0) {
            unsigned __int128 t00 = (b00*b00 + b01*b10) % p_sq;
            unsigned __int128 t01 = (b00*b01 + b01*b11) % p_sq;
            unsigned __int128 t10 = (b10*b00 + b11*b10) % p_sq;
            unsigned __int128 t11 = (b10*b01 + b11*b11) % p_sq;
            b00=t00; b01=t01; b10=t10; b11=t11;
        }
    }
    return m01 == 0; // F(n) = m01
}

int CPUBackend::wieferich_batch(const uint64_t *primes, uint8_t *results, uint32_t count) {
    int hits = 0;
    for (uint32_t i = 0; i < count; i++) {
        results[i] = cpu_wieferich_test(primes[i]) ? 1 : 0;
        if (results[i]) hits++;
    }
    return hits;
}

int CPUBackend::wallsunsun_batch(const uint64_t *primes, uint8_t *results, uint32_t count) {
    int hits = 0;
    for (uint32_t i = 0; i < count; i++) {
        results[i] = cpu_wallsunsun_test(primes[i]) ? 1 : 0;
        if (results[i]) hits++;
    }
    return hits;
}

int CPUBackend::twin_batch(const uint64_t *cands, uint8_t *results, uint32_t count) {
    int hits = 0;
    for (uint32_t i = 0; i < count; i++) {
        results[i] = (is_prime(cands[i]) && is_prime(cands[i] + 2)) ? 1 : 0;
        if (results[i]) hits++;
    }
    return hits;
}

int CPUBackend::sophie_batch(const uint64_t *cands, uint8_t *results, uint32_t count) {
    int hits = 0;
    for (uint32_t i = 0; i < count; i++) {
        results[i] = (is_prime(cands[i]) && is_prime(2 * cands[i] + 1)) ? 1 : 0;
        if (results[i]) hits++;
    }
    return hits;
}

int CPUBackend::cousin_batch(const uint64_t *cands, uint8_t *results, uint32_t count) {
    int hits = 0;
    for (uint32_t i = 0; i < count; i++) {
        results[i] = (is_prime(cands[i]) && is_prime(cands[i] + 4)) ? 1 : 0;
        if (results[i]) hits++;
    }
    return hits;
}

int CPUBackend::sexy_batch(const uint64_t *cands, uint8_t *results, uint32_t count) {
    int hits = 0;
    for (uint32_t i = 0; i < count; i++) {
        results[i] = (is_prime(cands[i]) && is_prime(cands[i] + 6)) ? 1 : 0;
        if (results[i]) hits++;
    }
    return hits;
}

int CPUBackend::primality_batch(const uint64_t *cands, uint8_t *results, uint32_t count) {
    int hits = 0;
    for (uint32_t i = 0; i < count; i++) {
        results[i] = is_prime(cands[i]) ? 1 : 0;
        if (results[i]) hits++;
    }
    return hits;
}

// Wilson test on CPU: (p-1)! + 1 ≡ 0 (mod p²)
// Cap at 10M to prevent hangs — beyond this, use segmented approach.
static bool cpu_wilson_test(uint64_t p) {
    if (p < 5 || p > 10000000ULL) return false;
    unsigned __int128 p_sq = (unsigned __int128)p * p;
    unsigned __int128 fact = 1;
    for (uint64_t i = 2; i < p; i++) {
        fact = fact * i % p_sq;
    }
    return (fact + 1) % p_sq == 0;
}

int CPUBackend::wilson_batch(const uint64_t *primes, uint8_t *results, uint32_t count) {
    int hits = 0;
    for (uint32_t i = 0; i < count; i++) {
        results[i] = cpu_wilson_test(primes[i]) ? 1 : 0;
        if (results[i]) hits++;
    }
    return hits;
}

int CPUBackend::wilson_segmented(uint64_t prime, uint32_t num_segments,
                                  uint64_t *partial_lo, uint64_t *partial_hi) {
    unsigned __int128 p_sq = (unsigned __int128)prime * prime;
    uint64_t range = prime - 2;
    uint64_t chunk = range / num_segments;

    for (uint32_t s = 0; s < num_segments; s++) {
        uint64_t start = 2 + s * chunk;
        uint64_t end = (s == num_segments - 1) ? prime : start + chunk;
        unsigned __int128 product = 1;
        for (uint64_t i = start; i < end; i++) {
            product = product * i % p_sq;
        }
        partial_lo[s] = (uint64_t)product;
        partial_hi[s] = (uint64_t)(product >> 64);
    }
    return 0;
}

// ── CPU fallback: Mersenne trial factoring ───────────────────────────
// Unpack q and mu from the packed format, compute 2^p mod q.
static bool cpu_mersenne_trial(const uint32_t *packed, uint64_t p) {
    // Unpack q as 96-bit (we only need q, not mu for CPU path)
    unsigned __int128 q = (unsigned __int128)packed[2] << 64 |
                          (unsigned __int128)packed[1] << 32 | packed[0];
    if (q < 2) return false;
    unsigned __int128 acc = 2;
    // Find top bit
    int top = 63;
    while (top > 0 && !((p >> top) & 1)) top--;
    for (int bit = top - 1; bit >= 0; bit--) {
        acc = acc * acc % q;
        if ((p >> bit) & 1)
            acc = (acc << 1) % q;
    }
    return acc == 1;
}

int CPUBackend::mersenne_trial_batch(const uint32_t *candidates, uint8_t *results,
                                      uint32_t count, uint64_t exponent) {
    int hits = 0;
    for (uint32_t i = 0; i < count; i++) {
        results[i] = cpu_mersenne_trial(candidates + i * 6, exponent) ? 1 : 0;
        if (results[i]) hits++;
    }
    return hits;
}

// ── CPU fallback: Fermat factor search ───────────────────────────────
static bool cpu_fermat_factor(const uint32_t *packed, uint64_t m) {
    unsigned __int128 q = (unsigned __int128)packed[2] << 64 |
                          (unsigned __int128)packed[1] << 32 | packed[0];
    if (q < 2) return false;
    unsigned __int128 x = 2;
    for (uint64_t i = 0; i < m; i++) {
        x = x * x % q;
    }
    return x == q - 1;
}

int CPUBackend::fermat_factor_batch(const uint32_t *candidates, uint8_t *results,
                                     uint32_t count, uint64_t fermat_index) {
    int hits = 0;
    for (uint32_t i = 0; i < count; i++) {
        results[i] = cpu_fermat_factor(candidates + i * 6, fermat_index) ? 1 : 0;
        if (results[i]) hits++;
    }
    return hits;
}

} // namespace prime
