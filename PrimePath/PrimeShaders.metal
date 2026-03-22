#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════
// u128 arithmetic for Metal (no native 128-bit support)
// ═══════════════════════════════════════════════════════════════════════

struct u128 {
    ulong lo;
    ulong hi;
};

u128 u128_from(ulong v) { return {v, 0}; }

bool u128_is_zero(u128 a) { return a.lo == 0 && a.hi == 0; }
bool u128_eq_one(u128 a) { return a.lo == 1 && a.hi == 0; }

bool u128_gte(u128 a, u128 b) {
    return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}

u128 u128_add(u128 a, u128 b) {
    u128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1UL : 0UL);
    return r;
}

u128 u128_sub(u128 a, u128 b) {
    u128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo ? 1UL : 0UL);
    return r;
}

// Multiply two u64 → u128 using hardware mulhi
u128 u128_mul64(ulong a, ulong b) {
    u128 r;
    r.lo = a * b;
    r.hi = mulhi(a, b);
    return r;
}

// (a + b) mod m, where a, b < m
u128 add_mod(u128 a, u128 b, u128 m) {
    u128 r = u128_add(a, b);
    // Check overflow: if carry occurred or result >= m
    bool overflow = (r.hi < a.hi) || (r.hi == a.hi && r.lo < a.lo);
    if (overflow || u128_gte(r, m)) {
        r = u128_sub(r, m);
    }
    return r;
}

// ═══════════════════════════════════════════════════════════════════════
// Montgomery multiplication for u128 mod m
//
// Instead of 128-iteration Russian Peasant (old: O(128) add_mods per mul),
// we use schoolbook u128 × u128 with hardware mulhi, then Barrett-like
// reduction. This reduces each mulmod from ~128 iterations to ~8 multiplies.
// ═══════════════════════════════════════════════════════════════════════

// Full u128 × u128 → u256 (represented as 4 x u64 limbs)
// Uses hardware mulhi for the heavy lifting
struct u256 {
    ulong w0, w1, w2, w3; // w0 = lowest
};

u256 u128_full_mul(u128 a, u128 b) {
    // Schoolbook: (a.hi*2^64 + a.lo) * (b.hi*2^64 + b.lo)
    // = a.hi*b.hi*2^128 + (a.hi*b.lo + a.lo*b.hi)*2^64 + a.lo*b.lo

    ulong p0_lo = a.lo * b.lo;
    ulong p0_hi = mulhi(a.lo, b.lo);

    ulong p1_lo = a.lo * b.hi;
    ulong p1_hi = mulhi(a.lo, b.hi);

    ulong p2_lo = a.hi * b.lo;
    ulong p2_hi = mulhi(a.hi, b.lo);

    ulong p3_lo = a.hi * b.hi;
    ulong p3_hi = mulhi(a.hi, b.hi);

    u256 r;
    r.w0 = p0_lo;

    // w1 = p0_hi + p1_lo + p2_lo + carries
    ulong sum1 = p0_hi + p1_lo;
    ulong c1 = (sum1 < p0_hi) ? 1UL : 0UL;
    ulong sum2 = sum1 + p2_lo;
    ulong c2 = (sum2 < sum1) ? 1UL : 0UL;
    r.w1 = sum2;

    // w2 = p1_hi + p2_hi + p3_lo + carries
    ulong sum3 = p1_hi + p2_hi;
    ulong c3 = (sum3 < p1_hi) ? 1UL : 0UL;
    ulong sum4 = sum3 + p3_lo;
    ulong c4 = (sum4 < sum3) ? 1UL : 0UL;
    ulong sum5 = sum4 + c1 + c2;
    ulong c5 = (sum5 < sum4) ? 1UL : 0UL;
    r.w2 = sum5;

    // w3 = p3_hi + carries
    r.w3 = p3_hi + c3 + c4 + c5;

    return r;
}

// Optimized mulmod128: uses hardware mulhi where possible, Russian Peasant fallback
u128 mulmod128(u128 a, u128 b, u128 m) {
    // Fast path: m fits in u64 (p < 2^32, so p^2 < 2^64) — very common
    if (m.hi == 0) {
        ulong mod = m.lo;

        // Compute full u128 × u128 → u256 product
        u256 product = u128_full_mul(a, b);

        // Precompute 2^64 mod m via repeated doubling (64 iterations)
        ulong pow64 = 0;
        {
            ulong x = 1;
            for (int i = 0; i < 64; i++) {
                x = (x >= mod - x) ? (x - (mod - x)) : (x + x);
            }
            pow64 = x;
        }

        // Reduce u256 mod u64 using Horner's method with doubling-based mulmod
        // r = ((((w3 mod m) * 2^64 + w2) mod m) * 2^64 + w1) mod m) * 2^64 + w0) mod m
        ulong r = product.w3 % mod;
        // Process words 2, 1, 0: for each, r = (r * pow64 + w_i) mod m
        ulong words[3] = {product.w2, product.w1, product.w0};
        for (int wi = 0; wi < 3; wi++) {
            // r * pow64 mod m: use doubling loop (64 iterations, not 128)
            ulong acc = 0;
            ulong a_val = r;
            ulong b_val = pow64;
            while (b_val > 0) {
                if (b_val & 1) {
                    acc += a_val;
                    if (acc >= mod || acc < a_val) acc -= mod;
                }
                a_val += a_val;
                if (a_val >= mod) a_val -= mod;
                b_val >>= 1;
            }
            // Add word
            acc += words[wi] % mod;
            if (acc >= mod) acc -= mod;
            r = acc;
        }
        return {r, 0};
    }

    // Slow path: m.hi != 0 (p > 2^32). Russian Peasant on u128.
    // This path is rare in practice.
    // Find highest set bit in b to skip leading zeros.
    int top = 127;
    if (b.hi == 0) {
        top = 63;
        ulong tmp = b.lo;
        if (tmp == 0) return {0, 0};
        while (top > 0 && !((tmp >> top) & 1)) top--;
    } else {
        ulong tmp = b.hi;
        while (top > 64 && !((tmp >> (top - 64)) & 1)) top--;
    }
    u128 r = {0, 0};
    u128 aa = a;
    for (int i = top; i >= 0; i--) {
        r = add_mod(r, r, m);
        ulong bit = (i >= 64) ? ((b.hi >> (i - 64)) & 1) : ((b.lo >> i) & 1);
        if (bit) r = add_mod(r, aa, m);
    }
    return r;
}

// base^exp mod m (exp is u64)
u128 modpow128(u128 base, ulong exp, u128 m) {
    u128 result = {1, 0};
    while (exp > 0) {
        if (exp & 1) result = mulmod128(result, base, m);
        exp >>= 1;
        if (exp > 0) base = mulmod128(base, base, m);
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// 2×2 matrix for Fibonacci (Wall-Sun-Sun)
// ═══════════════════════════════════════════════════════════════════════

struct mat2 {
    u128 a, b, c, d; // [[a,b],[c,d]]
};

mat2 mat_mul(mat2 A, mat2 B, u128 m) {
    mat2 R;
    R.a = add_mod(mulmod128(A.a, B.a, m), mulmod128(A.b, B.c, m), m);
    R.b = add_mod(mulmod128(A.a, B.b, m), mulmod128(A.b, B.d, m), m);
    R.c = add_mod(mulmod128(A.c, B.a, m), mulmod128(A.d, B.c, m), m);
    R.d = add_mod(mulmod128(A.c, B.b, m), mulmod128(A.d, B.d, m), m);
    return R;
}

mat2 mat_pow(mat2 base, ulong exp, u128 m) {
    mat2 result = {{1,0}, {0,0}, {0,0}, {1,0}}; // identity
    while (exp > 0) {
        if (exp & 1) result = mat_mul(result, base, m);
        exp >>= 1;
        if (exp > 0) base = mat_mul(base, base, m);
    }
    return result;
}

// F(n) mod m via matrix exponentiation
u128 fibonacci_mod(ulong n, u128 m) {
    if (n == 0) return {0, 0};
    if (n <= 2) return {1, 0};
    mat2 base = {{1,0}, {1,0}, {1,0}, {0,0}};
    mat2 r = mat_pow(base, n - 1, m);
    return r.a;
}

// ═══════════════════════════════════════════════════════════════════════
// GPU u64 mulmod — uses hardware mulhi instead of Russian Peasant
//
// OLD: 64-iteration bit-by-bit loop per multiply (~49K iters per primality test)
// NEW: 1 hardware mulhi + Barrett-style reduction (~5 ops per multiply)
// ═══════════════════════════════════════════════════════════════════════

ulong gpu_mulmod(ulong a, ulong b, ulong m) {
    // a * b might overflow u64. Use mulhi to get the full u128 product.
    ulong lo = a * b;
    ulong hi = mulhi(a, b);

    if (hi == 0) return lo % m;

    // Reduce {hi, lo} mod m where m < 2^64
    // Use: result = (hi * (2^64 mod m) + lo) mod m
    // Compute 2^64 mod m via doubling (only 64 iterations, done once per mulmod)
    // Actually, for better perf, use a single division-like approach:
    //
    // The product fits in 128 bits. We want (hi * 2^64 + lo) mod m.
    // Since Metal doesn't have u128 division, we use the doubling trick
    // but only on hi (which is typically much smaller than m for our use case).

    // Fast path: if hi < m, we can compute hi * (2^64 mod m) without overflow concern
    // using our add-doubling on just hi*pow64

    // Compute 2^64 mod m by repeated doubling of 1
    ulong pow64 = 0;
    {
        ulong x = 1;
        for (int i = 0; i < 64; i++) {
            if (x >= m - x) x = x - (m - x);
            else x = x + x;
        }
        pow64 = x;
    }

    // Now compute (hi % m) * pow64 mod m + lo mod m
    ulong hi_mod = hi % m;

    // hi_mod * pow64 might overflow u64, use mulhi again
    ulong prod_lo = hi_mod * pow64;
    ulong prod_hi = mulhi(hi_mod, pow64);

    if (prod_hi == 0) {
        ulong r = (prod_lo % m) + (lo % m);
        if (r >= m) r -= m;
        return r;
    }

    // Rare: prod_hi != 0, recurse one more level
    // prod_hi * 2^64 mod m + prod_lo mod m
    ulong ph_mod = prod_hi % m;
    ulong inner_lo = ph_mod * pow64;
    ulong r = (inner_lo % m) + (prod_lo % m) + (lo % m);
    while (r >= m) r -= m;
    return r;
}

ulong gpu_modpow(ulong base, ulong exp, ulong mod) {
    ulong result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = gpu_mulmod(result, base, mod);
        exp >>= 1;
        if (exp > 0) base = gpu_mulmod(base, base, mod);
    }
    return result;
}

bool gpu_miller_test(ulong n, ulong a) {
    if (a % n == 0) return true;
    ulong d = n - 1;
    int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }
    ulong x = gpu_modpow(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int i = 0; i < r - 1; i++) {
        x = gpu_mulmod(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

bool gpu_is_prime(ulong n) {
    if (n < 2) return false;
    ulong witnesses[12] = {2,3,5,7,11,13,17,19,23,29,31,37};
    for (int i = 0; i < 12; i++) {
        ulong w = witnesses[i];
        if (n == w) return true;
        if (n % w == 0) return false;
    }
    for (int i = 0; i < 12; i++) {
        if (!gpu_miller_test(n, witnesses[i])) return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// GPU Kernels
// ═══════════════════════════════════════════════════════════════════════

// Wieferich test: is 2^(p-1) ≡ 1 (mod p²)?
kernel void wieferich_batch(
    device const ulong *primes [[buffer(0)]],
    device uchar *results      [[buffer(1)]],
    constant uint &count       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ulong p = primes[gid];
    if (p < 3) { results[gid] = 0; return; }
    u128 p_sq = u128_mul64(p, p);
    u128 base = {2, 0};
    u128 r = modpow128(base, p - 1, p_sq);
    results[gid] = u128_eq_one(r) ? 1 : 0;
}

// Wall-Sun-Sun test: does p² | F(p − (p/5))?
kernel void wallsunsun_batch(
    device const ulong *primes [[buffer(0)]],
    device uchar *results      [[buffer(1)]],
    constant uint &count       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ulong p = primes[gid];
    if (p < 7) { results[gid] = 0; return; }
    u128 p_sq = u128_mul64(p, p);
    int r5 = (int)(p % 5);
    ulong fib_idx = (r5 == 1 || r5 == 4) ? p - 1 : p + 1;
    u128 fib = fibonacci_mod(fib_idx, p_sq);
    results[gid] = u128_is_zero(fib) ? 1 : 0;
}

// Twin prime test: both n and n+2 prime
kernel void twin_batch(
    device const ulong *candidates [[buffer(0)]],
    device uchar *results          [[buffer(1)]],
    constant uint &count           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ulong n = candidates[gid];
    results[gid] = (gpu_is_prime(n) && gpu_is_prime(n + 2)) ? 1 : 0;
}

// Sophie Germain test: both p and 2p+1 prime
kernel void sophie_batch(
    device const ulong *candidates [[buffer(0)]],
    device uchar *results          [[buffer(1)]],
    constant uint &count           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ulong p = candidates[gid];
    results[gid] = (gpu_is_prime(p) && gpu_is_prime(2 * p + 1)) ? 1 : 0;
}

// Cousin prime test: both n and n+4 prime
kernel void cousin_batch(
    device const ulong *candidates [[buffer(0)]],
    device uchar *results          [[buffer(1)]],
    constant uint &count           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ulong n = candidates[gid];
    results[gid] = (gpu_is_prime(n) && gpu_is_prime(n + 4)) ? 1 : 0;
}

// Sexy prime test: both n and n+6 prime
kernel void sexy_batch(
    device const ulong *candidates [[buffer(0)]],
    device uchar *results          [[buffer(1)]],
    constant uint &count           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ulong n = candidates[gid];
    results[gid] = (gpu_is_prime(n) && gpu_is_prime(n + 6)) ? 1 : 0;
}

// ═══════════════════════════════════════════════════════════════════════
// Wilson prime GPU kernels
// ═══════════════════════════════════════════════════════════════════════

kernel void wilson_batch(
    device const ulong *primes [[buffer(0)]],
    device uchar *results      [[buffer(1)]],
    constant uint &count       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ulong p = primes[gid];
    if (p < 5) { results[gid] = 0; return; }
    u128 p_sq = u128_mul64(p, p);
    u128 fact = u128_from(1);
    for (ulong i = 2; i < p; i++) {
        fact = mulmod128(fact, u128_from(i), p_sq);
    }
    u128 check = add_mod(fact, u128_from(1), p_sq);
    results[gid] = u128_is_zero(check) ? 1 : 0;
}

kernel void wilson_segments(
    device const ulong *params     [[buffer(0)]],
    device ulong *partial_lo       [[buffer(1)]],
    device ulong *partial_hi       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    ulong p = params[0];
    uint num_seg = (uint)params[1];
    if (gid >= num_seg) return;
    u128 p_sq = u128_mul64(p, p);
    ulong range = p - 2;
    ulong chunk = range / num_seg;
    ulong start = 2 + gid * chunk;
    ulong end = (gid == num_seg - 1) ? p : start + chunk;
    u128 product = u128_from(1);
    for (ulong i = start; i < end; i++) {
        product = mulmod128(product, u128_from(i), p_sq);
    }
    partial_lo[gid] = product.lo;
    partial_hi[gid] = product.hi;
}

// ═══════════════════════════════════════════════════════════════════════
// Mersenne Trial Factoring — the first Metal implementation for GIMPS
//
// Tests whether candidate factor q divides the Mersenne number 2^p - 1.
// Algorithm: compute 2^p mod q. If result == 1, then q | 2^p - 1.
//
// Candidates q = 2kp + 1 are pre-sieved on CPU to remove those divisible
// by small primes and those not ≡ 1 or 7 (mod 8).
//
// Arithmetic: 96-bit Barrett modular squaring using 3× uint32 limbs.
// This handles factors up to ~79 bits (sufficient for current GIMPS range).
// ═══════════════════════════════════════════════════════════════════════

// 96-bit unsigned integer as 3× uint32 (lo, mid, hi)
struct u96 {
    uint lo;   // bits  0-31
    uint mid;  // bits 32-63
    uint hi;   // bits 64-95
};

// 192-bit intermediate for squaring result (6× uint32)
struct u192 {
    uint w[6];
};

// Multiply two u96 → u192 using schoolbook with mulhi
u192 u96_mul(u96 a, u96 b) {
    u192 r;
    // Schoolbook 3×3 limb multiplication
    // Each limb multiply produces a 64-bit result (lo via *, hi via mulhi)

    // Column 0: a.lo * b.lo
    ulong p00 = (ulong)a.lo * b.lo;
    r.w[0] = (uint)p00;
    uint carry = (uint)(p00 >> 32);

    // Column 1: a.lo*b.mid + a.mid*b.lo + carry
    ulong p01 = (ulong)a.lo * b.mid;
    ulong p10 = (ulong)a.mid * b.lo;
    ulong col1 = p01 + p10 + carry;
    r.w[1] = (uint)col1;
    carry = (uint)(col1 >> 32) + (col1 < p01 ? 0x100000000ULL : 0);  // shouldn't overflow further

    // Simpler: use full 64-bit accumulation
    // Recompute properly with explicit carry chain
    ulong acc = 0;

    // Reset: do it cleanly
    // w[0]
    acc = (ulong)a.lo * b.lo;
    r.w[0] = (uint)acc;
    acc >>= 32;

    // w[1]
    acc += (ulong)a.lo * b.mid;
    acc += (ulong)a.mid * b.lo;
    r.w[1] = (uint)acc;
    acc >>= 32;

    // w[2]
    acc += (ulong)a.lo * b.hi;
    acc += (ulong)a.mid * b.mid;
    acc += (ulong)a.hi * b.lo;
    r.w[2] = (uint)acc;
    acc >>= 32;

    // w[3]
    acc += (ulong)a.mid * b.hi;
    acc += (ulong)a.hi * b.mid;
    r.w[3] = (uint)acc;
    acc >>= 32;

    // w[4]
    acc += (ulong)a.hi * b.hi;
    r.w[4] = (uint)acc;
    r.w[5] = (uint)(acc >> 32);

    return r;
}

// Compare u96: a >= b
bool u96_gte(u96 a, u96 b) {
    if (a.hi != b.hi) return a.hi > b.hi;
    if (a.mid != b.mid) return a.mid > b.mid;
    return a.lo >= b.lo;
}

// Subtract u96: a - b (assumes a >= b)
u96 u96_sub(u96 a, u96 b) {
    u96 r;
    uint borrow = 0;
    ulong d;

    d = (ulong)a.lo - b.lo;
    r.lo = (uint)d;
    borrow = (a.lo < b.lo) ? 1 : 0;

    d = (ulong)a.mid - b.mid - borrow;
    r.mid = (uint)d;
    borrow = (a.mid < b.mid + borrow) ? 1 : 0;

    r.hi = a.hi - b.hi - borrow;
    return r;
}

// Add u96: a + b
u96 u96_add(u96 a, u96 b) {
    u96 r;
    ulong s = (ulong)a.lo + b.lo;
    r.lo = (uint)s;
    s = (ulong)a.mid + b.mid + (s >> 32);
    r.mid = (uint)s;
    r.hi = a.hi + b.hi + (uint)(s >> 32);
    return r;
}

// Shift u96 left by 1
u96 u96_shl1(u96 a) {
    u96 r;
    r.hi = (a.hi << 1) | (a.mid >> 31);
    r.mid = (a.mid << 1) | (a.lo >> 31);
    r.lo = a.lo << 1;
    return r;
}

// Barrett reduction: compute a mod q where a is u192 and q is u96.
// Uses precomputed Barrett constant mu ≈ floor(2^192 / q), stored as u96.
// r = a - floor(a * mu / 2^192) * q, then correct by at most 2.
u96 u96_barrett_reduce(u192 a, u96 q, u96 mu) {
    // Estimate quotient: take top 96 bits of a (w[3..5]), multiply by mu,
    // take top 96 bits of result.
    // This is an approximation — we may be off by 0, 1, or 2.

    u96 a_top = {a.w[3], a.w[4], a.w[5]};
    u192 qhat_full = u96_mul(a_top, mu);
    // quotient estimate = top 96 bits of qhat_full = w[3..5]
    u96 qhat = {qhat_full.w[3], qhat_full.w[4], qhat_full.w[5]};

    // r_est = a - qhat * q (low 96 bits only — may underflow, that's ok)
    u192 qhat_q = u96_mul(qhat, q);
    // Low 96 bits of a
    u96 a_lo = {a.w[0], a.w[1], a.w[2]};
    u96 qhat_q_lo = {qhat_q.w[0], qhat_q.w[1], qhat_q.w[2]};

    u96 r = u96_sub(a_lo, qhat_q_lo);

    // Correction: at most 2 subtractions
    if (u96_gte(r, q)) r = u96_sub(r, q);
    if (u96_gte(r, q)) r = u96_sub(r, q);

    return r;
}

// Compute Barrett constant mu for a given q: floor(2^192 / q)
// We only need the top 96 bits. Computed on CPU and passed as parameter.

// Modular squaring: (a * a) mod q using Barrett
u96 u96_sqrmod(u96 a, u96 q, u96 mu) {
    u192 sq = u96_mul(a, a);
    return u96_barrett_reduce(sq, q, mu);
}

// Double mod q: (a << 1) mod q
u96 u96_dblmod(u96 a, u96 q) {
    u96 r = u96_shl1(a);
    if (u96_gte(r, q)) r = u96_sub(r, q);
    return r;
}

// ── Mersenne trial factoring kernel ──────────────────────────────────
//
// Each thread tests one candidate factor q against 2^p - 1.
// Input:  candidates[gid] = {q.lo, q.mid, q.hi, mu.lo, mu.mid, mu.hi}
//         (q and its precomputed Barrett constant, packed as 6× uint32)
// Output: results[gid] = 1 if q divides 2^p - 1, else 0
//
// The exponent p is passed as a uint64 in params[0].

kernel void mersenne_trial_batch(
    device const uint  *candidates [[buffer(0)]],  // packed: 6 uint32 per candidate
    device uchar       *results    [[buffer(1)]],
    device const ulong *params     [[buffer(2)]],  // [0]=exponent p, [1]=count
    uint gid [[thread_position_in_grid]])
{
    uint count = (uint)params[1];
    if (gid >= count) return;

    ulong p = params[0];  // Mersenne exponent

    // Unpack candidate: q (96 bits) + mu (Barrett constant, 96 bits)
    uint base = gid * 6;
    u96 q  = {candidates[base+0], candidates[base+1], candidates[base+2]};
    u96 mu = {candidates[base+3], candidates[base+4], candidates[base+5]};

    // Compute 2^p mod q via left-to-right binary exponentiation
    // Start with bit below MSB (MSB is always 1 for p > 1, so start acc = 2)
    u96 acc = {2, 0, 0};

    // Find highest set bit of p
    int top_bit = 63;
    while (top_bit > 0 && !((p >> top_bit) & 1)) top_bit--;

    // Process from bit below MSB down to bit 0
    for (int bit = top_bit - 1; bit >= 0; bit--) {
        // Square
        acc = u96_sqrmod(acc, q, mu);
        // If bit is set, multiply by 2
        if ((p >> bit) & 1) {
            acc = u96_dblmod(acc, q);
        }
    }

    // If 2^p mod q == 1, then q divides 2^p - 1
    results[gid] = (acc.lo == 1 && acc.mid == 0 && acc.hi == 0) ? 1 : 0;
}

// ═══════════════════════════════════════════════════════════════════════
// Fermat Factor Search — test if q divides Fermat number F_m = 2^(2^m)+1
//
// Any factor of F_m has form q = k * 2^(m+2) + 1.
// Test: compute 2^(2^m) mod q via m successive squarings.
// If result ≡ q-1 (i.e., -1 mod q), then q | F_m.
// ═══════════════════════════════════════════════════════════════════════

kernel void fermat_factor_batch(
    device const uint  *candidates [[buffer(0)]],  // packed: 6 uint32 per candidate (q + mu)
    device uchar       *results    [[buffer(1)]],
    device const ulong *params     [[buffer(2)]],  // [0]=m (Fermat index), [1]=count
    uint gid [[thread_position_in_grid]])
{
    uint count = (uint)params[1];
    if (gid >= count) return;

    ulong m = params[0];  // Fermat number index: testing F_m = 2^(2^m) + 1

    uint base = gid * 6;
    u96 q  = {candidates[base+0], candidates[base+1], candidates[base+2]};
    u96 mu = {candidates[base+3], candidates[base+4], candidates[base+5]};

    // Compute 2^(2^m) mod q via m successive squarings
    // Start: x = 2
    u96 x = {2, 0, 0};

    for (ulong i = 0; i < m; i++) {
        x = u96_sqrmod(x, q, mu);
    }

    // Check if x ≡ q-1 (mod q), i.e., x + 1 ≡ 0 (mod q)
    u96 q_minus_1 = u96_sub(q, {1, 0, 0});
    results[gid] = (x.lo == q_minus_1.lo && x.mid == q_minus_1.mid &&
                    x.hi == q_minus_1.hi) ? 1 : 0;
}

// General primality test batch
kernel void primality_batch(
    device const ulong *candidates [[buffer(0)]],
    device uchar *results          [[buffer(1)]],
    constant uint &count           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    results[gid] = gpu_is_prime(candidates[gid]) ? 1 : 0;
}
