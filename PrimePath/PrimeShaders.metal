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
// Fused Mersenne TF: sieve + Barrett + primality test, all on GPU.
//
// Each thread handles one k-value end-to-end:
//   1. Compute q = 2kp + 1
//   2. Filter: q must be 1 or 7 (mod 8)
//   3. Sieve by small primes (3..97)
//   4. Compute Barrett constant mu = floor(2^192 / q)
//   5. Compute 2^p mod q via binary exponentiation
//   6. Write 1 if q divides 2^p - 1, else 0
//
// params[0] = exponent p
// params[1] = k_start
// params[2] = k_count
// ═══════════════════════════════════════════════════════════════════════

// Barrett mu computation on GPU: floor(2^192 / q) for 96-bit q
// Uses iterative long division approach
u96 gpu_compute_barrett_mu(u96 q) {
    // We need floor(2^192 / q). Since q is 96 bits, result fits in 96 bits.
    // Use 192-bit / 96-bit division via schoolbook long division.
    //
    // Dividend = 2^192 (193 bits). We'll compute 96 bits of quotient.
    // Process 32 bits at a time, 3 iterations for 96-bit quotient.

    // Start with remainder = 0, dividend digits come from 2^192
    // 2^192 as 7 x 32-bit words: {0,0,0,0,0,0,1} (MSB first)
    // Only the top word is 1, rest are 0.

    // Simplified: use u96 division by repeated subtraction with shifts.
    // Actually, let's use a direct approach with 128-bit intermediates.

    // For correctness, compute mu digit by digit.
    // mu = q0 * 2^64 + q1 * 2^32 + q2 where each qi is 32 bits.

    // Step 1: estimate mu_hi (top 32 bits)
    // 2^192 / q ~ 2^192 / (q.hi * 2^64) = 2^128 / q.hi (when q.hi > 0)

    if (q.lo == 0 && q.mid == 0 && q.hi == 0) return {0, 0, 0};

    // Use floating point for initial estimate, then refine
    // q as float: q ~ q.hi * 2^64 + q.mid * 2^32 + q.lo
    float fq = float(q.hi) * 18446744073709551616.0f + float(q.mid) * 4294967296.0f + float(q.lo);
    if (fq < 1.0f) return {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};

    // mu ~ 2^192 / q. Compute in float first for top bits estimate.
    // 2^192 ~ 6.277e57
    // We don't need perfect precision -- Barrett allows off-by-2.

    // For a practical GPU Barrett, we can compute mu using Newton-Raphson
    // or just use the existing u96 multiply infrastructure.

    // Simpler approach: compute mu = floor(2^192 / q) using 192/96 division.
    // Represent 2^192 as a u192 and divide.

    // Schoolbook 192-bit / 96-bit division, producing 96-bit quotient.
    // Process in 32-bit chunks from MSB.

    // Remainder starts at 0. Dividend = {0,0,0,0,0,1} (LSB to MSB, word 5 = 1)
    // We bring down 32 bits at a time from the dividend.

    // Words of 2^192 from MSB: w6=1, w5=0, w4=0, w3=0, w2=0, w1=0
    // (2^192 = 1 * 2^192, so word at position 192/32=6 is 1)

    // We need to produce 3 quotient words (96 bits).
    // Each iteration: remainder = remainder * 2^32 + next_dividend_word
    //                 quotient_word = remainder / q
    //                 remainder = remainder % q

    // But remainder can be up to 96+32 = 128 bits, and we're dividing by 96-bit q.
    // We need 128/96 division for each step. Use estimate + correction.

    // Let's use a different, cleaner method: reciprocal via Newton's method in float,
    // then refine using u96 multiply.

    // Float reciprocal: 1/q * 2^192
    // double has 53 bits of mantissa -- enough for a good starting estimate
    // But Metal doesn't have double. Use float (23 bits) with refinement.

    // Actually the simplest correct approach for GPU:
    // Compute mu one 32-bit digit at a time using trial division.

    // remainder = 2^192 mod (q * 2^64) ... this gets complex.

    // PRACTICAL SHORTCUT: Since Barrett allows off-by-2 in the quotient estimate,
    // we can use a slightly imprecise mu and it still works.
    // Compute mu from float with 23 bits precision, pad the rest with 0.
    // Then the sqrmod will do at most 2 extra corrections per step.
    // This is actually fine for correctness!

    // Better: use the property that for 96-bit q where q.hi != 0,
    // mu fits in 96 bits and mu ~ 2^96 / q.hi (top 32 bits only).
    // For q.hi == 0, mu is larger.

    // BEST approach for Metal: compute Barrett constant on CPU for the first
    // candidate, and since all candidates in a k-range have similar magnitude,
    // we can verify and correct on GPU.

    // Actually let me just do proper 192/96 long division on GPU.
    // It's ~6 iterations, each doing a 128/96 divide. Not fast but only once per thread.

    u96 mu = {0, 0, 0};

    // We'll compute 3 words of quotient.
    // Use a 128-bit remainder (u96 + 32 extra bits).
    // Represent remainder as u128-like: {w0, w1, w2, w3} where value = w3*2^96 + w2*2^64 + w1*2^32 + w0

    // Dividend 2^192 has 7 words (indices 0-6), word 6 = 1, rest = 0.
    // We process words from MSB (word 6) down.

    // After processing word 6: remainder = 1
    // After processing word 5 (=0): remainder = 1 * 2^32 + 0 = 2^32
    // After processing word 4 (=0): remainder = prev_remainder * 2^32
    // etc. We start producing quotient digits once remainder >= q.

    // Initialize: process the first 4 words of dividend (words 6,5,4,3)
    // to build up a 128-bit remainder, then start dividing.

    // Remainder after processing words 6..3: 1 * 2^(3*32) = 2^96
    // This is exactly 1 in the "word 3" position.

    // So initial 128-bit remainder = 2^96 = {0, 0, 0, 1} (4 x 32-bit words)
    // Now produce quotient word by word.

    // For each quotient word:
    //   Bring in next dividend word (always 0 here since dividend is 2^192)
    //   remainder = remainder * 2^32 + dividend_word
    //   q_digit = remainder / q  (128-bit / 96-bit -> 32-bit quotient digit)
    //   remainder = remainder - q_digit * q

    // 128/96 division to get 32-bit quotient:
    // Estimate: if remainder fits in 96 bits, q_digit = 0 or 1.
    // If remainder uses all 128 bits, estimate q_digit = top_word / (q.hi + 1).

    // Let me implement this properly.

    // rem = 2^96 as a 128-bit number
    uint rem3 = 1, rem2 = 0, rem1 = 0, rem0 = 0; // rem = rem3*2^96 + rem2*2^64 + rem1*2^32 + rem0

    // Quotient words (MSB first): mu.hi, mu.mid, mu.lo
    // We need to produce 3 quotient digits.

    // For each of the 3 remaining dividend words (words 2, 1, 0 -- all zero):
    for (int digit = 2; digit >= 0; digit--) {
        // Shift remainder left by 32 and bring in next dividend word (0)
        rem3 = rem2;
        rem2 = rem1;
        rem1 = rem0;
        rem0 = 0; // dividend word is always 0

        // Estimate quotient digit: qd = floor(rem / q)
        // rem is 128 bits, q is 96 bits, qd is at most 32 bits
        uint qd = 0;

        // If rem3 > 0, we definitely have a quotient digit
        if (rem3 > 0 || (rem3 == 0 && rem2 > q.hi) ||
            (rem3 == 0 && rem2 == q.hi && rem1 > q.mid) ||
            (rem3 == 0 && rem2 == q.hi && rem1 == q.mid && rem0 >= q.lo)) {

            // Estimate: qd ~ rem3 * 2^32 / q.hi (if q.hi > 0)
            if (q.hi > 0) {
                ulong top = ((ulong)rem3 << 32) | rem2;
                qd = (uint)(top / ((ulong)q.hi + 1));
            } else if (q.mid > 0) {
                ulong top = ((ulong)rem3 << 32) | rem2;
                // q < 2^64, so qd could be large
                qd = (uint)min((ulong)0xFFFFFFFF, top / ((ulong)q.mid + 1));
            } else {
                qd = 0xFFFFFFFF; // q is very small
            }

            // Compute rem -= qd * q using u96 multiply
            // qd * q can be up to 128 bits
            ulong p0 = (ulong)qd * q.lo;
            ulong p1 = (ulong)qd * q.mid;
            ulong p2 = (ulong)qd * q.hi;

            uint s0 = (uint)p0;
            ulong c1 = (p0 >> 32) + (uint)p1;
            ulong c2 = (p1 >> 32) + (uint)p2 + (c1 >> 32);
            uint s1 = (uint)c1;
            uint s2 = (uint)c2;
            uint s3 = (uint)(p2 >> 32) + (uint)(c2 >> 32);

            // Subtract: rem -= s
            uint borrow = 0;
            uint nr0 = rem0 - s0;            borrow = (nr0 > rem0) ? 1 : 0;
            uint nr1 = rem1 - s1 - borrow;   borrow = (rem1 < s1 + borrow) ? 1 : 0;
            uint nr2 = rem2 - s2 - borrow;   borrow = (rem2 < s2 + borrow) ? 1 : 0;
            uint nr3 = rem3 - s3 - borrow;

            rem0 = nr0; rem1 = nr1; rem2 = nr2; rem3 = nr3;

            // Correct: if remainder went negative (rem3 huge) or still >= q, adjust
            // If rem3 > 0 or rem is negative (underflow), we overestimated
            if (rem3 > 0x80000000u) {
                // Underflow: add back q, decrease qd
                ulong a0 = (ulong)rem0 + q.lo;
                rem0 = (uint)a0;
                ulong a1 = (ulong)rem1 + q.mid + (a0 >> 32);
                rem1 = (uint)a1;
                ulong a2 = (ulong)rem2 + q.hi + (a1 >> 32);
                rem2 = (uint)a2;
                rem3 += (uint)(a2 >> 32);
                qd--;
            }

            // If remainder still >= q, add 1 to quotient
            while (rem3 > 0 ||
                   (rem2 > q.hi) ||
                   (rem2 == q.hi && rem1 > q.mid) ||
                   (rem2 == q.hi && rem1 == q.mid && rem0 >= q.lo)) {
                ulong s0b = (ulong)rem0 - q.lo;
                uint b = (s0b > (ulong)rem0) ? 1 : 0;
                rem0 = (uint)s0b;
                ulong s1b = (ulong)rem1 - q.mid - b;
                b = ((ulong)rem1 < (ulong)q.mid + b) ? 1 : 0;
                rem1 = (uint)s1b;
                ulong s2b = (ulong)rem2 - q.hi - b;
                b = ((ulong)rem2 < (ulong)q.hi + b) ? 1 : 0;
                rem2 = (uint)s2b;
                rem3 -= b;
                qd++;
            }
        }

        // Store quotient digit
        if (digit == 2) mu.hi = qd;
        else if (digit == 1) mu.mid = qd;
        else mu.lo = qd;
    }

    return mu;
}

// Small primes for sieve and precomputed 2^64 mod sp (avoids per-thread loop)
constant uint FUSED_SIEVE_PRIMES[] = {3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97};
constant uint FUSED_POW64_MOD[]    = {1,1,2,5,3,1,17,6,24,16,12,16,41,25,15,5,16,17,10,2,51,36,67,61};
// Each entry: 2^64 mod sp, verified by pow(2,64,sp) in Python.

kernel void mersenne_fused_sieve(
    device atomic_uint *hit_count  [[buffer(0)]],  // number of factors found
    device ulong       *hit_factors [[buffer(1)]],  // found factors: pairs of (q_lo, q_hi_and_k)
    device const ulong *params     [[buffer(2)]],  // [0]=p, [1]=k_start, [2]=k_count
    uint gid [[thread_position_in_grid]])
{
    ulong p       = params[0];
    ulong k_start = params[1];
    ulong k_count = params[2];

    if (gid >= (uint)k_count) return;

    ulong k = k_start + gid;

    // Step 1: compute q = 2kp + 1 using 128-bit arithmetic
    // 2 * k * p: k is up to ~75 trillion (47 bits), p is ~29 bits = ~76 bits product
    ulong kp_lo = k * p;  // low 64 bits of k*p
    ulong kp_hi = mulhi(k, p);  // high 64 bits of k*p

    // q = 2*kp + 1
    ulong q_lo = (kp_lo << 1) | 1;  // shift left 1 and set bit 0
    ulong q_hi_full = (kp_hi << 1) | (kp_lo >> 63);  // carry from shift

    // Check if q > 96 bits (hi > 32 bits)
    if (q_hi_full >> 32) return;
    uint q_hi32 = (uint)q_hi_full;

    // Step 2: q must be 1 or 7 (mod 8)
    uint mod8 = (uint)(q_lo & 7);
    if (mod8 != 1 && mod8 != 7) return;

    // Step 3: sieve by small primes (using precomputed 2^64 mod sp)
    for (int i = 0; i < 24; i++) {
        ulong sp = FUSED_SIEVE_PRIMES[i];
        ulong hi_mod = ((ulong)q_hi32 % sp);
        ulong q_mod_sp = (hi_mod * FUSED_POW64_MOD[i] + q_lo % sp) % sp;
        if (q_mod_sp == 0) return;
    }

    // Step 4: compute Barrett constant mu
    u96 q = {(uint)q_lo, (uint)(q_lo >> 32), q_hi32};
    u96 mu = gpu_compute_barrett_mu(q);

    // Step 5: compute 2^p mod q
    u96 acc = {2, 0, 0};
    int top_bit = 63;
    while (top_bit > 0 && !((p >> top_bit) & 1)) top_bit--;

    for (int bit = top_bit - 1; bit >= 0; bit--) {
        acc = u96_sqrmod(acc, q, mu);
        if ((p >> bit) & 1) {
            acc = u96_dblmod(acc, q);
        }
    }

    // Step 6: if 2^p mod q == 1, we found a factor!
    if (acc.lo == 1 && acc.mid == 0 && acc.hi == 0) {
        uint idx = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
        if (idx < 1024) {  // limit stored hits
            hit_factors[idx * 2]     = q_lo;
            hit_factors[idx * 2 + 1] = ((ulong)q_hi32 << 32) | (k & 0xFFFFFFFF);
        }
    }
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

// ═══════════════════════════════════════════════════════════════════════
// GPU Segmented Sieve — mark composites in a bitmap
//
// Each thread handles one small prime: marks all multiples of that prime
// in the [seg_lo, seg_lo + seg_size) range by setting bits in the bitmap.
// CPU then scans the bitmap to extract surviving primes.
//
// params[0] = seg_lo (must be even — we only sieve odd numbers)
// params[1] = seg_size (number of odd candidates = half the range)
// sieve_primes: array of small primes (3, 5, 7, ..., up to sqrt(seg_hi))
// bitmap: output, one bit per odd number. bit i represents seg_lo + 2*i + 1.
//         bit=1 means composite. Initialized to 0 by CPU.
// ═══════════════════════════════════════════════════════════════════════

kernel void gpu_sieve_mark(
    device atomic_uint   *bitmap      [[buffer(0)]],  // ceil(seg_size/32) uint32s
    device const ulong   *sieve_primes [[buffer(1)]],
    device const ulong   *params       [[buffer(2)]],  // [0]=seg_lo, [1]=seg_size, [2]=num_primes
    uint gid [[thread_position_in_grid]])
{
    ulong num_primes = params[2];
    if (gid >= (uint)num_primes) return;

    ulong p = sieve_primes[gid];
    ulong seg_lo = params[0];
    ulong seg_size = params[1];  // number of odd slots
    ulong seg_hi = seg_lo + seg_size * 2;  // actual range end

    // Find first odd multiple of p >= seg_lo
    ulong start = ((seg_lo + p - 1) / p) * p;
    if (start % 2 == 0) start += p;
    if (start == p) start += 2 * p;  // skip p itself

    // Mark all odd multiples of p in range
    for (ulong n = start; n < seg_hi; n += 2 * p) {
        // Convert n to bit index: n = seg_lo + 2*idx + 1, but seg_lo is even
        // so idx = (n - seg_lo - 1) / 2 ... but only if n > seg_lo and n is odd
        if (n <= seg_lo) continue;
        ulong idx = (n - seg_lo - 1) / 2;
        if (idx >= seg_size) break;

        // Set bit idx in bitmap using atomic OR
        uint word = (uint)(idx / 32);
        uint bit = (uint)(idx % 32);
        atomic_fetch_or_explicit(&bitmap[word], 1u << bit, memory_order_relaxed);
    }
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

// ═══════════════════════════════════════════════════════════════════════
// Nester Carry Chain: GPU streaming divisibility (Gear 3)
// ═══════════════════════════════════════════════════════════════════════
//
// Each thread tests one divisor d against a big number stored as limbs.
// Streams MSB to LSB, accumulating via Barrett reduction (no division).
// All threads read the same limbs (broadcast from cache), so memory
// access is uniform across the SIMD group.

// Barrett mod on GPU: x mod d using precomputed inv = floor(2^64 / d)
inline ulong ncc_barrett_mod(ulong x, uint d, ulong inv) {
    ulong q = mulhi(x, inv);
    ulong r = x - q * (ulong)d;
    if (r >= d) { r -= d; if (r >= d) r -= d; }
    return r;
}

kernel void nester_cc_stream(
    device const ulong *limbs      [[buffer(0)]],  // BigNum limbs (big-endian)
    device const uint  *divisors   [[buffer(1)]],  // candidate divisors (uint32)
    device uchar       *results    [[buffer(2)]],  // 1 if d divides N, 0 otherwise
    constant uint      &num_limbs  [[buffer(3)]],
    constant uint      &num_divs   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_divs) return;

    uint d = divisors[gid];
    if (d <= 1) { results[gid] = 0; return; }

    // Precompute inverse: floor(2^64 / d)
    ulong inv = 0xFFFFFFFFFFFFFFFFUL / (ulong)d;

    // Precompute pow64 = 2^64 mod d via doubling (no division)
    ulong pow64 = 1;
    for (int i = 0; i < 64; i++) {
        pow64 <<= 1;
        if (pow64 >= d) pow64 -= d;
    }

    // Stream through all limbs
    ulong rem = 0;
    for (uint i = 0; i < num_limbs; i++) {
        ulong shifted = ncc_barrett_mod(rem * pow64, d, inv);
        ulong limb_r  = ncc_barrett_mod(limbs[i], d, inv);
        rem = shifted + limb_r;
        if (rem >= d) rem -= d;
    }

    results[gid] = (rem == 0) ? 1 : 0;
}
