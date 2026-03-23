// MetalModArith.metal
// Standalone GPU modular arithmetic library for Apple Silicon (Metal)
//
// Provides multi-precision modular arithmetic primitives that run on the GPU.
// Designed for number theory, cryptography, and related batch computations.
//
// Types:  u96 (3x uint32), u128 (2x uint64), u192 (6x uint32), u256 (4x uint64)
// Ops:    add, sub, mul, shift, compare, Barrett reduction, modular multiply,
//         modular exponentiation, Miller-Rabin primality testing
//
// All functions are device-side. Application kernels should be defined in your
// own .metal files and call these primitives.

#include <metal_stdlib>
using namespace metal;

// =========================================================================
// u128 -- 128-bit unsigned integer (2x uint64 limbs)
// =========================================================================

struct u128 {
    ulong lo;
    ulong hi;
};

/// Construct u128 from a 64-bit value.
u128 u128_from(ulong v) { return {v, 0}; }

/// True if the value is zero.
bool u128_is_zero(u128 a) { return a.lo == 0 && a.hi == 0; }

/// True if the value equals one.
bool u128_eq_one(u128 a) { return a.lo == 1 && a.hi == 0; }

/// Greater-than-or-equal comparison.
bool u128_gte(u128 a, u128 b) {
    return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}

/// 128-bit addition. No overflow detection -- caller must ensure no carry-out
/// is needed, or use add_mod for modular addition.
u128 u128_add(u128 a, u128 b) {
    u128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1UL : 0UL);
    return r;
}

/// 128-bit subtraction. Caller must ensure a >= b (or interpret as wrap).
u128 u128_sub(u128 a, u128 b) {
    u128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo ? 1UL : 0UL);
    return r;
}

/// Multiply two 64-bit values into a 128-bit result using hardware mulhi.
u128 u128_mul64(ulong a, ulong b) {
    u128 r;
    r.lo = a * b;
    r.hi = mulhi(a, b);
    return r;
}

/// Modular addition: (a + b) mod m, where a, b < m.
u128 u128_addmod(u128 a, u128 b, u128 m) {
    u128 r = u128_add(a, b);
    bool overflow = (r.hi < a.hi) || (r.hi == a.hi && r.lo < a.lo);
    if (overflow || u128_gte(r, m)) {
        r = u128_sub(r, m);
    }
    return r;
}

// =========================================================================
// u256 -- 256-bit unsigned integer (4x uint64 limbs), used as intermediate
// =========================================================================

struct u256 {
    ulong w0, w1, w2, w3; // w0 = least significant
};

/// Full 128x128 -> 256-bit multiplication using schoolbook with hardware mulhi.
u256 u128_full_mul(u128 a, u128 b) {
    // (a.hi*2^64 + a.lo) * (b.hi*2^64 + b.lo)
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

    // w1 = p0_hi + p1_lo + p2_lo
    ulong sum1 = p0_hi + p1_lo;
    ulong c1 = (sum1 < p0_hi) ? 1UL : 0UL;
    ulong sum2 = sum1 + p2_lo;
    ulong c2 = (sum2 < sum1) ? 1UL : 0UL;
    r.w1 = sum2;

    // w2 = p1_hi + p2_hi + p3_lo + carries from w1
    ulong sum3 = p1_hi + p2_hi;
    ulong c3 = (sum3 < p1_hi) ? 1UL : 0UL;
    ulong sum4 = sum3 + p3_lo;
    ulong c4 = (sum4 < sum3) ? 1UL : 0UL;
    ulong sum5 = sum4 + c1 + c2;
    ulong c5 = (sum5 < sum4) ? 1UL : 0UL;
    r.w2 = sum5;

    // w3 = p3_hi + remaining carries
    r.w3 = p3_hi + c3 + c4 + c5;

    return r;
}

// =========================================================================
// mulmod128 -- modular multiplication for 128-bit values
//
// Two paths:
//   Fast: when modulus fits in 64 bits, uses Horner reduction of u256 product.
//   Slow: when modulus is full 128-bit, uses Russian Peasant (add-and-double).
// =========================================================================

/// Compute (a * b) mod m for 128-bit operands.
u128 mulmod128(u128 a, u128 b, u128 m) {
    // --- Fast path: m fits in u64 ---
    if (m.hi == 0) {
        ulong mod = m.lo;
        u256 product = u128_full_mul(a, b);

        // Precompute 2^64 mod m via repeated doubling
        ulong pow64 = 0;
        {
            ulong x = 1;
            for (int i = 0; i < 64; i++) {
                x = (x >= mod - x) ? (x - (mod - x)) : (x + x);
            }
            pow64 = x;
        }

        // Horner reduction: ((w3 * 2^64 + w2) * 2^64 + w1) * 2^64 + w0) mod m
        ulong r = product.w3 % mod;
        ulong words[3] = {product.w2, product.w1, product.w0};
        for (int wi = 0; wi < 3; wi++) {
            // r * pow64 mod m via binary method
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
            acc += words[wi] % mod;
            if (acc >= mod) acc -= mod;
            r = acc;
        }
        return {r, 0};
    }

    // --- Slow path: full 128-bit modulus, Russian Peasant ---
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
        r = u128_addmod(r, r, m);
        ulong bit = (i >= 64) ? ((b.hi >> (i - 64)) & 1) : ((b.lo >> i) & 1);
        if (bit) r = u128_addmod(r, aa, m);
    }
    return r;
}

// =========================================================================
// modpow128 -- modular exponentiation: base^exp mod m
//
// Exponent is 64-bit. For 128-bit exponents, chain two calls or extend.
// =========================================================================

/// Compute base^exp mod m using binary (square-and-multiply) method.
u128 modpow128(u128 base, ulong exp, u128 m) {
    u128 result = {1, 0};
    while (exp > 0) {
        if (exp & 1) result = mulmod128(result, base, m);
        exp >>= 1;
        if (exp > 0) base = mulmod128(base, base, m);
    }
    return result;
}

// =========================================================================
// u96 -- 96-bit unsigned integer (3x uint32 limbs)
//
// Useful for Mersenne trial factoring and other contexts where 64 bits is
// not enough but 128 bits is overkill.
// =========================================================================

struct u96 {
    uint lo;   // bits  0-31
    uint mid;  // bits 32-63
    uint hi;   // bits 64-95
};

/// 192-bit intermediate result for u96 multiplication.
struct u192 {
    uint w[6]; // w[0] = least significant
};

/// Greater-than-or-equal comparison for u96.
bool u96_gte(u96 a, u96 b) {
    if (a.hi != b.hi) return a.hi > b.hi;
    if (a.mid != b.mid) return a.mid > b.mid;
    return a.lo >= b.lo;
}

/// Subtraction: a - b. Caller must ensure a >= b.
u96 u96_sub(u96 a, u96 b) {
    u96 r;
    uint borrow = 0;

    ulong d = (ulong)a.lo - b.lo;
    r.lo = (uint)d;
    borrow = (a.lo < b.lo) ? 1 : 0;

    d = (ulong)a.mid - b.mid - borrow;
    r.mid = (uint)d;
    borrow = (a.mid < b.mid + borrow) ? 1 : 0;

    r.hi = a.hi - b.hi - borrow;
    return r;
}

/// Addition: a + b. No overflow detection.
u96 u96_add(u96 a, u96 b) {
    u96 r;
    ulong s = (ulong)a.lo + b.lo;
    r.lo = (uint)s;
    s = (ulong)a.mid + b.mid + (s >> 32);
    r.mid = (uint)s;
    r.hi = a.hi + b.hi + (uint)(s >> 32);
    return r;
}

/// Left shift by 1 bit.
u96 u96_shl1(u96 a) {
    u96 r;
    r.hi = (a.hi << 1) | (a.mid >> 31);
    r.mid = (a.mid << 1) | (a.lo >> 31);
    r.lo = a.lo << 1;
    return r;
}

/// Schoolbook 96x96 -> 192-bit multiplication using hardware uint32 multiply.
u192 u96_mul(u96 a, u96 b) {
    u192 r;
    ulong acc = 0;

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

/// Barrett reduction: compute a mod q where a is u192 and q is u96.
///
/// Requires precomputed Barrett constant mu = floor(2^192 / q).
/// Result is exact (corrected by at most 2 subtractions).
u96 u96_barrett_reduce(u192 a, u96 q, u96 mu) {
    // Estimate quotient from top 96 bits of a multiplied by mu
    u96 a_top = {a.w[3], a.w[4], a.w[5]};
    u192 qhat_full = u96_mul(a_top, mu);
    u96 qhat = {qhat_full.w[3], qhat_full.w[4], qhat_full.w[5]};

    // Compute remainder estimate from low 96 bits
    u192 qhat_q = u96_mul(qhat, q);
    u96 a_lo = {a.w[0], a.w[1], a.w[2]};
    u96 qhat_q_lo = {qhat_q.w[0], qhat_q.w[1], qhat_q.w[2]};

    u96 r = u96_sub(a_lo, qhat_q_lo);

    // At most 2 correction steps
    if (u96_gte(r, q)) r = u96_sub(r, q);
    if (u96_gte(r, q)) r = u96_sub(r, q);

    return r;
}

/// Modular squaring: (a^2) mod q using Barrett reduction.
u96 u96_sqrmod(u96 a, u96 q, u96 mu) {
    u192 sq = u96_mul(a, a);
    return u96_barrett_reduce(sq, q, mu);
}

/// Double mod: (2*a) mod q via shift-and-subtract.
u96 u96_dblmod(u96 a, u96 q) {
    u96 r = u96_shl1(a);
    if (u96_gte(r, q)) r = u96_sub(r, q);
    return r;
}

// =========================================================================
// gpu_mulmod64 -- 64-bit modular multiplication using hardware mulhi
//
// Uses the hardware mulhi instruction to compute the full 128-bit product
// of two 64-bit values, then reduces mod m without needing a software u128
// division. Handles multi-level overflow for very large products.
// =========================================================================

/// Compute (a * b) mod m for 64-bit operands.
ulong gpu_mulmod64(ulong a, ulong b, ulong m) {
    ulong lo = a * b;
    ulong hi = mulhi(a, b);

    if (hi == 0) return lo % m;

    // Reduce {hi, lo} mod m: result = (hi * (2^64 mod m) + lo) mod m
    // Compute 2^64 mod m by doubling 1 sixty-four times
    ulong pow64 = 0;
    {
        ulong x = 1;
        for (int i = 0; i < 64; i++) {
            if (x >= m - x) x = x - (m - x);
            else x = x + x;
        }
        pow64 = x;
    }

    ulong hi_mod = hi % m;

    // hi_mod * pow64 may overflow u64 -- use mulhi again
    ulong prod_lo = hi_mod * pow64;
    ulong prod_hi = mulhi(hi_mod, pow64);

    if (prod_hi == 0) {
        ulong r = (prod_lo % m) + (lo % m);
        if (r >= m) r -= m;
        return r;
    }

    // Rare third-level reduction
    ulong ph_mod = prod_hi % m;
    ulong inner_lo = ph_mod * pow64;
    ulong r = (inner_lo % m) + (prod_lo % m) + (lo % m);
    while (r >= m) r -= m;
    return r;
}

// =========================================================================
// gpu_modpow64 -- 64-bit modular exponentiation
// =========================================================================

/// Compute base^exp mod m for 64-bit operands.
ulong gpu_modpow64(ulong base, ulong exp, ulong mod) {
    ulong result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = gpu_mulmod64(result, base, mod);
        exp >>= 1;
        if (exp > 0) base = gpu_mulmod64(base, base, mod);
    }
    return result;
}

// =========================================================================
// Miller-Rabin primality testing (deterministic for n < 3.3e24)
//
// Uses the first 12 prime witnesses {2..37}, which gives a deterministic
// result for all 64-bit integers.
// =========================================================================

/// Single-witness Miller-Rabin test. Returns true if n is probably prime
/// with respect to witness a.
bool gpu_miller_test(ulong n, ulong a) {
    if (a % n == 0) return true;
    ulong d = n - 1;
    int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }
    ulong x = gpu_modpow64(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int i = 0; i < r - 1; i++) {
        x = gpu_mulmod64(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

/// Deterministic primality test for any 64-bit integer.
/// Uses 12 witnesses, which is sufficient for all values below 2^64.
bool gpu_is_prime(ulong n) {
    if (n < 2) return false;
    ulong witnesses[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
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

// =========================================================================
// Example kernels -- ready-to-use batch compute kernels
//
// These demonstrate how to build GPU kernels on top of the primitives above.
// You can include these directly or use them as templates for your own.
// =========================================================================

/// Batch modular exponentiation: results[i] = bases[i]^exponents[i] mod moduli[i]
/// Each input element is packed as 4x uint64: {base.lo, base.hi, exponent, mod.lo, mod.hi}
/// but for simplicity we use separate buffers with u64 base, exp, mod.
///
/// Input layout (buffer 0): interleaved triples [base, exp, mod] as uint64
/// Output layout (buffer 1): results as 2x uint64 [lo, hi] per element
/// Buffer 2: element count
kernel void modpow_batch(
    device const ulong *input   [[buffer(0)]],  // [base, exp, mod] x count
    device ulong       *output  [[buffer(1)]],  // [result_lo, result_hi] x count
    constant uint      &count   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    ulong base_val = input[gid * 3 + 0];
    ulong exp_val  = input[gid * 3 + 1];
    ulong mod_val  = input[gid * 3 + 2];

    u128 base_u128 = u128_from(base_val);
    u128 mod_u128  = u128_from(mod_val);

    u128 result = modpow128(base_u128, exp_val, mod_u128);

    output[gid * 2 + 0] = result.lo;
    output[gid * 2 + 1] = result.hi;
}

/// Batch Miller-Rabin primality test.
/// Input: array of uint64 candidates. Output: array of uint8 (1 = prime, 0 = composite).
kernel void primality_batch(
    device const ulong *candidates [[buffer(0)]],
    device uchar       *results    [[buffer(1)]],
    constant uint      &count      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    results[gid] = gpu_is_prime(candidates[gid]) ? 1 : 0;
}

/// Batch 64-bit modular multiplication.
/// Input layout (buffer 0): interleaved triples [a, b, mod] as uint64
/// Output layout (buffer 1): uint64 results
kernel void mulmod_batch(
    device const ulong *input   [[buffer(0)]],  // [a, b, mod] x count
    device ulong       *output  [[buffer(1)]],  // result x count
    constant uint      &count   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    ulong a   = input[gid * 3 + 0];
    ulong b   = input[gid * 3 + 1];
    ulong mod = input[gid * 3 + 2];

    output[gid] = gpu_mulmod64(a, b, mod);
}
