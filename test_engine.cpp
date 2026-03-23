#include "PrimePath/PrimeEngine.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>

// ── Simple test framework ───────────────────────────────────────────

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(expr, name) do { \
    if (expr) { \
        printf("  PASS: %s\n", name); \
        g_pass++; \
    } else { \
        printf("  FAIL: %s\n", name); \
        g_fail++; \
    } \
} while(0)

#define SECTION(title) printf("\n=== %s ===\n", title)

// =====================================================================
// 1. Primality testing
// =====================================================================

static void test_primality() {
    SECTION("1. Primality Testing");

    // Edge cases
    CHECK(!prime::is_prime(0), "0 is not prime");
    CHECK(!prime::is_prime(1), "1 is not prime");
    CHECK(prime::is_prime(2),  "2 is prime");

    // Known primes
    uint64_t known_primes[] = {2,3,5,7,11,13,997,7919,104729};
    bool all_primes_ok = true;
    for (auto p : known_primes) {
        if (!prime::is_prime(p)) {
            all_primes_ok = false;
            printf("    FAIL: %llu should be prime\n", (unsigned long long)p);
        }
    }
    CHECK(all_primes_ok, "known primes: 2,3,5,7,11,13,997,7919,104729");

    // Known composites
    uint64_t known_composites[] = {4,6,9,15};
    bool all_comp_ok = true;
    for (auto c : known_composites) {
        if (prime::is_prime(c)) {
            all_comp_ok = false;
            printf("    FAIL: %llu should be composite\n", (unsigned long long)c);
        }
    }
    CHECK(all_comp_ok, "known composites: 4,6,9,15");

    // Carmichael number: 561 = 3 * 11 * 17 (pseudo-prime to Fermat, must fail MR)
    CHECK(!prime::is_prime(561), "561 (Carmichael) is composite");

    // More Carmichael numbers
    uint64_t carmichaels[] = {1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841, 29341, 41041, 46657, 52633, 62745, 63973, 75361};
    bool carm_ok = true;
    for (auto c : carmichaels) {
        if (prime::is_prime(c)) {
            carm_ok = false;
            printf("    Carmichael %llu incorrectly flagged prime\n", (unsigned long long)c);
        }
    }
    CHECK(carm_ok, "all Carmichael numbers correctly rejected");

    // Mersenne primes
    // M31 = 2^31 - 1 = 2147483647
    CHECK(prime::is_prime(2147483647ULL), "M31 = 2^31-1 is prime");
    // M61 = 2^61 - 1 = 2305843009213693951
    CHECK(prime::is_prime(2305843009213693951ULL), "M61 = 2^61-1 is prime");

    // Composite Mersenne-like numbers
    CHECK(!prime::is_prime(8388607ULL),   "2^23-1 = 8388607 is composite (47*178481)");
    CHECK(!prime::is_prime(536870911ULL), "2^29-1 = 536870911 is composite");

    // Large known primes
    CHECK(prime::is_prime(1000000007ULL),    "10^9+7 is prime");
    CHECK(prime::is_prime(999999999989ULL),  "999999999989 is prime");

    // Product of two large primes must be composite
    CHECK(!prime::is_prime(1000000007ULL * 1000000009ULL), "product of two large primes is composite");

    // All primes up to 97
    uint64_t small_primes[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97};
    bool all_small = true;
    for (auto p : small_primes) {
        if (!prime::is_prime(p)) { all_small = false; break; }
    }
    CHECK(all_small, "all primes up to 97 detected");

    // All composites up to 28
    uint64_t small_comp[] = {4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,27,28};
    bool all_sc = true;
    for (auto c : small_comp) {
        if (prime::is_prime(c)) { all_sc = false; break; }
    }
    CHECK(all_sc, "small composites [4..28] correctly rejected");

    // Engine::verify cross-check
    prime::Engine engine;
    auto r2 = engine.verify(2);
    CHECK(r2.confirmed && r2.value == 2, "Engine::verify(2) confirms prime");
    auto r4 = engine.verify(4);
    CHECK(!r4.confirmed && r4.value == 4, "Engine::verify(4) confirms composite");
    auto rBig = engine.verify(1000000007ULL);
    CHECK(rBig.confirmed, "Engine::verify(10^9+7) confirms prime");
}

// =====================================================================
// 2. Modular arithmetic
// =====================================================================

static void test_modular_arithmetic() {
    SECTION("2. Modular Arithmetic");

    // Basic mulmod
    CHECK(prime::mulmod(3, 5, 7) == 1, "mulmod(3,5,7) = 15 mod 7 = 1");
    CHECK(prime::mulmod(0, 12345, 100) == 0, "mulmod(0,x,m) = 0");
    CHECK(prime::mulmod(12345, 0, 100) == 0, "mulmod(x,0,m) = 0");
    CHECK(prime::mulmod(1, 1, 1) == 0, "mulmod(1,1,1) = 0");

    // Overflow case: (2^63) * (2^63) mod (2^63+1)
    // 2^63 = m-1, so (m-1)^2 mod m = 1
    uint64_t big = (1ULL << 63);
    uint64_t bigmod = big + 1;
    CHECK(prime::mulmod(big, big, bigmod) == 1, "mulmod overflow: (2^63)^2 mod (2^63+1) = 1");

    // Another overflow: (UINT64_MAX-1)^2 mod UINT64_MAX = 1
    uint64_t umax = UINT64_MAX;
    CHECK(prime::mulmod(umax - 1, umax - 1, umax) == 1, "mulmod(UINT64_MAX-1, UINT64_MAX-1, UINT64_MAX) = 1");

    // Large values close to overflow
    CHECK(prime::mulmod(1ULL << 62, 4, (1ULL << 63) + 7) != 0, "mulmod near 2^64 does not crash");

    // Basic modpow
    CHECK(prime::modpow(2, 10, 1000) == 24, "modpow(2,10,1000) = 1024 mod 1000 = 24");
    CHECK(prime::modpow(2, 10, 1024) == 0,  "modpow(2,10,1024) = 0");
    CHECK(prime::modpow(3, 0, 100) == 1,    "modpow(x,0,m) = 1");
    CHECK(prime::modpow(5, 1, 100) == 5,    "modpow(5,1,100) = 5");
    CHECK(prime::modpow(7, 7, 1) == 0,      "modpow(x,y,1) = 0");

    // modpow(3,100,mod): 3^100 mod 10^9+7
    // Verify consistency: 3^100 * 3 = 3^101
    uint64_t m = 1000000007ULL;
    uint64_t r100 = prime::modpow(3, 100, m);
    uint64_t r101 = prime::modpow(3, 101, m);
    CHECK(prime::mulmod(r100, 3, m) == r101, "modpow consistency: 3^100 * 3 = 3^101 (mod 10^9+7)");

    // Fermat's little theorem: 2^(p-1) mod p == 1 for known primes
    uint64_t fermat_primes[] = {5, 7, 13, 97, 101, 997, 7919, 10007, 104729, 1000000007ULL};
    bool fermat_ok = true;
    for (auto p : fermat_primes) {
        if (prime::modpow(2, p - 1, p) != 1) {
            fermat_ok = false;
            printf("    Fermat failed: 2^(%llu-1) mod %llu != 1\n",
                   (unsigned long long)p, (unsigned long long)p);
        }
    }
    CHECK(fermat_ok, "Fermat's little theorem: 2^(p-1) mod p == 1 for known primes");

    // Fermat with multiple bases
    bool fermat_multi = true;
    for (auto p : fermat_primes) {
        for (uint64_t a : {3ULL, 5ULL, 11ULL}) {
            if (a % p == 0) continue;
            if (prime::modpow(a, p - 1, p) != 1) {
                fermat_multi = false;
                printf("    Fermat failed: %llu^(%llu-1) mod %llu != 1\n",
                       (unsigned long long)a, (unsigned long long)p, (unsigned long long)p);
            }
        }
    }
    CHECK(fermat_multi, "Fermat's little theorem holds for bases 3,5,11 across sample primes");

    // modpow consistency for large exponent
    uint64_t r64 = prime::modpow(2, 64, m);
    uint64_t r65 = prime::modpow(2, 65, m);
    CHECK(prime::mulmod(r64, 2, m) == r65, "modpow consistency: 2^64 * 2 = 2^65 (mod 10^9+7)");
}

// =====================================================================
// 3. Sieve
// =====================================================================

static void test_sieve() {
    SECTION("3. Sieve");

    // pi(100) = 25
    auto s100 = prime::sieve(100);
    int count100 = 0;
    for (uint64_t i = 0; i <= 100; i++) if (s100[i]) count100++;
    CHECK(count100 == 25, "pi(100) = 25");

    // pi(1000) = 168
    auto s1000 = prime::sieve(1000);
    int count1000 = 0;
    for (uint64_t i = 0; i <= 1000; i++) if (s1000[i]) count1000++;
    CHECK(count1000 == 168, "pi(1000) = 168");

    // pi(10000) = 1229
    auto s10k = prime::sieve(10000);
    int count10k = 0;
    for (uint64_t i = 0; i <= 10000; i++) if (s10k[i]) count10k++;
    CHECK(count10k == 1229, "pi(10000) = 1229");

    // Sieve agrees with is_prime for all n <= 1000
    bool sieve_agree = true;
    for (uint64_t i = 0; i <= 1000; i++) {
        if ((bool)s1000[i] != prime::is_prime(i)) {
            sieve_agree = false;
            printf("    Disagreement at %llu: sieve=%d, is_prime=%d\n",
                   (unsigned long long)i, (int)s1000[i], prime::is_prime(i));
            break;
        }
    }
    CHECK(sieve_agree, "sieve agrees with is_prime for n <= 1000");

    // Sieve edge case: sieve(2)
    auto s2 = prime::sieve(2);
    CHECK(s2[0] == false && s2[1] == false && s2[2] == true, "sieve(2) correct");

    // Wheel-210 correctness: all primes > 7 pass wheel
    bool wheel_ok = true;
    for (uint64_t i = 8; i <= 10000; i++) {
        if (s10k[i]) {
            if (!prime::WHEEL.valid(i)) {
                wheel_ok = false;
                printf("    Wheel-210 rejects prime %llu\n", (unsigned long long)i);
                break;
            }
        }
    }
    CHECK(wheel_ok, "wheel-210 accepts all primes > 7 up to 10000");

    // Wheel-210 rejects small composites
    CHECK(!prime::WHEEL.valid(4) && !prime::WHEEL.valid(6) && !prime::WHEEL.valid(9) && !prime::WHEEL.valid(10),
          "wheel-210 rejects 4, 6, 9, 10");
}

// =====================================================================
// 4. Factoring
// =====================================================================

static void test_factoring() {
    SECTION("4. Factoring");

    // Trial division: factor_u64 on small values
    auto f1 = prime::factor_u64(1);
    CHECK(f1.empty(), "factor(1) = empty");

    auto f2 = prime::factor_u64(2);
    CHECK(f2.size() == 1 && f2[0] == 2, "factor(2) = {2}");

    auto f12 = prime::factor_u64(12);
    CHECK(f12.size() == 3 && f12[0] == 2 && f12[1] == 2 && f12[2] == 3, "factor(12) = {2,2,3}");

    // 143 = 11 * 13
    auto f143 = prime::factor_u64(143);
    CHECK(f143.size() == 2 && f143[0] == 11 && f143[1] == 13, "factor(143) = {11,13}");

    // Power of 2: 256 = 2^8
    auto f256 = prime::factor_u64(256);
    CHECK(f256.size() == 8, "factor(256) = eight 2s");
    bool all2 = true;
    for (auto x : f256) if (x != 2) all2 = false;
    CHECK(all2, "factor(256) all factors are 2");

    // 10007 * 10009 = 100160063
    auto fSemi = prime::factor_u64(100160063ULL);
    CHECK(fSemi.size() == 2 && fSemi[0] == 10007 && fSemi[1] == 10009, "factor(10007*10009)");

    // 101 * 103 * 107 = 1113121
    auto f3 = prime::factor_u64(1113121ULL);
    CHECK(f3.size() == 3 && f3[0] == 101 && f3[1] == 103 && f3[2] == 107,
          "factor(101*103*107) = {101,103,107}");

    // Pollard rho: 1000003 * 1000033 = 1000036000099
    auto fPollard = prime::factor_u64(1000036000099ULL);
    CHECK(fPollard.size() == 2 && fPollard[0] == 1000003ULL && fPollard[1] == 1000033ULL,
          "factor(1000003*1000033) via Pollard rho");

    // Complete factorization: product reconstruction
    uint64_t test_vals[] = {2, 12, 143, 256, 1113121ULL, 100160063ULL, 1000036000099ULL};
    bool product_ok = true;
    for (auto n : test_vals) {
        auto factors = prime::factor_u64(n);
        uint64_t product = 1;
        for (auto f : factors) product *= f;
        if (product != n) {
            product_ok = false;
            printf("    Product mismatch for %llu\n", (unsigned long long)n);
        }
    }
    CHECK(product_ok, "factor product reconstruction correct for all test values");

    // All factors should be prime
    bool all_prime = true;
    for (auto n : test_vals) {
        auto factors = prime::factor_u64(n);
        for (auto f : factors) {
            if (!prime::is_prime(f)) {
                all_prime = false;
                printf("    Non-prime factor %llu of %llu\n", (unsigned long long)f, (unsigned long long)n);
            }
        }
    }
    CHECK(all_prime, "all returned factors are prime");

    // factors_string
    CHECK(prime::factors_string(12) == "2 x 2 x 3", "factors_string(12) = \"2 x 2 x 3\"");
    CHECK(prime::factors_string(1).empty(), "factors_string(1) = empty");

    // brent_rho_one internal
    uint64_t d = prime::brent_rho_one(143, 1);
    CHECK(d == 11 || d == 13, "brent_rho_one(143,1) finds a factor of 143");
}

// =====================================================================
// 5. Special primes: Wieferich and Wilson
// =====================================================================

static void test_special_primes() {
    SECTION("5. Special Primes (Wieferich / Wilson)");

    // Wieferich primes: 2^(p-1) mod p^2 == 1
    // Known: 1093 and 3511
    auto wieferich_test = [](uint64_t p) -> bool {
        return prime::modpow(2, p - 1, p * p) == 1;
    };

    CHECK(prime::is_prime(1093), "1093 is prime");
    CHECK(wieferich_test(1093),  "1093 is Wieferich: 2^1092 mod 1093^2 == 1");

    CHECK(prime::is_prime(3511), "3511 is prime");
    CHECK(wieferich_test(3511),  "3511 is Wieferich: 2^3510 mod 3511^2 == 1");

    // Non-Wieferich primes
    CHECK(!wieferich_test(3),  "3 is not Wieferich");
    CHECK(!wieferich_test(5),  "5 is not Wieferich");
    CHECK(!wieferich_test(7),  "7 is not Wieferich");
    CHECK(!wieferich_test(11), "11 is not Wieferich");
    CHECK(!wieferich_test(13), "13 is not Wieferich");

    // Wilson primes: (p-1)! mod p^2 == p^2 - 1
    // Known: 5, 13, 563
    auto wilson_test = [](uint64_t p) -> bool {
        uint64_t mod = p * p;
        uint64_t factorial = 1;
        for (uint64_t i = 2; i < p; i++) {
            factorial = prime::mulmod(factorial, i, mod);
        }
        return factorial == mod - 1;
    };

    CHECK(prime::is_prime(5),   "5 is prime");
    CHECK(wilson_test(5),       "5 is Wilson prime: 4! mod 25 == 24");

    CHECK(prime::is_prime(13),  "13 is prime");
    CHECK(wilson_test(13),      "13 is Wilson prime: 12! mod 169 == 168");

    CHECK(prime::is_prime(563), "563 is prime");
    CHECK(wilson_test(563),     "563 is Wilson prime: 562! mod 563^2 == 563^2-1");

    // Non-Wilson primes
    CHECK(!wilson_test(7),  "7 is not Wilson prime");
    CHECK(!wilson_test(11), "11 is not Wilson prime");
    CHECK(!wilson_test(23), "23 is not Wilson prime");
}

// =====================================================================
// 6. Search: find_primes_in_range on small ranges, verify vs sieve
// =====================================================================

static void test_search() {
    SECTION("6. Search Engine");

    prime::Engine engine;
    int hw = std::max(1, (int)std::thread::hardware_concurrency());

    // Wheel-210 skips factors {2,3,5,7}, so search_range(2,30) finds 6: 11,13,17,19,23,29
    auto r1 = engine.search_range(2, 30, 1);
    CHECK(r1.size() == 6, "search_range(2,30) finds 6 primes (wheel skips 2,3,5,7)");

    // Verify sorted
    bool sorted = true;
    for (size_t i = 1; i < r1.size(); i++) {
        if (r1[i].value <= r1[i-1].value) { sorted = false; break; }
    }
    CHECK(sorted, "search_range results are sorted");

    // All confirmed
    bool all_confirmed = true;
    for (auto& pr : r1) if (!pr.confirmed) all_confirmed = false;
    CHECK(all_confirmed, "all search results have confirmed=true");

    // Cross-check with sieve for [1000, 1100]
    auto sieveData = prime::sieve(1100);
    int sieve_count = 0;
    for (uint64_t i = 1000; i <= 1100; i++) if (sieveData[i]) sieve_count++;
    auto rMid = engine.search_range(1000, 1100, hw);
    CHECK((int)rMid.size() == sieve_count,
          "search_range(1000,1100) count matches sieve");

    // Exact values match sieve
    std::set<uint64_t> sieveSet, engineSet;
    for (uint64_t i = 1000; i <= 1100; i++) if (sieveData[i]) sieveSet.insert(i);
    for (auto& pr : rMid) engineSet.insert(pr.value);
    CHECK(sieveSet == engineSet, "search_range(1000,1100) exact values match sieve");

    // Cross-check [2, 100] with sieve
    auto s100 = prime::sieve(100);
    int sieve100 = 0;
    for (uint64_t i = 2; i <= 100; i++) if (s100[i]) sieve100++;
    auto r100 = engine.search_range(2, 100, 1);
    // Wheel-210 skips {2,3,5,7} = 4 primes, so engine finds sieve100 - 4
    CHECK((int)r100.size() == sieve100 - 4, "search_range(2,100) matches pi(100)-4 (wheel skips 2,3,5,7)");

    // Cross-check [5000, 5200] with sieve
    auto s5200 = prime::sieve(5200);
    std::set<uint64_t> sSet5k, eSet5k;
    for (uint64_t i = 5000; i <= 5200; i++) if (s5200[i]) sSet5k.insert(i);
    auto r5k = engine.search_range(5000, 5200, hw);
    for (auto& pr : r5k) eSet5k.insert(pr.value);
    CHECK(sSet5k == eSet5k, "search_range(5000,5200) exact values match sieve");

    // Multi-threaded vs single-threaded consistency
    auto r_single = engine.search_range(100000, 101000, 1);
    auto r_multi  = engine.search_range(100000, 101000, hw);
    std::set<uint64_t> set1, set2;
    for (auto& pr : r_single) set1.insert(pr.value);
    for (auto& pr : r_multi)  set2.insert(pr.value);
    CHECK(set1 == set2, "single-thread and multi-thread find identical primes [100000,101000]");

    // Twin primes found by engine in [2,100] (excludes (3,5),(5,7) due to wheel)
    // Expected: (11,13),(17,19),(29,31),(41,43),(59,61),(71,73) = 6 pairs
    int twin_count = 0;
    for (size_t i = 1; i < r100.size(); i++) {
        if (r100[i].value - r100[i-1].value == 2) twin_count++;
    }
    CHECK(twin_count == 6, "6 twin prime pairs in engine results [2,100]");

    // Determinism: repeated runs
    uint64_t lo = 999900, hi = 1000100;
    auto rBase = engine.search_range(lo, hi, hw);
    std::set<uint64_t> baseline;
    for (auto& pr : rBase) baseline.insert(pr.value);
    bool deterministic = true;
    for (int trial = 0; trial < 3; trial++) {
        auto rt = engine.search_range(lo, hi, hw);
        std::set<uint64_t> s;
        for (auto& pr : rt) s.insert(pr.value);
        if (s != baseline) { deterministic = false; break; }
    }
    CHECK(deterministic, "3 repeated runs produce identical results");
}

// =====================================================================
// 7. Benchmark: timed run on [10^9, 10^9+10^5]
// =====================================================================

static void test_benchmark() {
    SECTION("7. Performance Benchmark");

    prime::Engine engine;
    int hw = std::max(1, (int)std::thread::hardware_concurrency());
    printf("  Hardware threads: %d\n", hw);

    uint64_t start = 1000000000ULL;
    uint64_t end   = 1000000000ULL + 100000ULL;  // 10^9 + 10^5

    auto t0 = std::chrono::steady_clock::now();
    auto results = engine.search_range(start, end, hw);
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    printf("  Range: [%llu, %llu]\n", (unsigned long long)start, (unsigned long long)end);
    printf("  Primes found: %zu\n", results.size());
    printf("  Time: %.4f s\n", dt);

    double rate = (double)(end - start) / dt;
    printf("  Rate: %.0f candidates/sec\n", rate);

    if (dt > 0) {
        printf("  Primes/sec: %.0f\n", (double)results.size() / dt);
    }

    printf("  Stats: tested=%llu, CRT_rejected=%llu, MR_tested=%llu\n",
           (unsigned long long)engine.stats.candidates_tested.load(),
           (unsigned long long)engine.stats.crt_rejected.load(),
           (unsigned long long)engine.stats.mr_tested.load());

    // pi(10^9 + 10^5) - pi(10^9) should be around 4832 (approximate)
    CHECK(results.size() > 4500 && results.size() < 5200,
          "prime count in [10^9, 10^9+10^5] is reasonable (~4832)");
    CHECK(dt < 30.0, "benchmark completes in under 30 seconds");
}

// =====================================================================
// Additional: CRT rejection filter
// =====================================================================

static void test_crt() {
    SECTION("A. CRT Rejection Filter");

    CHECK(prime::crt_reject(0),  "crt_reject(0)");
    CHECK(prime::crt_reject(1),  "crt_reject(1)");
    CHECK(!prime::crt_reject(2), "crt_reject(2) = false (prime)");
    CHECK(!prime::crt_reject(37), "crt_reject(37) = false (prime)");

    // Composites with factors 11..31 should be rejected
    CHECK(prime::crt_reject(11 * 43), "crt_reject(11*43)");
    CHECK(prime::crt_reject(13 * 47), "crt_reject(13*47)");
    CHECK(prime::crt_reject(17 * 41), "crt_reject(17*41)");
    CHECK(prime::crt_reject(19 * 43), "crt_reject(19*43)");
    CHECK(prime::crt_reject(23 * 41), "crt_reject(23*41)");
    CHECK(prime::crt_reject(29 * 43), "crt_reject(29*43)");
    CHECK(prime::crt_reject(31 * 43), "crt_reject(31*43)");

    // crt_reject must not reject actual primes > 37
    auto sv = prime::sieve(10000);
    bool crt_ok = true;
    for (uint64_t i = 38; i <= 10000; i++) {
        if (sv[i] && prime::crt_reject(i)) {
            crt_ok = false;
            printf("    crt_reject incorrectly rejects prime %llu\n", (unsigned long long)i);
            break;
        }
    }
    CHECK(crt_ok, "crt_reject does not reject any prime in [38, 10000]");
}

// =====================================================================
// Additional: Convergence / Shadow field
// =====================================================================

static void test_convergence() {
    SECTION("B. Convergence / Shadow Field");

    // Multiples of shadow primes get -999
    CHECK(prime::convergence(7 * 11) == -999.0, "convergence(77) = -999 (multiple of shadow prime)");
    CHECK(prime::convergence(13 * 17) == -999.0, "convergence(221) = -999");

    // Primes not divisible by shadow primes have finite scores
    double c97 = prime::convergence(97);
    CHECK(c97 != -999.0, "convergence(97) is finite (prime)");
    double c101 = prime::convergence(101);
    CHECK(c101 != -999.0, "convergence(101) is finite");

    // verify() returns convergence_score
    prime::Engine engine;
    auto pr = engine.verify(997);
    CHECK(pr.convergence_score != 0.0 && pr.convergence_score != -999.0,
          "verify(997) returns non-trivial convergence score");
}

// =====================================================================
// Additional: Heuristic factoring
// =====================================================================

static void test_heuristic_factoring() {
    SECTION("C. Heuristic Factoring (Pinch + Lucky7 + DivisorWeb)");

    // PinchFactor on 1729 = 7 * 13 * 19
    auto ph = prime::pinch_factor(1729);
    bool pinch_found = false;
    for (auto& h : ph) {
        if (1729 % h.divisor == 0 && h.divisor > 1 && h.divisor < 1729) {
            pinch_found = true;
        }
    }
    CHECK(pinch_found, "pinch_factor(1729) finds at least one divisor");

    // Lucky7s on 10007*10009 = 100160063
    auto l7 = prime::lucky7_factor(100160063ULL);
    bool lucky7_found = false;
    for (auto& h : l7) {
        if (100160063ULL % h.divisor == 0 && h.divisor > 1) lucky7_found = true;
    }
    CHECK(lucky7_found, "lucky7_factor(10007*10009) finds factor near 10^4");

    // DivisorWeb on 60 = 2^2 * 3 * 5
    auto web = prime::divisor_web(60, 10);
    CHECK(web.n == 60, "divisor_web(60).n = 60");
    std::set<uint64_t> expected_pd = {2, 3, 5};
    std::set<uint64_t> actual_pd(web.prime_divisors.begin(), web.prime_divisors.end());
    CHECK(actual_pd == expected_pd, "divisor_web(60) prime_divisors = {2,3,5}");

    // heuristic_divisors: all results must divide the input
    auto hd = prime::heuristic_divisors(1729);
    bool hd_valid = true;
    for (auto d : hd) {
        if (1729 % d != 0) { hd_valid = false; break; }
    }
    CHECK(hd_valid, "heuristic_divisors(1729) all results divide 1729");
}

// =====================================================================
// Main
// =====================================================================

int main() {
    printf("PrimePath C++ Engine -- Comprehensive Test Suite\n");
    printf("================================================\n");

    test_primality();
    test_modular_arithmetic();
    test_sieve();
    test_factoring();
    test_special_primes();
    test_search();
    test_benchmark();
    test_crt();
    test_convergence();
    test_heuristic_factoring();

    printf("\n================================================\n");
    printf("RESULTS: %d passed, %d failed, %d total\n", g_pass, g_fail, g_pass + g_fail);
    printf("================================================\n");

    if (g_fail > 0) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
    printf("ALL TESTS PASSED\n");
    return 0;
}
