#include "PrimePath/PrimeEngine.hpp"
#include <cstdio>
#include <chrono>

int main() {
    prime::Engine engine;

    printf("PrimePath C++ Engine — Verification Test\n");
    printf("=========================================\n\n");

    // Test 1: Known primes
    printf("Test 1: Known primes\n");
    uint64_t test_primes[] = {2,3,5,7,11,13,97,101,127,997,1009,10007,100003,1000000007};
    for (auto p : test_primes) {
        auto r = engine.verify(p);
        printf("  %llu → %s\n", p, r.confirmed ? "PRIME ✓" : "FAIL ✗");
        if (!r.confirmed) { printf("ERROR: %llu should be prime!\n", p); return 1; }
    }
    printf("\n");

    // Test 2: Known composites
    printf("Test 2: Known composites (Carmichael numbers)\n");
    uint64_t carmichaels[] = {561,1105,1729,2465,2821,6601,8911};
    for (auto c : carmichaels) {
        auto r = engine.verify(c);
        printf("  %llu → %s\n", c, r.confirmed ? "PRIME?!" : "COMPOSITE ✓");
        if (r.confirmed) { printf("ERROR: %llu should be composite!\n", c); return 1; }
    }
    printf("\n");

    // Test 3: Multi-threaded search
    printf("Test 3: Multi-threaded search [1000000, 1001000]\n");
    int hw_threads = std::thread::hardware_concurrency();
    printf("  Using %d threads\n", hw_threads);

    auto t0 = std::chrono::steady_clock::now();
    auto results = engine.search_range(1000000, 1001000, hw_threads);
    auto dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    printf("  Found %zu primes in %.2fms\n", results.size(), dt);
    printf("  First 10: ");
    for (int i = 0; i < 10 && i < (int)results.size(); i++) {
        printf("%llu ", results[i].value);
    }
    printf("\n");

    // Verify against sieve
    auto sieve = prime::sieve(1001000);
    int sieve_count = 0;
    for (uint64_t n = 1000000; n <= 1001000; n++) {
        if (sieve[n]) sieve_count++;
    }
    printf("  Sieve count: %d, Engine count: %zu → %s\n\n",
        sieve_count, results.size(),
        sieve_count == (int)results.size() ? "MATCH ✓" : "MISMATCH ✗");
    if (sieve_count != (int)results.size()) return 1;

    // Test 4: Speed benchmark
    printf("Test 4: Speed benchmark — search [10^9, 10^9 + 10^6]\n");
    t0 = std::chrono::steady_clock::now();
    auto big_results = engine.search_range(1000000000ULL, 1000000000ULL + 1000000, hw_threads);
    dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    printf("  Found %zu primes in %.3fs\n", big_results.size(), dt);
    printf("  Rate: %.0f candidates/sec\n", engine.stats.primes_per_second());
    printf("  Stats: tested=%llu, CRT=%llu, MR=%llu\n",
        engine.stats.candidates_tested.load(),
        engine.stats.crt_rejected.load(),
        engine.stats.mr_tested.load());

    printf("\nAll tests passed!\n");
    return 0;
}
