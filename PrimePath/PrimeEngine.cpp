#include "PrimeEngine.hpp"
#include <algorithm>
#include <cmath>

namespace prime {

PrimeResult Engine::verify(uint64_t n) {
    PrimeResult r;
    r.value = n;
    r.convergence_score = convergence(n, 10);
    r.confirmed = is_prime(n);
    return r;
}

void Engine::search_chunk(uint64_t start, uint64_t end,
                          PrimeCallback prime_cb, int thread_id) {
    // Align start to wheel-210 boundary
    uint64_t ring = start / 210;
    if (ring * 210 < start && ring < UINT64_MAX / 210) ring++;

    // Scan using wheel-210 spokes
    for (uint64_t r = (start > 210) ? start / 210 : 0; ; r++) {
        for (int si = 0; si < WHEEL210_COUNT; si++) {
            if (!stats.running.load(std::memory_order_relaxed)) return;

            uint64_t n = r * 210 + SPOKES_210[si];
            if (n < start) continue;
            if (n > end) return;
            if (n < 2) continue;

            stats.candidates_tested.fetch_add(1, std::memory_order_relaxed);

            // Stage 1: CRT rejection (primes 11–31)
            if (crt_reject(n)) {
                stats.crt_rejected.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Stage 2: Miller-Rabin
            stats.mr_tested.fetch_add(1, std::memory_order_relaxed);
            if (is_prime(n)) {
                stats.primes_found.fetch_add(1, std::memory_order_relaxed);
                PrimeResult pr;
                pr.value = n;
                pr.convergence_score = convergence(n, 10);
                pr.confirmed = true;
                if (prime_cb) {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    prime_cb(pr);
                }
            }
        }
    }
}

void Engine::search(uint64_t start, uint64_t end, int n_threads,
                    PrimeCallback prime_cb, ProgressCallback progress_cb) {
    stats.reset();
    stats.range_start = start;
    stats.range_end = end;
    stats.running = true;
    stats.start_time = std::chrono::steady_clock::now();

    if (n_threads < 1) n_threads = 1;
    if (n_threads > 256) n_threads = 256;

    // Divide range into chunks
    uint64_t range = end - start;
    uint64_t chunk = range / n_threads;
    if (chunk == 0) chunk = 1;

    workers.clear();
    for (int t = 0; t < n_threads; t++) {
        uint64_t c_start = start + t * chunk;
        uint64_t c_end = (t == n_threads - 1) ? end : c_start + chunk - 1;
        if (c_start > end) break;
        workers.emplace_back(&Engine::search_chunk, this, c_start, c_end, prime_cb, t);
    }

    // Progress reporting thread
    if (progress_cb) {
        std::thread progress_thread([this, progress_cb]() {
            while (stats.running.load()) {
                progress_cb(stats);
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
            progress_cb(stats); // final update
        });
        progress_thread.detach();
    }

    // Wait for all workers
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
    stats.running = false;
    workers.clear();
}

void Engine::stop() {
    stats.running = false;
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
    workers.clear();
}

std::vector<PrimeResult> Engine::search_range(uint64_t start, uint64_t end, int n_threads) {
    std::vector<PrimeResult> results;
    search(start, end, n_threads,
        [&results](const PrimeResult& pr) {
            results.push_back(pr);
        },
        nullptr
    );
    // Sort by value
    std::sort(results.begin(), results.end(),
        [](const PrimeResult& a, const PrimeResult& b) { return a.value < b.value; });
    return results;
}

} // namespace prime
