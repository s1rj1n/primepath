#pragma once
#include "TaskManager.hpp"
#include "EvenShadow.hpp"
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <chrono>

namespace prime {

struct BenchmarkResult {
    std::string component;
    std::string scale;
    uint64_t items_processed;
    double elapsed_sec;
    double throughput;      // items/sec
    std::string unit;
    std::string notes;
};

class Benchmark {
public:
    explicit Benchmark(TaskManager* mgr);

    // Run all benchmarks. log_cb receives formatted lines for display.
    // Checks should_run between tests so user can cancel.
    void run_all(std::function<void(const std::string&)> log_cb,
                 std::atomic<bool>& should_run);

private:
    TaskManager* _mgr;

    // Individual benchmarks
    void bench_segmented_sieve(std::function<void(const std::string&)>& log,
                               std::atomic<bool>& run);
    void bench_cpu_primality(std::function<void(const std::string&)>& log,
                             std::atomic<bool>& run);
    void bench_gpu_batches(std::function<void(const std::string&)>& log,
                           std::atomic<bool>& run);
    void bench_even_shadow(std::function<void(const std::string&)>& log,
                           std::atomic<bool>& run);
    void bench_pair_scan(std::function<void(const std::string&)>& log,
                         std::atomic<bool>& run);
    void bench_sieve_pipeline(std::function<void(const std::string&)>& log,
                              std::atomic<bool>& run);

    // Helpers
    static std::string fmt_num(uint64_t n);
    static std::string fmt_rate(double rate);
    static std::string fmt_time(double sec);
};

} // namespace prime
