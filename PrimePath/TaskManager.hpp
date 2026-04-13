#pragma once
#include "PrimeEngine.hpp"
#include "GPUBackend.hpp"
#include "ThreadPool.hpp"
#include "MatrixSieve.hpp"
#include "PseudoprimePredictor.hpp"
#include "EvenShadow.hpp"
#include "LoadBalancer.hpp"
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <functional>
#include <memory>
#include <set>
#include <deque>
#include <future>

namespace prime {

// ═══════════════════════════════════════════════════════════════════════
// Search task types — the high-value targets
// ═══════════════════════════════════════════════════════════════════════

enum class TaskType {
    Wieferich,      // 2^(p-1) ≡ 1 (mod p²) — only 2 known
    WallSunSun,     // p² | F(p-(p/5))       — NONE known
    Wilson,         // (p-1)! ≡ -1 (mod p²)  — only 3 known
    TwinPrime,      // p and p+2 both prime
    SophieGermain,  // p and 2p+1 both prime
    CousinPrime,    // p and p+4 both prime
    SexyPrime,      // p and p+6 both prime
    GeneralPrime,   // find all primes in range
    Emirp,          // p and reverse(p) both prime, p ≠ reverse(p)
    MersenneTrial,  // trial factor 2^p-1 by candidates 2kp+1
    FermatFactor,   // find factors of Fermat numbers F_m = 2^(2^m)+1
};

inline const char* task_name(TaskType t) {
    switch (t) {
        case TaskType::Wieferich:     return "Wieferich";
        case TaskType::WallSunSun:    return "Wall-Sun-Sun";
        case TaskType::Wilson:        return "Wilson";
        case TaskType::TwinPrime:     return "Twin";
        case TaskType::SophieGermain: return "Sophie Germain";
        case TaskType::CousinPrime:   return "Cousin";
        case TaskType::SexyPrime:     return "Sexy";
        case TaskType::GeneralPrime:  return "General";
        case TaskType::Emirp:         return "Emirp";
        case TaskType::MersenneTrial: return "Mersenne TF";
        case TaskType::FermatFactor:  return "Fermat Factor";
    }
    return "Unknown";
}

inline const char* task_key(TaskType t) {
    switch (t) {
        case TaskType::Wieferich:     return "wieferich";
        case TaskType::WallSunSun:    return "wallsunsun";
        case TaskType::Wilson:        return "wilson";
        case TaskType::TwinPrime:     return "twin";
        case TaskType::SophieGermain: return "sophie";
        case TaskType::CousinPrime:   return "cousin";
        case TaskType::SexyPrime:     return "sexy";
        case TaskType::GeneralPrime:  return "general";
        case TaskType::Emirp:         return "emirp";
        case TaskType::MersenneTrial: return "mersenne_tf";
        case TaskType::FermatFactor:  return "fermat_factor";
    }
    return "unknown";
}

inline TaskType task_from_key(const std::string& key) {
    if (key == "wieferich")  return TaskType::Wieferich;
    if (key == "wallsunsun") return TaskType::WallSunSun;
    if (key == "wilson")     return TaskType::Wilson;
    if (key == "twin")       return TaskType::TwinPrime;
    if (key == "sophie")     return TaskType::SophieGermain;
    if (key == "cousin")     return TaskType::CousinPrime;
    if (key == "sexy")       return TaskType::SexyPrime;
    if (key == "emirp")         return TaskType::Emirp;
    if (key == "mersenne_tf")   return TaskType::MersenneTrial;
    if (key == "fermat_factor") return TaskType::FermatFactor;
    return TaskType::GeneralPrime;
}

enum class TaskStatus { Idle, Running, Paused };

// ── Single search task ──────────────────────────────────────────────

struct SearchTask {
    TaskType type;
    TaskStatus status = TaskStatus::Idle;
    uint64_t start_pos;         // where this search begins
    uint64_t current_pos;       // where we are now
    uint64_t end_pos;           // 0 = unlimited
    uint64_t found_count = 0;
    uint64_t tested_count = 0;
    std::atomic<bool> should_run{false};
    std::thread worker;
    double rate = 0;            // candidates/sec

    // Mersenne TF: known factors from worktodo.txt (already-discovered primes
    // to skip during trial factoring). See James's known-factor spec.
    std::vector<std::string> known_factors;

    // Mersenne TF: bit range for the current assignment
    double bit_lo = 0;
    double bit_hi = 0;

    SearchTask() = default;
    SearchTask(TaskType t, uint64_t start, uint64_t end = 0)
        : type(t), start_pos(start), current_pos(start), end_pos(end) {}

    // Non-copyable due to atomic + thread
    SearchTask(SearchTask&& o) noexcept
        : type(o.type), status(o.status), start_pos(o.start_pos),
          current_pos(o.current_pos), end_pos(o.end_pos),
          found_count(o.found_count), tested_count(o.tested_count),
          rate(o.rate), known_factors(std::move(o.known_factors)),
          bit_lo(o.bit_lo), bit_hi(o.bit_hi) {
        should_run.store(o.should_run.load());
    }
};

// ── Discovery record ────────────────────────────────────────────────

enum class PrimeClass { Prime, Pseudoprime, Composite };

struct Discovery {
    TaskType type;
    uint64_t value;
    uint64_t value2;        // for pairs (twin, cousin, sexy, sophie)
    PrimeClass pclass;      // classification
    uint64_t tested_at;     // position when found (for audit trail)
    std::string divisors;   // for composites: "p1 x p2 x ..." or empty for primes
    std::string timestamp;
};

// ── Callbacks ───────────────────────────────────────────────────────

using LogCallback = std::function<void(const std::string&)>;
using DiscoveryCallback = std::function<void(const Discovery&)>;

// ═══════════════════════════════════════════════════════════════════════
// SievePipeline — deep-prefetch prime sieve shared by all workers
//
// Uses the thread pool to sieve N segments ahead of the current position.
// While the GPU crunches the current batch, all CPU cores stay busy
// sieving future segments. Workers just call next_segment() to get
// a ready-to-process vector of primes.
// ═══════════════════════════════════════════════════════════════════════

class TaskManager; // forward

class SievePipeline {
public:
    // depth: how many segments to keep pre-sieved
    SievePipeline(TaskManager* mgr, uint64_t start, uint64_t seg_size, int depth = 8);

    // Get the next pre-sieved segment (blocks until ready). Returns primes in segment.
    std::vector<uint64_t> next_segment();

    // Current segment start position
    uint64_t current_start() const { return _current_start; }

    // Drain remaining futures on shutdown
    void drain();

private:
    struct PrefetchEntry {
        uint64_t seg_start;
        std::future<std::vector<uint64_t>> future;
    };

    TaskManager* _mgr;
    uint64_t _current_start;
    uint64_t _seg_size;
    int _depth;
    uint64_t _next_enqueue;  // next segment start to enqueue
    std::deque<PrefetchEntry> _queue;
    void enqueue_one();
};

// ═══════════════════════════════════════════════════════════════════════
// TaskManager — orchestrates parallel searches with persistence
// ═══════════════════════════════════════════════════════════════════════

class TaskManager {
    friend class SievePipeline;  // pipeline needs access to _pool and segmented_sieve
    friend class Benchmark;      // benchmark needs ensure_small_primes
public:
    // data_dir: folder for search_progress.txt and discoveries.txt
    TaskManager(const std::string& data_dir);
    ~TaskManager();

    // Initialize default tasks with frontier positions
    void init_defaults();

    // Persistence
    void load_state();
    void save_state();
    void save_discovery(const Discovery& d);
    void flush_all_files();  // rewrite all .txt files from memory (safe if editor clobbered them)

    // Task control
    void start_task(TaskType t);
    void pause_task(TaskType t);
    void stop_all();

    // GPU backend (abstract — Metal on macOS, Vulkan on Windows, CPU fallback)
    void set_gpu(GPUBackend *gpu) { _gpu = gpu; }
    GPUBackend* gpu() const { return _gpu; }
    std::mutex& gpu_mutex() { return _gpu_mutex; }

    // Callbacks
    void set_log_callback(LogCallback cb) { _log_cb = cb; }
    void set_discovery_callback(DiscoveryCallback cb) { _disc_cb = cb; }

    // Access
    std::map<TaskType, SearchTask>& tasks() { return _tasks; }
    const std::vector<Discovery>& discoveries() const { return _discoveries; }
    ThreadPool& pool() { return *_pool; }

    // Public logging (for static helper functions)
    void log_msg(const std::string& msg) { log(msg); }

    // Segmented sieve (public for pair search helper)
    std::vector<uint64_t> segmented_sieve(uint64_t lo, uint64_t hi);
    const MatrixSieve* matrix_sieve() const { return _matrix_sieve.get(); }
    const PseudoprimePredictor* predictor() const { return _predictor.get(); }
    std::string timestamp();

    // Periodic scan summary to per-test result files
    void save_scan_summary(TaskType t, uint64_t range_lo, uint64_t range_hi,
                           uint64_t tested, uint64_t hits);

    // GPU pacing — enforce minimum gap between dispatches for ~50% util
    void pace_gpu();
    void finish_gpu();

    // UI responsiveness — throttle workers when user is interacting
    void signal_ui_activity();
    bool should_throttle() const;
    static constexpr double THROTTLE_SECONDS = 10.0;  // yield CPU for 10s after last mouse/keyboard activity

    // Mersenne TF sieve batch size (configurable from GIMPS panel)
    std::atomic<uint64_t> mersenne_k_batch{100000000}; // default 100M

    // Mersenne TF: abort on first factor (default false = complete full bitlevel)
    std::atomic<bool> mersenne_abort_on_factor{false};

    // Mode-based GPU ownership: only one task type uses GPU at a time.
    // GPU_OWNER_NONE = shared (round-robin via mutex), any specific type = exclusive.
    // Mersenne TF sets this to MersenneTrial. Other tasks check and go CPU-only.
    std::atomic<int> gpu_owner{-1}; // -1 = shared, else = (int)TaskType owner

    // Carry-chain mulmod toggle: when true, use hardware carry-chain approach
    // instead of binary shift-and-add for 128-bit modular multiplication
    std::atomic<bool> use_carry_chain{false};

private:
    std::string _data_dir;
    std::map<TaskType, SearchTask> _tasks;
    std::vector<Discovery> _discoveries;
    std::mutex _mutex;
    std::mutex _save_mutex;
    std::mutex _gpu_mutex;  // serialize GPU access across worker threads
    std::atomic<int64_t> _last_gpu_dispatch_us{0}; // timestamp of last GPU dispatch (microseconds)
    static constexpr int64_t GPU_MIN_GAP_US = 2000; // 2ms min gap between GPU dispatches → ~50% util
    GPUBackend *_gpu = nullptr;
    LogCallback _log_cb;
    DiscoveryCallback _disc_cb;

    void log(const std::string& msg);

    // Worker functions for each task type
    void run_wieferich(SearchTask& task);
    void run_wallsunsun(SearchTask& task);
    void run_wilson(SearchTask& task);
    void run_twin(SearchTask& task);
    void run_sophie(SearchTask& task);
    void run_cousin(SearchTask& task);
    void run_sexy(SearchTask& task);
    void run_general(SearchTask& task);
    void run_emirp(SearchTask& task);
    void run_mersenne_trial(SearchTask& task);
    void run_fermat_factor(SearchTask& task);

    std::vector<uint64_t> _small_primes;
    void ensure_small_primes(uint64_t up_to);
    std::unique_ptr<ThreadPool> _pool;  // shared thread pool for parallel sieve
    std::unique_ptr<MatrixSieve> _matrix_sieve;  // matrix-based pre-filter (ANE/SIMD)
    std::unique_ptr<PseudoprimePredictor> _predictor;  // pre-generates pseudoprimes

    // Dedup: known discoveries keyed by (type, value) to avoid double-reporting
    std::set<std::pair<TaskType, uint64_t>> _known_values;
    void populate_known_primes();
    bool is_known(TaskType t, uint64_t v) const;

    // UI throttle state
    std::atomic<int64_t> _last_ui_activity_ms{0};  // ms since epoch of last interaction

    // NEON load balancer — shared across all tasks
    std::unique_ptr<LoadBalancer> _balancer;

public:
    LoadBalancer* balancer() { return _balancer.get(); }
};

} // namespace prime
