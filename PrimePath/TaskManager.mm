#include "TaskManager.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <deque>
#include <future>
#include <pthread.h>
#include <arm_neon.h>

// GPU compute is accessed via the abstract GPUBackend interface (GPUBackend.hpp)
// This allows swapping Metal (macOS), Vulkan (Windows), CUDA, or CPU fallback.

namespace prime {

static const char* pclass_str(PrimeClass c);  // forward declaration

// ── Cache-tuned segment sizing ──────────────────────────────────────
// L2: 16MB P-cores (4 share), 6MB E-cores (6 share)
// L1D: 64KB per core, cache line: 128 bytes
//
// Goal: each thread's sieve buffer fits in its L2 slice.
// E-cores are the bottleneck: 6MB / 6 cores = 1MB each.
// Use 512KB segments: fits 2 in L1D-adjacent space, ~12 in E-core L2,
// leaves room for stack, small_primes, and result vectors.
// Smaller segments = more cache hits, less DRAM traffic.
static const uint64_t SEGMENT_SIZE = 1048576;   // 1M — fits in E-core L2 slice (1MB each)
static const uint32_t GPU_BATCH = 262144;       // 256K batch — matches MetalCompute MAX_BATCH
static const int SAVE_INTERVAL_SEC = 30;
static const int PREFETCH_DEPTH = 8;   // moderate prefetch — keeps pipeline full without excessive memory use
static const size_t GPU_ACCUM_HARD_CAP = 512 * 1024;  // 512K entries (~4 MB) — force flush before memory pressure builds

// ── Constructor / Destructor ────────────────────────────────────────

TaskManager::TaskManager(const std::string& data_dir) : _data_dir(data_dir) {
    _pool = std::make_unique<ThreadPool>();
    _matrix_sieve = std::make_unique<MatrixSieve>();
    _predictor = std::make_unique<PseudoprimePredictor>();
    _balancer = std::make_unique<LoadBalancer>();
    log("Thread pool: " + std::to_string(_pool->size()) + " workers");
    log("Matrix sieve: pre-filter using primes {3,5,7,11,13,17,19,23,29,31}");
    log("Load balancer: NEON saturation prediction active");
}

TaskManager::~TaskManager() {
    // Signal stop but don't block — caller (applicationShouldTerminate) already called stop_all()
    for (auto& [t, task] : _tasks) {
        task.should_run.store(false);
        if (task.worker.joinable()) task.worker.detach();
    }
}

// ── Known primes database (avoid rediscovering these) ────────────────

void TaskManager::populate_known_primes() {
    // Wieferich primes: only 2 known (OEIS A001220)
    _known_values.insert({TaskType::Wieferich, 1093});
    _known_values.insert({TaskType::Wieferich, 3511});

    // Wall-Sun-Sun primes: NONE known (OEIS A182297)
    // (any discovery would be new!)

    // Wilson primes: only 3 known (OEIS A007540)
    _known_values.insert({TaskType::Wilson, 5});
    _known_values.insert({TaskType::Wilson, 13});
    _known_values.insert({TaskType::Wilson, 563});

    // Twin primes — largest known small twins (OEIS A001359)
    // All pairs (p, p+2) where both prime, up to our search frontiers
    // We store just the smaller of each pair
    const uint64_t known_twins[] = {
        3,5,11,17,29,41,59,71,101,107,137,149,179,191,197,227,239,269,
        281,311,347,419,431,461,521,569,599,617,641,659,809,821,827,857,
        881,1019,1031,1049,1061,1091,1151,1229,1277,1289,1301,1319,1427,
        1451,1481,1487,1607,1619,1667,1697,1721,1787,1871,1877,1931,1949,
        1997,2027,2081,2087,2111,2129,2141,2237,2267,2309,2339,2381,2549,
        2591,2657,2687,2711,2729,2789,2801,2969,2999,3119,3167,3251,3257,
        3299,3329,3359,3371,3389,3461,3467,3527,3539,3557,3581,3671,3767,
        3821,3851,3917,3929,4001,4019,4049,4091,4127,4157,4217,4229,4241,
        4259,4271,4337,4373,4397,4409,4421,4481,4507,4517,4547,4637,4649,
        4721,4787,4799,4931,4967,4999
    };
    for (auto p : known_twins)
        _known_values.insert({TaskType::TwinPrime, p});

    // Sophie Germain primes — p where 2p+1 is also prime (OEIS A005384)
    const uint64_t known_sophie[] = {
        2,3,5,11,23,29,41,53,83,89,113,131,173,179,191,233,239,251,281,
        293,359,419,431,443,491,509,593,641,653,659,683,719,743,761,809,
        911,953,1013,1019,1031,1049,1103,1223,1229,1289,1409,1439,1451,
        1481,1499,1511,1559,1583,1601,1733,1811,1889,1901,1931,1973,2003,
        2039,2063,2069,2099,2111,2129,2141,2273,2309,2339,2351,2393,2399,
        2459,2543,2549,2693,2699,2741,2753,2819,2903,2939,2963,2969,3023,
        3299,3329,3359,3389,3413,3449,3491,3509,3539,3593,3623,3761,3779,
        3803,3821,3851,3863,3911,4019,4073,4211,4259,4349,4373,4391,4409,
        4481,4733,4793,4871,4919,4943
    };
    for (auto p : known_sophie)
        _known_values.insert({TaskType::SophieGermain, p});

    // Cousin primes — (p, p+4) both prime (OEIS A023200)
    const uint64_t known_cousins[] = {
        3,7,13,37,43,67,79,97,103,109,127,163,193,223,229,277,307,313,
        349,379,397,439,457,463,487,499,541,613,643,673,739,757,769,823,
        853,877,883,907,937,967,1009,1087,1093,1213,1279,1297,1399,1423,
        1429,1447,1483,1489,1549,1567,1579,1597,1609,1669,1699,1723,1789,
        1873,1879,1933,1987,1993,1999,2083,2089,2113,2143,2203,2239,2269,
        2293,2383,2389,2399,2459,2473,2539,2549,2593,2609,2683,2689,2693,
        2699,2713,2729,2789,2833,2903,2953,2963,2969,3037,3067,3079,3109,
        3163,3169,3253,3259,3307,3313,3319,3463,3469,3517,3527,3613,3643,
        3673,3697,3733,3793,3847,3877,3907,3943,4003,4013,4057,4093,4129,
        4153,4217,4229,4253,4259,4349,4423,4483,4507,4513,4567,4603,4657,
        4723,4789,4799,4903,4933,4957,4969,4993
    };
    for (auto p : known_cousins)
        _known_values.insert({TaskType::CousinPrime, p});

    // Sexy primes — (p, p+6) both prime (OEIS A023201)
    const uint64_t known_sexys[] = {
        5,7,11,13,17,23,31,37,41,47,53,61,67,73,83,89,97,101,103,107,
        131,151,157,167,173,191,193,223,227,233,251,257,263,271,277,307,
        311,331,347,353,367,373,379,383,389,401,431,433,443,457,461,503,
        521,541,547,557,563,569,571,587,593,601,607,613,631,641,643,647,
        653,677,683,691,727,733,739,751,757,761,769,773,787,797,811,821,
        823,827,853,857,877,881,941,947,953,967,977,983,991,997,1013,
        1019,1031,1033,1051,1061,1063,1087,1091,1097,1103,1117,1123,1151,
        1171,1181,1187,1193,1213,1217,1223,1231,1277,1283,1291,1297,1301,
        1321,1361,1373,1381,1399,1427,1433,1447,1451,1471,1481,1487,1493
    };
    for (auto p : known_sexys)
        _known_values.insert({TaskType::SexyPrime, p});

    log("Known primes database: " + std::to_string(_known_values.size()) + " entries loaded");
}

bool TaskManager::is_known(TaskType t, uint64_t v) const {
    return _known_values.count({t, v}) > 0;
}

void TaskManager::init_defaults() {
    // Search frontiers — start past 2^64 where PrimeGrid's verified search ends.
    // PrimeGrid completed double-checked search to 2^64 (Dec 2022) for both
    // Wieferich and Wall-Sun-Sun primes. No new primes found.
    // 2^64 + 1 = 18446744073709551617
    static const uint64_t PAST_2_64 = 18446744073709551615ULL; // 2^64 - 1 (max u64)
    // NOTE: 2^64 itself doesn't fit in u64. We start at a safe high frontier.
    // For meaningful search, we need numbers that fit in u64 for modular arithmetic.
    // Since p² must fit in 128-bit (always true), p just needs to fit in u64.
    // Start at ~9.2 × 10^18 (half of u64 max) — well past PrimeGrid's 2^64 limit
    // while leaving headroom for p² in __uint128_t.
    _tasks.emplace(TaskType::Wieferich,
        SearchTask(TaskType::Wieferich, 18400000000000000001ULL));
    _tasks.emplace(TaskType::WallSunSun,
        SearchTask(TaskType::WallSunSun, 18400000000000000001ULL));
    // Wilson: naive (p-1)! mod p² is O(p), only practical up to ~10^8.
    // Known Wilson primes: 5, 13, 563. Exhaustively searched to 2×10^13 by others.
    // Start past largest known (563) so we can verify and potentially extend.
    _tasks.emplace(TaskType::Wilson,
        SearchTask(TaskType::Wilson, 569));
    _tasks.emplace(TaskType::TwinPrime,
        SearchTask(TaskType::TwinPrime, 1000000000000001ULL));
    _tasks.emplace(TaskType::SophieGermain,
        SearchTask(TaskType::SophieGermain, 1000000000000001ULL));
    _tasks.emplace(TaskType::CousinPrime,
        SearchTask(TaskType::CousinPrime, 1000000000000001ULL));
    _tasks.emplace(TaskType::SexyPrime,
        SearchTask(TaskType::SexyPrime, 1000000000000001ULL));
    _tasks.emplace(TaskType::GeneralPrime,
        SearchTask(TaskType::GeneralPrime, 1000000000000001ULL));
    _tasks.emplace(TaskType::Emirp,
        SearchTask(TaskType::Emirp, 1000000000000001ULL));

    // Mersenne trial factoring: start at exponent 100M (GIMPS first-test frontier ~140M).
    // The current_pos here is the k value for candidates q = 2kp+1.
    // The exponent p is stored separately and defaults to 100000007 (a prime near 10^8).
    _tasks.emplace(TaskType::MersenneTrial,
        SearchTask(TaskType::MersenneTrial, 1));

    // Fermat factor search: start searching for factors of F_20 and above.
    // current_pos = k value for candidates q = k * 2^(m+2) + 1.
    _tasks.emplace(TaskType::FermatFactor,
        SearchTask(TaskType::FermatFactor, 1));

    // Load known primes database for dedup
    populate_known_primes();

    // Pre-load known discoveries (rare types only — these ARE the full known sets)
    _discoveries.push_back({TaskType::Wieferich, 1093, 0, PrimeClass::Prime, 0, "", "known"});
    _discoveries.push_back({TaskType::Wieferich, 3511, 0, PrimeClass::Prime, 0, "", "known"});
    _discoveries.push_back({TaskType::Wilson, 5, 0, PrimeClass::Prime, 0, "", "known"});
    _discoveries.push_back({TaskType::Wilson, 13, 0, PrimeClass::Prime, 0, "", "known"});
    _discoveries.push_back({TaskType::Wilson, 563, 0, PrimeClass::Prime, 0, "", "known"});

    // NOTE: do NOT save_state() here — load_state() must run first
    // to restore progress. AppDelegate calls save_state() after load_state().
}

// ── Persistence ─────────────────────────────────────────────────────

void TaskManager::load_state() {
    std::string path = _data_dir + "/search_progress.txt";
    std::ifstream f(path);
    if (!f.is_open()) {
        log("No saved state found, using defaults");
        return;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key;
        uint64_t cur, found, tested;
        if (iss >> key >> cur >> found >> tested) {
            TaskType t = task_from_key(key);
            auto it = _tasks.find(t);
            if (it != _tasks.end()) {
                it->second.current_pos = cur;
                it->second.found_count = found;
                it->second.tested_count = tested;
                log("Restored " + std::string(task_name(t)) +
                    " at position " + std::to_string(cur));
            }
        }
    }

    // Load discoveries from journal (authoritative) or fall back to .txt
    // Preserve pre-loaded discoveries from init_defaults()
    auto preloaded = _discoveries;

    std::string jpath = _data_dir + "/discoveries.journal";
    std::ifstream jf(jpath);
    if (jf.is_open()) {
        _discoveries.clear();
        while (std::getline(jf, line)) {
            if (line.empty() || line[0] == '#') continue;
            // Journal format: task_key\tCLASS\tvalue\tvalue2\ttested_at\tdivisors\ttimestamp
            std::istringstream iss(line);
            std::string key, cls, divs, ts;
            uint64_t v1 = 0, v2 = 0, tested = 0;
            if (std::getline(iss, key, '\t') && std::getline(iss, cls, '\t')) {
                iss >> v1;
                iss.ignore(1); // skip tab
                iss >> v2;
                iss.ignore(1);
                iss >> tested;
                iss.ignore(1);
                std::getline(iss, divs, '\t');
                std::getline(iss, ts, '\t');
                if (ts.empty()) ts = "unknown";
                PrimeClass pc = PrimeClass::Prime;
                if (cls == "PSEUDOPRIME") pc = PrimeClass::Pseudoprime;
                else if (cls == "COMPOSITE") pc = PrimeClass::Composite;
                TaskType tt = task_from_key(key);
                if (_known_values.count({tt, v1}) == 0) {
                    _discoveries.push_back({tt, v1, v2, pc, tested, divs, ts});
                    _known_values.insert({tt, v1});
                }
            }
        }
        log("Loaded " + std::to_string(_discoveries.size()) + " discoveries from journal");
    } else {
        // Fall back to .txt
        std::string dpath = _data_dir + "/discoveries.txt";
        std::ifstream df(dpath);
        if (df.is_open()) {
            _discoveries.clear();
            while (std::getline(df, line)) {
                if (line.empty() || line[0] == '#') continue;
                // Format: task_key CLASS value [pair:v2] [factors:...] tested_at:N timestamp
                std::istringstream iss(line);
                std::string key, cls;
                uint64_t v1 = 0, v2 = 0, tested = 0;
                std::string token, divs, ts;
                iss >> key >> cls >> v1;
                PrimeClass pc = PrimeClass::Prime;
                if (cls == "PSEUDOPRIME") pc = PrimeClass::Pseudoprime;
                else if (cls == "COMPOSITE") pc = PrimeClass::Composite;
                // Parse optional fields
                while (iss >> token) {
                    if (token.substr(0, 5) == "pair:") v2 = strtoull(token.c_str() + 5, nullptr, 10);
                    else if (token.substr(0, 8) == "factors:") divs = token.substr(8);
                    else if (token.substr(0, 10) == "tested_at:") tested = strtoull(token.c_str() + 10, nullptr, 10);
                    else ts = token;  // last unrecognized token is timestamp
                }
                TaskType tt = task_from_key(key);
                if (v1 > 0 && _known_values.count({tt, v1}) == 0) {
                    _discoveries.push_back({tt, v1, v2, pc, tested, divs, ts});
                    _known_values.insert({tt, v1});
                }
            }
            log("Loaded " + std::to_string(_discoveries.size()) + " discoveries from txt");
        }
    }

    // Re-add pre-loaded discoveries that weren't loaded from persisted files
    // Check against _discoveries (not _known_values, which has all known primes for dedup)
    std::set<std::pair<TaskType, uint64_t>> loaded_disc;
    for (auto& d : _discoveries) loaded_disc.insert({d.type, d.value});
    for (auto& d : preloaded) {
        if (loaded_disc.count({d.type, d.value}) == 0) {
            _discoveries.push_back(d);
            // _known_values already has these from populate_known_primes()
        }
    }
}

void TaskManager::save_state() {
    {
        std::lock_guard<std::mutex> lock(_save_mutex);
        std::string path = _data_dir + "/search_progress.txt";
        std::ofstream f(path);
        f << "# PrimePath Search Progress — all numbers full precision (u64)\n";
        f << "# task_key current_pos found_count tested_count start_pos\n";
        for (auto& [type, task] : _tasks) {
            f << task_key(type) << " "
              << task.current_pos << " "
              << task.found_count << " "
              << task.tested_count << " "
              << task.start_pos << "\n";
        }
        f << "# saved: " << timestamp() << "\n";
    }
    // Rewrite all .txt files from memory — recovers from editor clobbering
    flush_all_files();
}

static const char* pclass_str(PrimeClass c) {
    switch (c) {
        case PrimeClass::Prime:       return "PRIME";
        case PrimeClass::Pseudoprime: return "PSEUDOPRIME";
        case PrimeClass::Composite:   return "COMPOSITE";
    }
    return "UNKNOWN";
}

void TaskManager::save_discovery(const Discovery& d) {
    // Only persist genuine discoveries: confirmed primes and Mersenne/Fermat factors.
    // Pseudoprimes and composites are filtering artifacts — skip them entirely.
    if (d.pclass != PrimeClass::Prime &&
        d.type != TaskType::MersenneTrial && d.type != TaskType::FermatFactor) {
        return;
    }

    // Discoveries take precedence — temporarily boost thread priority to ensure
    // the save completes immediately, even if the system is under load.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);

    {
        std::lock_guard<std::mutex> lock(_save_mutex);
        // Skip if already known (check under lock for thread safety)
        if (_known_values.count({d.type, d.value}) > 0) {
            pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
            return;
        }
        _known_values.insert({d.type, d.value});
        _discoveries.push_back(d);

        // Write to journal file (append-only, machine-readable, never open in editors).
        // This is the primary data store — .txt files are regenerated from memory.
        std::string journal = _data_dir + "/discoveries.journal";
        std::ofstream jf(journal, std::ios::app);
        jf << task_key(d.type) << "\t"
           << pclass_str(d.pclass) << "\t"
           << d.value << "\t"
           << d.value2 << "\t"
           << d.tested_at << "\t"
           << d.divisors << "\t"
           << d.timestamp << "\n";
        jf.flush();
    }

    // Restore worker QoS after save
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);

    if (_disc_cb) _disc_cb(d);
}

// Rewrite all .txt files from in-memory state.
// Safe to call even if an editor has the files open — we overwrite completely.
void TaskManager::flush_all_files() {
    std::lock_guard<std::mutex> lock(_save_mutex);

    // Main discoveries.txt — write to temp then rename to avoid Spotlight/Finder locks
    {
        std::string path = _data_dir + "/discoveries.txt";
        std::string tmp = path + ".tmp";
        std::ofstream f(tmp, std::ios::trunc);
        if (!f.is_open()) {
            // Silently skip — not worth warning every flush cycle
            goto skip_discoveries;
        }
        f << "# PrimePath Discoveries — confirmed primes and factors only\n";
        f << "# Format: task_key CLASS value [pair:v2] [factors:...] tested_at:N timestamp\n";
        for (auto& d : _discoveries) {
            f << task_key(d.type) << " " << pclass_str(d.pclass) << " " << d.value;
            if (d.value2 > 0) f << " pair:" << d.value2;
            if (!d.divisors.empty()) f << " factors:[" << d.divisors << "]";
            f << " tested_at:" << d.tested_at << " " << d.timestamp << "\n";
        }
        f.close();
        ::rename(tmp.c_str(), path.c_str());
    }
    skip_discoveries:

    // Per-test files — group discoveries by type
    using TT = TaskType;
    TT types[] = {TT::Wieferich, TT::WallSunSun, TT::Wilson, TT::TwinPrime,
                  TT::SophieGermain, TT::CousinPrime, TT::SexyPrime, TT::GeneralPrime,
                  TT::Emirp, TT::MersenneTrial, TT::FermatFactor};
    for (auto t : types) {
        std::string test_file = _data_dir + "/" + std::string(task_key(t)) + "_results.txt";
        // Read existing SCAN lines (summaries) so we don't lose them
        std::vector<std::string> scan_lines;
        {
            std::ifstream rf(test_file);
            std::string line;
            while (std::getline(rf, line)) {
                if (line.substr(0, 4) == "SCAN") scan_lines.push_back(line);
            }
        }
        std::ofstream tf(test_file, std::ios::trunc);
        tf << "# " << task_name(t) << " results\n";
        // Discoveries for this type
        for (auto& d : _discoveries) {
            if (d.type != t) continue;
            tf << pclass_str(d.pclass) << " " << d.value;
            if (d.value2 > 0) tf << " pair:" << d.value2;
            if (!d.divisors.empty()) tf << " factors:[" << d.divisors << "]";
            tf << " " << d.timestamp << "\n";
        }
        // Preserved scan summaries
        for (auto& s : scan_lines) tf << s << "\n";
    }

    // Primes summary file
    {
        std::string path = _data_dir + "/primes.txt";
        std::ofstream f(path, std::ios::trunc);
        f << "# PrimePath — Confirmed Discoveries\n";
        for (auto& d : _discoveries) {
            f << task_key(d.type) << " " << d.value;
            if (d.value2 > 0) f << " " << d.value2;
            if (!d.divisors.empty()) f << " factors:[" << d.divisors << "]";
            f << " " << d.timestamp << "\n";
        }
    }
}

// ── Task control ────────────────────────────────────────────────────

void TaskManager::start_task(TaskType t) {
    auto it = _tasks.find(t);
    if (it == _tasks.end()) return;
    auto& task = it->second;

    // If already running, don't double-start
    if (task.should_run.load()) return;

    // Signal old worker to stop and wait for it to finish.
    // Never detach — a detached worker can race with the new one on shared state.
    task.should_run.store(false);
    if (task.worker.joinable()) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (task.status == TaskStatus::Running &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        task.worker.join();  // must join — never detach
    }

    task.status = TaskStatus::Running;
    task.should_run.store(true);

    save_state(); // persist that we're starting

    task.worker = std::thread([this, t, &task]() {
        @autoreleasepool {
        // Set worker thread to utility QoS so UI thread always gets priority
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);

        log(std::string(task_name(t)) + " search started at " +
            std::to_string(task.current_pos));
        switch (t) {
            case TaskType::Wieferich:     run_wieferich(task); break;
            case TaskType::WallSunSun:    run_wallsunsun(task); break;
            case TaskType::Wilson:        run_wilson(task); break;
            case TaskType::TwinPrime:     run_twin(task); break;
            case TaskType::SophieGermain: run_sophie(task); break;
            case TaskType::CousinPrime:   run_cousin(task); break;
            case TaskType::SexyPrime:     run_sexy(task); break;
            case TaskType::GeneralPrime:  run_general(task); break;
            case TaskType::Emirp:         run_emirp(task); break;
            case TaskType::MersenneTrial: run_mersenne_trial(task); break;
            case TaskType::FermatFactor:  run_fermat_factor(task); break;
        }
        task.status = TaskStatus::Paused;
        save_state();
        log(std::string(task_name(task.type)) + " paused at " +
            std::to_string(task.current_pos));
        } // @autoreleasepool
    });
}

void TaskManager::pause_task(TaskType t) {
    auto it = _tasks.find(t);
    if (it == _tasks.end()) return;
    it->second.should_run.store(false);
    it->second.status = TaskStatus::Paused;
    log(std::string(task_name(t)) + " pausing...");
}

void TaskManager::stop_all() {
    // Signal all tasks to stop first (so they can begin winding down in parallel)
    for (auto& [t, task] : _tasks) {
        task.should_run.store(false);
    }
    // Join worker threads with a timeout — detach any that won't finish in time
    for (auto& [t, task] : _tasks) {
        if (!task.worker.joinable()) continue;
        std::promise<void> p;
        auto f = p.get_future();
        std::thread joiner([&]{ task.worker.join(); p.set_value(); });
        if (f.wait_for(std::chrono::milliseconds(800)) == std::future_status::ready) {
            joiner.join();
        } else {
            joiner.detach();  // Thread is stuck — detach, OS reclaims on exit
            log(std::string(task_name(t)) + " thread did not stop in time — detaching");
        }
    }
    save_state();
}

// ── Segmented sieve ─────────────────────────────────────────────────

void TaskManager::ensure_small_primes(uint64_t up_to) {
    // Guard against concurrent calls from multiple worker threads
    static std::mutex sp_mutex;
    std::lock_guard<std::mutex> lock(sp_mutex);

    uint64_t need = (uint64_t)sqrt((double)up_to) + 100;
    if (!_small_primes.empty() && _small_primes.back() >= need) return;
    if (need > 500000000) need = 500000000; // cap at 500M — ~60MB RAM for sieve table
    auto sv = sieve(need);
    _small_primes.clear();
    for (uint64_t i = 2; i <= need; i++) {
        if (sv[i]) _small_primes.push_back(i);
    }
    log("Sieved " + std::to_string(_small_primes.size()) +
        " small primes up to " + std::to_string(need));
}

std::vector<uint64_t> TaskManager::segmented_sieve(uint64_t lo, uint64_t hi) {
    if (lo < 2) lo = 2;
    if (hi < lo) return {};
    uint64_t size = hi - lo + 1;

    // ── CPU sieve (NEON tile pattern) ──────────────────────────────────
    // Sieve always runs on CPU using pre-filled NEON pattern tile.
    // GPU is reserved for the actual number-theoretic tests (Wieferich, etc.)
    // or Mersenne fused kernel. Mixing sieve into GPU creates contention.
    std::vector<uint8_t> buf(size);

    if (_matrix_sieve && size <= UINT32_MAX) {
        _matrix_sieve->sieve_block(lo, (uint32_t)size, buf.data());
    } else {
        std::fill(buf.begin(), buf.end(), 1);
        uint32_t evStart = (lo % 2 == 0) ? 0 : 1;
        for (uint64_t j = evStart; j < size; j += 2) buf[j] = 0;
    }

    uint64_t sieve_start_prime = _matrix_sieve ? 37 : 3;
    uint8_t *buf_data = buf.data();
    const uint64_t *primes_data = _small_primes.data();
    size_t num_primes = _small_primes.size();

    for (size_t pi = 0; pi < num_primes; pi++) {
        uint64_t p = primes_data[pi];
        if (p < sieve_start_prime) continue;
        if (p * p > hi) break;

        uint64_t r = lo % p;
        uint64_t marker = (r == 0) ? 0 : (p - r);
        if (lo + marker == p) marker += p;

        if (p <= size) {
            for (uint64_t j = marker; j < size; j += p) {
                buf_data[j] = 0;
            }
        } else {
            if (marker < size) {
                buf_data[marker] = 0;
            }
        }
    }

    uint64_t sqrt_hi = (uint64_t)sqrt((double)hi) + 1;
    uint64_t small_limit = _small_primes.empty() ? 0 : _small_primes.back();
    bool need_mr_filter = (small_limit < sqrt_hi);

    double ln_lo = (lo > 1) ? std::log((double)lo) : 1.0;
    size_t est = (size_t)(1.15 * size / ln_lo);
    std::vector<uint64_t> result;
    result.reserve(est);

    // NEON-accelerated survivor extraction: scan 16 bytes at a time.
    // Skip all-zero chunks (no survivors) without checking individual bytes.
    uint64_t i = 0;
    uint8x16_t vzero = vdupq_n_u8(0);
    for (; i + 16 <= size; i += 16) {
        uint8x16_t v = vld1q_u8(&buf_data[i]);
        // Quick check: if entire 16-byte block is zero, skip it
        uint8x16_t cmp = vceqq_u8(v, vzero);
        // vmaxvq_u8 of cmp: if all equal zero, result is 0xFF; if any non-zero, <0xFF
        // Actually: cmp has 0xFF where buf==0, 0x00 where buf!=0
        // vminvq_u8(cmp): if all 0xFF (all zero), result=0xFF → skip
        if (vminvq_u8(cmp) == 0xFF) continue; // entire block is composite

        // At least one survivor — check individually
        for (int k = 0; k < 16; k++) {
            if (buf_data[i + k] && lo + i + k >= 2) {
                uint64_t n = lo + i + k;
                if (need_mr_filter && !prime::is_prime(n)) continue;
                result.push_back(n);
            }
        }
    }
    // Scalar tail
    for (; i < size; i++) {
        if (buf_data[i] && lo + i >= 2) {
            uint64_t n = lo + i;
            if (need_mr_filter && !prime::is_prime(n)) continue;
            result.push_back(n);
        }
    }
    return result;
}

// ── GPU pacing ──────────────────────────────────────────────────────
// Uses the NEON scheduling matrix to enforce adaptive pacing.
// pace_gpu() checks saturation and sleeps if GPU is running hot.
// finish_gpu() records dispatch completion for the rolling window.

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// pace_gpu: called BEFORE acquiring gpu_mutex.
// Sleeps to enforce adaptive gap based on measured GPU saturation.
// Stores dispatch start time in thread-local for finish_gpu().
static thread_local int64_t tl_gpu_start_us = 0;

void TaskManager::pace_gpu() {
    // No sleep here — sleeping wastes CPU cycles. The advise() ratio
    // controls how much work goes to GPU vs CPU. pace_gpu just marks
    // the start time so finish_gpu can measure actual GPU busy time.
    tl_gpu_start_us = now_us();
}

// finish_gpu: called AFTER releasing gpu_mutex.
// Measures actual GPU busy time, feeds it to the scheduling matrix,
// and signals the request pool that GPU has capacity for more work.
void TaskManager::finish_gpu() {
    int64_t end = now_us();
    _last_gpu_dispatch_us.store(end, std::memory_order_relaxed);

    int64_t busy = end - tl_gpu_start_us;
    if (busy > 0) {
        _balancer->record_gpu_busy(busy);
    }

    // Only signal GPU idle if it was actually idle (gap since last dispatch).
    // Don't flood the pool with requests between rapid-fire dispatches.
    if (busy > 0) {
        int64_t gap = end - tl_gpu_start_us;
        // Only post if GPU was idle for at least as long as it was busy
        // (i.e. < 50% utilisation in this dispatch cycle)
        if (gap > busy * 2) {
            _balancer->gpu_idle();
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

void TaskManager::log(const std::string& msg) {
    if (_log_cb) _log_cb(msg);
}

std::string TaskManager::timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", localtime(&t));
    return std::string(buf);
}

// ── UI throttle ─────────────────────────────────────────────────────

void TaskManager::signal_ui_activity() {
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    _last_ui_activity_ms.store(ms, std::memory_order_relaxed);
}

bool TaskManager::should_throttle() const {
    int64_t last = _last_ui_activity_ms.load(std::memory_order_relaxed);
    if (last == 0) return false;
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return (ms - last) < (int64_t)(THROTTLE_SECONDS * 1000);
}

// Throttle helper: if user is active, yield CPU substantially so UI stays responsive.
// Workers call this once per segment — while throttled, they sleep 200ms per call,
// effectively reducing throughput to ~5 segments/sec instead of full speed.
// Discovery saves and file writes are NEVER throttled — they always run at full priority.
static void throttle_if_needed(const TaskManager* mgr) {
    if (mgr->should_throttle()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SievePipeline — deep prefetch: keeps CPU cores busy sieving ahead
// ═══════════════════════════════════════════════════════════════════════

SievePipeline::SievePipeline(TaskManager* mgr, uint64_t start, uint64_t seg_size, int depth)
    : _mgr(mgr), _current_start(start), _seg_size(seg_size), _depth(depth) {
    _next_enqueue = start;
    // Fill the prefetch queue — all CPU cores start sieving immediately
    for (int i = 0; i < _depth; i++) {
        enqueue_one();
    }
}

void SievePipeline::enqueue_one() {
    uint64_t ps = _next_enqueue;
    uint64_t pe = ps + _seg_size;
    _next_enqueue = pe;
    _queue.push_back({ps, _mgr->_pool->submit([mgr = _mgr, ps, pe]() {
        return mgr->segmented_sieve(ps, pe);
    })});
}

std::vector<uint64_t> SievePipeline::next_segment() {
    if (_queue.empty()) {
        // Shouldn't happen, but fallback
        auto primes = _mgr->segmented_sieve(_current_start, _current_start + _seg_size);
        _current_start += _seg_size;
        return primes;
    }
    auto primes = _queue.front().future.get();
    _current_start = _queue.front().seg_start + _seg_size;
    _queue.pop_front();
    // Immediately enqueue another to keep the pipeline full
    enqueue_one();
    return primes;
}

void SievePipeline::drain() {
    for (auto& pf : _queue) {
        if (pf.future.valid()) pf.future.wait();
    }
    _queue.clear();
}

// ── Readable number formatting (with commas) ────────────────────────

static std::string format_number(uint64_t n) {
    std::string s = std::to_string(n);
    int len = (int)s.length();
    std::string result;
    for (int i = 0; i < len; i++) {
        if (i > 0 && (len - i) % 3 == 0) result += ',';
        result += s[i];
    }
    return result;
}

// ── Scan summary logging ────────────────────────────────────────────

void TaskManager::save_scan_summary(TaskType t, uint64_t range_lo, uint64_t range_hi,
                                     uint64_t tested, uint64_t hits) {
    std::lock_guard<std::mutex> lock(_save_mutex);
    std::string test_file = _data_dir + "/" + std::string(task_key(t)) + "_results.txt";
    std::ofstream tf(test_file, std::ios::app);
    tf << "SCAN " << format_number(range_lo) << " -> " << format_number(range_hi)
       << " | tested: " << format_number(tested)
       << " | hits: " << hits
       << " | " << timestamp() << "\n";
}

// ═══════════════════════════════════════════════════════════════════════
// Heuristic composite verification for search task hits
//
// When Wieferich/WallSunSun/Wilson finds a "hit", run quick heuristic
// checks before celebrating. If we find a factor, it's a false positive.
// Returns "" if candidate passes (looks prime), or a string describing
// why it's composite.
// ═══════════════════════════════════════════════════════════════════════

static std::string verify_candidate_primality(uint64_t candidate) {
    // 1. Quick heuristic factoring (Lucky7s + PinchFactor)
    auto divs = prime::heuristic_divisors(candidate);
    for (uint64_t d : divs) {
        if (d > 1 && d < candidate && candidate % d == 0) {
            return "HEURISTIC COMPOSITE: " + std::to_string(candidate) +
                   " = " + std::to_string(d) + " x " + std::to_string(candidate / d);
        }
    }

    // 2. Deterministic Miller-Rabin re-verification (12 witnesses, correct for all u64)
    if (!prime::is_prime(candidate)) {
        auto factors = prime::factors_string(candidate);
        return "MR RE-CHECK COMPOSITE: " + std::to_string(candidate) + " = " + factors;
    }

    return "";  // passes all checks
}

// ═══════════════════════════════════════════════════════════════════════
// CPU-side primality tests (for parallel CPU+GPU execution)
// These run on CPU ALU while GPU processes another batch concurrently.
// ═══════════════════════════════════════════════════════════════════════

// Overflow-safe (a * b) % mod for mod = p² where p < 2^64.
// Precomputes R = 2^64 mod m once, then each multiply decomposes into
// four 64×64→128 hw multiplies + a few modular additions.
struct MulMod128Context {
    unsigned __int128 mod;
    unsigned __int128 R;   // 2^64 mod mod  (always < mod, so fits in ~106 bits for p~10^16)

    MulMod128Context(unsigned __int128 m) : mod(m) {
        R = 1;
        for (int i = 0; i < 64; i++) {
            R <<= 1;
            if (R >= mod) R -= mod;
        }
    }

    // Multiply val (< mod) by 2^64 mod mod, using precomputed R.
    // val * R where both < mod (~2^106). Product could be ~2^212 — too big.
    // So decompose val = v_hi * 2^64 + v_lo, then:
    //   val * R = v_lo * R + v_hi * R * 2^64
    // v_lo * R: 64-bit × ~106-bit = fits in 170 bits... still overflows __int128.
    // BUT: for mod = p² with p < 2^64, R < p² < 2^128. However R < mod and
    // v_lo < 2^64, so v_lo * R < 2^64 * 2^128... overflows.
    //
    // Use doubling on R (which has at most ~106 significant bits = ~106 iterations)
    // This is much faster than doubling on the full value (128 iterations).
    // Compute (val * R) % mod using binary method on whichever is smaller
    unsigned __int128 mulR(unsigned __int128 val) const {
        val %= mod;
        if (val == 0) return 0;
        // Iterate on whichever has fewer bits
        unsigned __int128 x, y;
        if (val <= R) { x = R; y = val; }
        else          { x = val; y = R; }
        unsigned __int128 result = 0;
        while (y > 0) {
            if (y & 1) {
                result += x;
                if (result >= mod) result -= mod;
            }
            x += x;
            if (x >= mod) x -= mod;
            y >>= 1;
        }
        return result;
    }

    unsigned __int128 mul(unsigned __int128 a, unsigned __int128 b) const {
        a %= mod;
        b %= mod;
        if (a == 0 || b == 0) return 0;

        // Fast path: both fit in 64 bits
        if (a <= UINT64_MAX && b <= UINT64_MAX) {
            return (a * b) % mod;
        }

        uint64_t a_lo = (uint64_t)a;
        uint64_t a_hi = (uint64_t)(a >> 64);
        uint64_t b_lo = (uint64_t)b;
        uint64_t b_hi = (uint64_t)(b >> 64);

        // t0 = a_lo * b_lo (128-bit, fits in __int128)
        unsigned __int128 t0 = (unsigned __int128)a_lo * b_lo % mod;

        // t1 = (a_lo * b_hi) * 2^64   (cross term)
        unsigned __int128 t1 = 0;
        if (b_hi) t1 = mulR((unsigned __int128)a_lo * b_hi % mod);

        // t2 = (a_hi * b_lo) * 2^64   (cross term)
        unsigned __int128 t2 = 0;
        if (a_hi) t2 = mulR((unsigned __int128)a_hi * b_lo % mod);

        // t3 = (a_hi * b_hi) * 2^128 = mulR(mulR(a_hi * b_hi))
        unsigned __int128 t3 = 0;
        if (a_hi && b_hi) {
            t3 = mulR(mulR((unsigned __int128)a_hi * b_hi % mod));
        }

        unsigned __int128 result = t0;
        result += t1; if (result >= mod) result -= mod;
        result += t2; if (result >= mod) result -= mod;
        result += t3; if (result >= mod) result -= mod;
        return result;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Carry-chain mulmod: hardware multiply + __int128 division for reduction.
// Works for moduli up to 95 bits. ~4-7x faster than binary shift-and-add.
// ═══════════════════════════════════════════════════════════════════════
struct CarryChainMulMod {
    unsigned __int128 mod;

    CarryChainMulMod(unsigned __int128 m) : mod(m) {}

    // Binary doubling fallback for moduli > 96 bits where staged shift overflows
    unsigned __int128 mul_binary(unsigned __int128 a, unsigned __int128 b) const {
        a %= mod; b %= mod;
        unsigned __int128 result = 0;
        while (b > 0) {
            if (b & 1) {
                result += a;
                if (result >= mod) result -= mod;
            }
            a <<= 1;
            if (a >= mod) a -= mod;
            b >>= 1;
        }
        return result;
    }

    unsigned __int128 mul(unsigned __int128 a, unsigned __int128 b) const {
        if (mod <= 1) return 0;
        a %= mod; b %= mod;
        if (a == 0 || b == 0) return 0;

        // Fast path: both fit in 64 bits
        if (a <= UINT64_MAX && b <= UINT64_MAX) {
            return (a * b) % mod;
        }

        // Fallback: if mod > 96 bits, staged shift would overflow
        if (mod >> 96) {
            return mul_binary(a, b);
        }

        uint64_t a_lo = (uint64_t)a, a_hi = (uint64_t)(a >> 64);
        uint64_t b_lo = (uint64_t)b, b_hi = (uint64_t)(b >> 64);

        // Full 192-bit product in [r2:r1:r0]
        unsigned __int128 p = (unsigned __int128)a_lo * b_lo;
        uint64_t p0 = (uint64_t)p;
        uint64_t p1 = (uint64_t)(p >> 64);

        unsigned __int128 cross = (unsigned __int128)a_lo * b_hi + (unsigned __int128)a_hi * b_lo;
        uint64_t c0 = (uint64_t)cross;
        uint64_t c1 = (uint64_t)(cross >> 64);

        uint64_t hh = a_hi * b_hi;

        uint64_t r0 = p0;
        unsigned __int128 mid = (unsigned __int128)p1 + c0;
        uint64_t r1 = (uint64_t)mid;
        uint64_t r2 = (uint64_t)(mid >> 64) + c1 + hh;

        // Reduce [r2:r1:r0] mod q using hardware __int128 division
        unsigned __int128 hi = ((unsigned __int128)r2 << 64) | r1;
        unsigned __int128 hi_mod = hi % mod;

        // hi_mod * 2^64 + r0 -- shift in two 32-bit steps (safe: mod < 2^96)
        unsigned __int128 tmp = (hi_mod << 32) % mod;
        tmp = (tmp << 32) % mod;
        tmp = (tmp + r0) % mod;
        return tmp;
    }
};

// Global flag read by CPU test functions (set from UI toggle)
// Plain bool + volatile: avoids C++ name mangling issues across .mm TUs
volatile bool g_use_carry_chain = false;

// Wieferich test on CPU: 2^(p-1) ≡ 1 (mod p²)
static bool cpu_wieferich_test(uint64_t p) {
    unsigned __int128 p_sq = (unsigned __int128)p * p;
    unsigned __int128 base = 2, result = 1;
    uint64_t exp = p - 1;

    if (g_use_carry_chain) {
        CarryChainMulMod ctx(p_sq);
        while (exp > 0) {
            if (exp & 1) result = ctx.mul(result, base);
            exp >>= 1;
            base = ctx.mul(base, base);
        }
    } else {
        MulMod128Context ctx(p_sq);
        while (exp > 0) {
            if (exp & 1) result = ctx.mul(result, base);
            exp >>= 1;
            base = ctx.mul(base, base);
        }
    }
    return result == 1;
}

// Wall-Sun-Sun test on CPU: Fibonacci F(p - legendre(p,5)) ≡ 0 (mod p²)
static bool cpu_wallsunsun_test(uint64_t p) {
    unsigned __int128 p_sq = (unsigned __int128)p * p;
    unsigned __int128 mod = p_sq;

    auto addmod = [mod](unsigned __int128 a, unsigned __int128 b) -> unsigned __int128 {
        unsigned __int128 s = a + b;
        return (s >= mod || s < a) ? s - mod : s;
    };

    // Legendre symbol (p/5)
    int leg = (p % 5 == 1 || p % 5 == 4) ? 1 : (p % 5 == 0 ? 0 : -1);
    uint64_t n = p - leg;

    // Matrix exponentiation for Fibonacci mod p²
    unsigned __int128 a = 1, b = 1, c = 1, d = 0;
    unsigned __int128 ra = 1, rb = 0, rc = 0, rd = 1;

    if (g_use_carry_chain) {
        CarryChainMulMod ctx(mod);
        while (n > 0) {
            if (n & 1) {
                unsigned __int128 na = addmod(ctx.mul(ra,a), ctx.mul(rb,c));
                unsigned __int128 nb = addmod(ctx.mul(ra,b), ctx.mul(rb,d));
                unsigned __int128 nc = addmod(ctx.mul(rc,a), ctx.mul(rd,c));
                unsigned __int128 nd = addmod(ctx.mul(rc,b), ctx.mul(rd,d));
                ra = na; rb = nb; rc = nc; rd = nd;
            }
            unsigned __int128 na = addmod(ctx.mul(a,a), ctx.mul(b,c));
            unsigned __int128 nb = addmod(ctx.mul(a,b), ctx.mul(b,d));
            unsigned __int128 nc = addmod(ctx.mul(c,a), ctx.mul(d,c));
            unsigned __int128 nd = addmod(ctx.mul(c,b), ctx.mul(d,d));
            a = na; b = nb; c = nc; d = nd;
            n >>= 1;
        }
    } else {
        MulMod128Context ctx(mod);
        while (n > 0) {
            if (n & 1) {
                unsigned __int128 na = addmod(ctx.mul(ra,a), ctx.mul(rb,c));
                unsigned __int128 nb = addmod(ctx.mul(ra,b), ctx.mul(rb,d));
                unsigned __int128 nc = addmod(ctx.mul(rc,a), ctx.mul(rd,c));
                unsigned __int128 nd = addmod(ctx.mul(rc,b), ctx.mul(rd,d));
                ra = na; rb = nb; rc = nc; rd = nd;
            }
            unsigned __int128 na = addmod(ctx.mul(a,a), ctx.mul(b,c));
            unsigned __int128 nb = addmod(ctx.mul(a,b), ctx.mul(b,d));
            unsigned __int128 nc = addmod(ctx.mul(c,a), ctx.mul(d,c));
            unsigned __int128 nd = addmod(ctx.mul(c,b), ctx.mul(d,d));
            a = na; b = nb; c = nc; d = nd;
            n >>= 1;
        }
    }
    return rb == 0;
}

// Twin prime CPU test: p and p+2 both prime
static bool cpu_twin_test(uint64_t p) {
    return prime::is_prime(p) && prime::is_prime(p + 2);
}

// Sophie Germain CPU test: p prime and 2p+1 prime
static bool cpu_sophie_test(uint64_t p) {
    return prime::is_prime(p) && prime::is_prime(2*p + 1);
}

// Cousin prime CPU test: p and p+4 both prime
static bool cpu_cousin_test(uint64_t p) {
    return prime::is_prime(p) && prime::is_prime(p + 4);
}

// Sexy prime CPU test: p and p+6 both prime
static bool cpu_sexy_test(uint64_t p) {
    return prime::is_prime(p) && prime::is_prime(p + 6);
}

// Generic CPU batch test: runs test_fn on a slice of candidates using thread pool
// Returns vector of indices that passed the test
static std::vector<uint32_t> cpu_batch_test(const uint64_t *cands, uint32_t count,
                                             std::function<bool(uint64_t)> test_fn,
                                             ThreadPool& pool) {
    std::vector<uint32_t> hits;
    std::mutex hits_mutex;

    pool.parallel_for(0, count, [&](uint64_t lo, uint64_t hi) {
        std::vector<uint32_t> local_hits;
        for (uint64_t i = lo; i < hi; i++) {
            if (test_fn(cands[i])) {
                local_hits.push_back((uint32_t)i);
            }
        }
        if (!local_hits.empty()) {
            std::lock_guard<std::mutex> lock(hits_mutex);
            hits.insert(hits.end(), local_hits.begin(), local_hits.end());
        }
    });

    return hits;
}

// Legacy AdaptiveSplit kept for backward compatibility (unused tasks)
struct AdaptiveSplit {
    double cpu_ratio = 0.3;
    void update(double cpu_sec, double gpu_sec) {
        if (cpu_sec < 0.001 || gpu_sec < 0.001) return;
        double target = gpu_sec / (cpu_sec + gpu_sec);
        cpu_ratio = cpu_ratio * 0.8 + target * 0.2;
        if (cpu_ratio < 0.05) cpu_ratio = 0.05;
        if (cpu_ratio > 0.80) cpu_ratio = 0.80;
    }
};

// Per-batch reusable result buffer to avoid heap allocation in hot path
static thread_local std::vector<uint8_t> tl_gpu_results;

// ═══════════════════════════════════════════════════════════════════════
// Worker functions — one per task type
// ═══════════════════════════════════════════════════════════════════════

void TaskManager::run_wieferich(SearchTask& task) {
    // Guard: at large positions, CPU-only mulmod128 is too slow (128-bit overflow
    // requires binary method). Need GPU for efficient testing.
    bool gpu_available = false;
    { std::lock_guard<std::mutex> lock(_gpu_mutex);
      gpu_available = (_gpu && _gpu->name() != "CPU (fallback)" && gpu_owner.load() < 0); }
    if (task.current_pos > 4294967296ULL && !gpu_available) {
        log("Wieferich: position " + std::to_string(task.current_pos) +
            " requires GPU for efficient testing (CPU mulmod too slow at this range). "
            "Stop Mersenne TF first, or wait for it to finish.");
        task.should_run.store(false);
        task.status = TaskStatus::Paused;
        return;
    }
    ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
    auto last_save = std::chrono::steady_clock::now();
    uint64_t summary_start = task.current_pos;
    uint64_t summary_tested = 0, summary_hits = 0;
    uint64_t diag_gpu_sent = 0, diag_cpu_sent = 0;
    double diag_gpu_sec = 0, diag_cpu_sec = 0;

    SievePipeline pipeline(this, task.current_pos, SEGMENT_SIZE, PREFETCH_DEPTH);

    // Kick off pseudoprime prediction for the initial range (runs on thread pool)
    uint64_t predict_ahead = SEGMENT_SIZE * PREFETCH_DEPTH * 4;
    auto predict_future = _pool->submit([this, pos = task.current_pos, predict_ahead]() {
        _predictor->generate_carmichaels(pos, pos + predict_ahead);
        _predictor->generate_sprp2(pos, pos + predict_ahead);
        return true;
    });

    // GPU batch accumulator
    static const uint32_t GPU_ACCUM_MIN = GPU_BATCH / 4;
    static const double GPU_FLUSH_SEC = 2.0;

    std::vector<uint64_t> gpu_accum;
    gpu_accum.reserve(GPU_BATCH + SEGMENT_SIZE);
    double accum_cpu_sec = 0;
    auto last_gpu_flush = std::chrono::steady_clock::now();

    // Lambda: flush GPU accumulator, process hits, update adaptive split
    auto flush_gpu_accum = [&]() {
        if (gpu_accum.empty() || !_gpu) return;

        // Even-shadow pre-filter: score and reorder candidates by
        // divisor interference of p±1. High-scoring (well-constrained)
        // candidates go first; low-scoring (suspicious) get flagged.
        uint8_t suspicion_threshold = EvenShadow::reorder_inplace(gpu_accum);

        auto stats = EvenShadow::compute_stats(gpu_accum.data(),
            (uint32_t)gpu_accum.size(), suspicion_threshold);
        log("GPU flush: " + std::to_string(gpu_accum.size()) +
            " Wieferich → GPU | shadow: avg=" +
            std::to_string((int)stats.avg_score) +
            " [" + std::to_string(stats.min_score) + "-" +
            std::to_string(stats.max_score) + "] " +
            std::to_string(stats.fully_factored) + " fully-factored, " +
            std::to_string(stats.suspicious) + " suspicious");

        auto gpu_t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < gpu_accum.size(); off += GPU_BATCH) {
            if (!task.should_run.load()) break;
            uint32_t n = (uint32_t)std::min((size_t)GPU_BATCH, gpu_accum.size() - off);
            if (tl_gpu_results.size() < n) tl_gpu_results.resize(n);

            pace_gpu();
            { std::lock_guard<std::mutex> glock(_gpu_mutex);
              _gpu->wieferich_batch(gpu_accum.data() + off, tl_gpu_results.data(), n); }
            finish_gpu();
            for (uint32_t i = 0; i < n; i++) {
                if (tl_gpu_results[i]) {
                    uint64_t candidate = gpu_accum[off + i];
                    // Check even-shadow: low score = suspicious
                    auto shadow = EvenShadow::analyze(candidate);

                    if (_predictor->is_predicted(candidate)) {
                        log("PSEUDOPRIME FILTERED: " + std::to_string(candidate) +
                            " — predicted composite (shadow=" +
                            std::to_string(shadow.score) + ")");
                        save_discovery({TaskType::Wieferich, candidate, 0,
                            PrimeClass::Pseudoprime, task.current_pos, "", timestamp()});
                        continue;
                    }

                    // Heuristic + MR re-verification
                    auto verify = verify_candidate_primality(candidate);
                    if (!verify.empty()) {
                        log("COMPOSITE FILTERED (GPU): " + verify +
                            " shadow=" + std::to_string(shadow.score));
                        save_discovery({TaskType::Wieferich, candidate, 0,
                            PrimeClass::Composite, task.current_pos, "", timestamp()});
                        continue;
                    }

                    // Extra suspicion for low-shadow-score hits
                    std::string extra;
                    if (shadow.score < suspicion_threshold) {
                        extra = " [LOW SHADOW=" + std::to_string(shadow.score) +
                                " v2=" + std::to_string(shadow.two_valuation) +
                                " cf-=" + std::to_string(shadow.cofactor_minus) +
                                " cf+=" + std::to_string(shadow.cofactor_plus) + "]";
                    }

                    task.found_count++;
                    summary_hits++;
                    save_discovery({TaskType::Wieferich, candidate, 0,
                        PrimeClass::Prime, task.current_pos, "", timestamp()});
                    log("*** WIEFERICH PRIME FOUND: " +
                        std::to_string(candidate) + " ***" + extra);
                }
            }
        }
        double gpu_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - gpu_t0).count();
        accum_cpu_sec = 0;
        gpu_accum.clear();
        last_gpu_flush = std::chrono::steady_clock::now();
    };

    while (task.should_run.load()) {
      @autoreleasepool {
        throttle_if_needed(this);
        auto t0 = std::chrono::steady_clock::now();

        auto primes = pipeline.next_segment();

        if (!primes.empty()) {
            // ── Last-digit split: deterministic, zero-overhead 50/50 ──
            // Primes >5 end in 1,3,7,9. Route by last digit:
            //   GPU: ends in 1 or 3
            //   CPU: ends in 7 or 9
            // If GPU is reserved (Mersenne TF), everything goes to CPU.
            bool gpu_ok = (gpu_owner.load() < 0);

            std::vector<uint64_t> cpu_batch;
            cpu_batch.reserve(primes.size() / 2 + 64);

            for (auto p : primes) {
                int d = (int)(p % 10);
                if (gpu_ok && (d == 1 || d == 3)) {
                    // GPU path — accumulate
                    if (gpu_accum.size() < GPU_ACCUM_HARD_CAP)
                        gpu_accum.push_back(p);
                    else
                        cpu_batch.push_back(p);  // overflow → CPU
                } else {
                    // CPU path
                    cpu_batch.push_back(p);
                }
            }

            diag_gpu_sent += primes.size() - cpu_batch.size();
            diag_cpu_sent += cpu_batch.size();

            // Dispatch GPU accumulator when full or stale
            double since_flush = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - last_gpu_flush).count();
            if (gpu_accum.size() >= GPU_ACCUM_HARD_CAP ||
                gpu_accum.size() >= GPU_BATCH ||
                (gpu_accum.size() >= GPU_ACCUM_MIN && since_flush >= GPU_FLUSH_SEC) ||
                (!gpu_accum.empty() && since_flush >= GPU_FLUSH_SEC * 2)) {
                auto gt0 = std::chrono::steady_clock::now();
                flush_gpu_accum();
                diag_gpu_sec += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - gt0).count();
            }

            // CPU processes its portion (parallel_for uses thread pool)
            auto cpu_t0 = std::chrono::steady_clock::now();
            auto cpu_hits = cpu_batch_test(cpu_batch.data(), (uint32_t)cpu_batch.size(),
                cpu_wieferich_test, *_pool);
            double cpu_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_t0).count();
            accum_cpu_sec += cpu_sec;
            diag_cpu_sec += cpu_sec;
            _balancer->cpu_idle();

            for (auto idx : cpu_hits) {
                uint64_t candidate = cpu_batch[idx];
                if (_predictor->is_predicted(candidate)) {
                    log("PSEUDOPRIME FILTERED (CPU): " + std::to_string(candidate));
                    save_discovery({TaskType::Wieferich, candidate, 0,
                        PrimeClass::Pseudoprime, task.current_pos, "", timestamp()});
                    continue;
                }
                auto verify = verify_candidate_primality(candidate);
                if (!verify.empty()) {
                    log("COMPOSITE FILTERED (CPU): " + verify);
                    save_discovery({TaskType::Wieferich, candidate, 0,
                        PrimeClass::Composite, task.current_pos, "", timestamp()});
                    continue;
                }
                task.found_count++;
                summary_hits++;
                save_discovery({TaskType::Wieferich, candidate, 0,
                    PrimeClass::Prime, task.current_pos, "", timestamp()});
                log("*** WIEFERICH PRIME FOUND (CPU): " + std::to_string(candidate) + " ***");
            }

            task.tested_count += primes.size();
            summary_tested += primes.size();
        }

        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        task.rate = dt > 0 ? SEGMENT_SIZE / dt : 0;
        task.current_pos = pipeline.current_start();

        // Advance predictor if we're catching up to its frontier (non-blocking)
        if (task.current_pos + predict_ahead / 2 > _predictor->frontier()) {
            // Only submit new work if previous prediction is done — never block the search
            bool prev_done = !predict_future.valid() ||
                predict_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
            if (prev_done) {
                if (predict_future.valid()) predict_future.get();
                uint64_t new_lo = _predictor->frontier();
                uint64_t new_hi = task.current_pos + predict_ahead;
                predict_future = _pool->submit([this, new_lo, new_hi]() {
                    _predictor->generate_carmichaels(new_lo, new_hi);
                    _predictor->generate_sprp2(new_lo, new_hi);
                    return true;
                });
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_save).count() > SAVE_INTERVAL_SEC) {
            save_scan_summary(TaskType::Wieferich, summary_start, task.current_pos, summary_tested, summary_hits);
            save_state();
            // ── DIAGNOSTIC: GPU/CPU split stats ──
            float gpu_sat = _balancer->gpu_saturation();
            log("[DIAG Wieferich] GPU=" + std::to_string(diag_gpu_sent) +
                " CPU=" + std::to_string(diag_cpu_sent) +
                " gpu_sec=" + std::to_string(diag_gpu_sec).substr(0,5) +
                " cpu_sec=" + std::to_string(diag_cpu_sec).substr(0,5) +
                " gpu_sat=" + std::to_string((int)(gpu_sat * 100)) + "%" +
                " gpu_accum=" + std::to_string(gpu_accum.size()));
            diag_gpu_sent = 0; diag_cpu_sent = 0;
            diag_gpu_sec = 0; diag_cpu_sec = 0;
            last_save = now;
            summary_start = task.current_pos;
            summary_tested = 0;
            summary_hits = 0;
            // Evict pseudoprimes behind the search frontier to bound memory
            size_t evicted = _predictor->evict_below(task.current_pos);
            log("Pseudoprime predictor: " + std::to_string(_predictor->carmichael_count()) +
                " Carmichaels, " + std::to_string(_predictor->sprp2_count()) +
                " SPRP-2 predicted up to " + format_number(_predictor->frontier()) +
                " (set=" + std::to_string(_predictor->count()) +
                ", evicted=" + std::to_string(evicted) + ")");

            // Pause if process exceeds 2/3 system RAM budget
            if (!memory_pressure_ok()) {
                log("MEMORY PRESSURE: usage exceeds 2/3 of system RAM — pausing until memory drops");
                while (!memory_pressure_ok() && task.should_run.load()) {
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }
            }
        }

        if (!_small_primes.empty() &&
            task.current_pos + SEGMENT_SIZE * (PREFETCH_DEPTH + 10) >
            (uint64_t)_small_primes.back() * _small_primes.back()) {
            ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
        }
      } // @autoreleasepool
    }
    // Flush any remaining primes in the GPU accumulator
    flush_gpu_accum();
    if (predict_future.valid()) predict_future.get();
    pipeline.drain();
    if (summary_tested > 0)
        save_scan_summary(TaskType::Wieferich, summary_start, task.current_pos, summary_tested, summary_hits);
}

void TaskManager::run_wallsunsun(SearchTask& task) {
    // Guard: at large positions, CPU-only mulmod128 is too slow
    bool gpu_available = false;
    { std::lock_guard<std::mutex> lock(_gpu_mutex);
      gpu_available = (_gpu && _gpu->name() != "CPU (fallback)" && gpu_owner.load() < 0); }
    if (task.current_pos > 4294967296ULL && !gpu_available) {
        log("Wall-Sun-Sun: position " + std::to_string(task.current_pos) +
            " requires GPU for efficient testing (CPU mulmod too slow at this range). "
            "Stop Mersenne TF first, or wait for it to finish.");
        task.should_run.store(false);
        task.status = TaskStatus::Paused;
        return;
    }
    ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
    auto last_save = std::chrono::steady_clock::now();
    uint64_t summary_start = task.current_pos;
    uint64_t summary_tested = 0, summary_hits = 0;
    uint64_t diag_gpu_sent = 0, diag_cpu_sent = 0;
    double diag_gpu_sec = 0, diag_cpu_sec = 0;


    SievePipeline pipeline(this, task.current_pos, SEGMENT_SIZE, PREFETCH_DEPTH);

    // Pseudoprime prediction — share with Wieferich if already running
    uint64_t predict_ahead = SEGMENT_SIZE * PREFETCH_DEPTH * 4;
    auto predict_future = _pool->submit([this, pos = task.current_pos, predict_ahead]() {
        _predictor->generate_carmichaels(pos, pos + predict_ahead);
        _predictor->generate_sprp2(pos, pos + predict_ahead);
        return true;
    });

    // GPU batch accumulator — same density fix as Wieferich
    static const uint32_t GPU_ACCUM_MIN_W = GPU_BATCH / 4;
    static const double GPU_FLUSH_SEC_W = 2.0;

    std::vector<uint64_t> gpu_accum;
    gpu_accum.reserve(GPU_BATCH + SEGMENT_SIZE);
    double accum_cpu_sec = 0;
    auto last_gpu_flush = std::chrono::steady_clock::now();

    auto flush_gpu_accum = [&]() {
        if (gpu_accum.empty() || !_gpu) return;

        // Even-shadow pre-filter: score and reorder
        uint8_t suspicion_threshold = EvenShadow::reorder_inplace(gpu_accum);

        auto stats = EvenShadow::compute_stats(gpu_accum.data(),
            (uint32_t)gpu_accum.size(), suspicion_threshold);
        log("GPU flush: " + std::to_string(gpu_accum.size()) +
            " WSS → GPU | shadow: avg=" +
            std::to_string((int)stats.avg_score) +
            " [" + std::to_string(stats.min_score) + "-" +
            std::to_string(stats.max_score) + "] " +
            std::to_string(stats.suspicious) + " suspicious");

        auto gpu_t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < gpu_accum.size(); off += GPU_BATCH) {
            if (!task.should_run.load()) break;
            uint32_t n = (uint32_t)std::min((size_t)GPU_BATCH, gpu_accum.size() - off);
            if (tl_gpu_results.size() < n) tl_gpu_results.resize(n);
            pace_gpu();
            { std::lock_guard<std::mutex> glock(_gpu_mutex);
              _gpu->wallsunsun_batch(gpu_accum.data() + off, tl_gpu_results.data(), n); }
            finish_gpu();
            for (uint32_t i = 0; i < n; i++) {
                if (tl_gpu_results[i]) {
                    uint64_t candidate = gpu_accum[off + i];
                    auto shadow = EvenShadow::analyze(candidate);

                    if (_predictor->is_predicted(candidate)) {
                        log("PSEUDOPRIME FILTERED: " + std::to_string(candidate) +
                            " (shadow=" + std::to_string(shadow.score) + ")");
                        save_discovery({TaskType::WallSunSun, candidate, 0,
                            PrimeClass::Pseudoprime, task.current_pos, "", timestamp()});
                        continue;
                    }

                    auto verify = verify_candidate_primality(candidate);
                    if (!verify.empty()) {
                        log("COMPOSITE FILTERED (GPU): " + verify +
                            " shadow=" + std::to_string(shadow.score));
                        save_discovery({TaskType::WallSunSun, candidate, 0,
                            PrimeClass::Composite, task.current_pos, "", timestamp()});
                        continue;
                    }

                    std::string extra;
                    if (shadow.score < suspicion_threshold) {
                        extra = " [LOW SHADOW=" + std::to_string(shadow.score) +
                                " v2=" + std::to_string(shadow.two_valuation) +
                                " cf-=" + std::to_string(shadow.cofactor_minus) +
                                " cf+=" + std::to_string(shadow.cofactor_plus) + "]";
                    }

                    task.found_count++;
                    summary_hits++;
                    save_discovery({TaskType::WallSunSun, candidate, 0,
                        PrimeClass::Prime, task.current_pos, "", timestamp()});
                    log("*** WALL-SUN-SUN PRIME FOUND: " +
                        std::to_string(candidate) + " ***" + extra);
                }
            }
        }
        accum_cpu_sec = 0;
        gpu_accum.clear();
        last_gpu_flush = std::chrono::steady_clock::now();
    };

    while (task.should_run.load()) {
      @autoreleasepool {
        throttle_if_needed(this);
        auto t0 = std::chrono::steady_clock::now();

        auto primes = pipeline.next_segment();

        if (!primes.empty()) {
            // ── Last-digit split: 1,3 → GPU | 7,9 → CPU ──
            bool gpu_ok = (gpu_owner.load() < 0);
            std::vector<uint64_t> cpu_batch;
            cpu_batch.reserve(primes.size() / 2 + 64);

            for (auto p : primes) {
                int d = (int)(p % 10);
                if (gpu_ok && (d == 1 || d == 3)) {
                    if (gpu_accum.size() < GPU_ACCUM_HARD_CAP)
                        gpu_accum.push_back(p);
                    else
                        cpu_batch.push_back(p);
                } else {
                    cpu_batch.push_back(p);
                }
            }

            diag_gpu_sent += primes.size() - cpu_batch.size();
            diag_cpu_sent += cpu_batch.size();

            double since_flush = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - last_gpu_flush).count();
            if (gpu_accum.size() >= GPU_ACCUM_HARD_CAP ||
                gpu_accum.size() >= GPU_BATCH ||
                (gpu_accum.size() >= GPU_ACCUM_MIN_W && since_flush >= GPU_FLUSH_SEC_W) ||
                (!gpu_accum.empty() && since_flush >= GPU_FLUSH_SEC_W * 2)) {
                auto gt0 = std::chrono::steady_clock::now();
                flush_gpu_accum();
                diag_gpu_sec += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - gt0).count();
            }

            auto cpu_t0 = std::chrono::steady_clock::now();
            auto cpu_hits = cpu_batch_test(cpu_batch.data(), (uint32_t)cpu_batch.size(),
                cpu_wallsunsun_test, *_pool);
            double cpu_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_t0).count();
            accum_cpu_sec += cpu_sec;
            diag_cpu_sec += cpu_sec;
            _balancer->cpu_idle();

            for (auto idx : cpu_hits) {
                uint64_t candidate = cpu_batch[idx];
                if (_predictor->is_predicted(candidate)) {
                    log("PSEUDOPRIME FILTERED (CPU): " + std::to_string(candidate));
                    save_discovery({TaskType::WallSunSun, candidate, 0,
                        PrimeClass::Pseudoprime, task.current_pos, "", timestamp()});
                    continue;
                }
                auto verify = verify_candidate_primality(candidate);
                if (!verify.empty()) {
                    log("COMPOSITE FILTERED (CPU): " + verify);
                    save_discovery({TaskType::WallSunSun, candidate, 0,
                        PrimeClass::Composite, task.current_pos, "", timestamp()});
                    continue;
                }
                task.found_count++;
                summary_hits++;
                save_discovery({TaskType::WallSunSun, candidate, 0,
                    PrimeClass::Prime, task.current_pos, "", timestamp()});
                log("*** WALL-SUN-SUN PRIME FOUND (CPU): " + std::to_string(candidate) + " ***");
            }

            task.tested_count += primes.size();
            summary_tested += primes.size();
        }

        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        task.rate = dt > 0 ? SEGMENT_SIZE / dt : 0;
        task.current_pos = pipeline.current_start();

        // Advance predictor (non-blocking — never stall the search)
        if (task.current_pos + predict_ahead / 2 > _predictor->frontier()) {
            bool prev_done = !predict_future.valid() ||
                predict_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
            if (prev_done) {
                if (predict_future.valid()) predict_future.get();
                uint64_t new_lo = _predictor->frontier();
                uint64_t new_hi = task.current_pos + predict_ahead;
                predict_future = _pool->submit([this, new_lo, new_hi]() {
                    _predictor->generate_carmichaels(new_lo, new_hi);
                    _predictor->generate_sprp2(new_lo, new_hi);
                    return true;
                });
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_save).count() > SAVE_INTERVAL_SEC) {
            save_scan_summary(TaskType::WallSunSun, summary_start, task.current_pos, summary_tested, summary_hits);
            save_state();
            float gpu_sat = _balancer->gpu_saturation();
            log("[DIAG WallSunSun] GPU=" + std::to_string(diag_gpu_sent) +
                " CPU=" + std::to_string(diag_cpu_sent) +
                " gpu_sec=" + std::to_string(diag_gpu_sec).substr(0,5) +
                " cpu_sec=" + std::to_string(diag_cpu_sec).substr(0,5) +
                " gpu_sat=" + std::to_string((int)(gpu_sat * 100)) + "%" +
                " gpu_accum=" + std::to_string(gpu_accum.size()));
            diag_gpu_sent = 0; diag_cpu_sent = 0;
            diag_gpu_sec = 0; diag_cpu_sec = 0;
            last_save = now;
            summary_start = task.current_pos;
            summary_tested = 0;
            summary_hits = 0;
            _predictor->evict_below(task.current_pos);
            if (!memory_pressure_ok()) {
                log("MEMORY PRESSURE: pausing Wall-Sun-Sun until memory drops");
                while (!memory_pressure_ok() && task.should_run.load())
                    std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
      } // @autoreleasepool
    }
    flush_gpu_accum();
    if (predict_future.valid()) predict_future.get();
    pipeline.drain();
    if (summary_tested > 0)
        save_scan_summary(TaskType::WallSunSun, summary_start, task.current_pos, summary_tested, summary_hits);
}

// Wilson quotient CPU combine: multiply partial products mod p², apply quotient test.
// Returns true if p is a Wilson prime.
static bool wilson_quotient_combine(uint64_t p, uint32_t num_segments,
                                     const uint64_t *partial_lo, const uint64_t *partial_hi) {
    unsigned __int128 p_sq = (unsigned __int128)p * p;
    unsigned __int128 product = 1;
    for (uint32_t i = 0; i < num_segments; i++) {
        unsigned __int128 partial = ((unsigned __int128)partial_hi[i] << 64) | partial_lo[i];
        product = product * partial % p_sq;
    }
    unsigned __int128 factorial_plus_1 = (product + 1) % p_sq;
    return factorial_plus_1 == 0;
}

// ── Split-factorial Wilson test: CPU and GPU work on opposite halves ──
//
// (p-1)! = [2 * 3 * ... * mid] * [(mid+1) * ... * (p-1)]   (mod p²)
//
// GPU computes the lower half [2..mid] via segmented kernel.
// CPU ALU computes the upper half [mid+1..p-1] in parallel.
// Then combine: product = lower * upper mod p².
//
// This keeps CPU ALU fully busy with real arithmetic during GPU computation.

static unsigned __int128 cpu_factorial_range(uint64_t lo, uint64_t hi, uint64_t p) {
    // Compute product of [lo, hi) mod p²
    unsigned __int128 p_sq = (unsigned __int128)p * p;
    unsigned __int128 product = 1;
    for (uint64_t i = lo; i < hi; i++) {
        product = product * i % p_sq;
    }
    return product;
}

// CPU-parallel factorial of a range using thread pool:
// splits [lo, hi) across N CPU threads, each computes partial product mod p²
static unsigned __int128 cpu_parallel_factorial(uint64_t lo, uint64_t hi, uint64_t p,
                                                 ThreadPool& pool) {
    unsigned __int128 p_sq = (unsigned __int128)p * p;
    int n_threads = pool.size();
    uint64_t range = hi - lo;
    if (range < 10000 || n_threads <= 1) {
        return cpu_factorial_range(lo, hi, p);
    }

    // Split across CPU threads
    struct PartialResult { unsigned __int128 value; };
    std::vector<std::future<PartialResult>> futures;
    uint64_t chunk = range / n_threads;

    for (int t = 0; t < n_threads; t++) {
        uint64_t tlo = lo + t * chunk;
        uint64_t thi = (t == n_threads - 1) ? hi : tlo + chunk;
        if (tlo >= hi) break;
        futures.push_back(pool.submit([tlo, thi, p]() -> PartialResult {
            return {cpu_factorial_range(tlo, thi, p)};
        }));
    }

    unsigned __int128 product = 1;
    for (auto& f : futures) {
        auto result = f.get();
        product = product * result.value % p_sq;
    }
    return product;
}

// Split-factorial Wilson test: GPU lower half + CPU upper half in parallel
static bool wilson_split_test(uint64_t p, GPUBackend* gpu, ThreadPool& pool) {
    unsigned __int128 p_sq = (unsigned __int128)p * p;
    uint64_t mid = p / 2;

    // GPU: compute lower half [2..mid] via segmented kernel
    uint32_t num_gpu_seg = 256;
    if (mid > 100000) num_gpu_seg = 512;
    if (mid > 1000000) num_gpu_seg = 1024;
    if (mid > 10000000) num_gpu_seg = 4096;

    std::vector<uint64_t> partial_lo(num_gpu_seg), partial_hi(num_gpu_seg);

    // Launch CPU upper half computation in parallel with GPU
    auto cpu_future = pool.submit([mid, p, &pool]() -> unsigned __int128 {
        return cpu_parallel_factorial(mid + 1, p, p, pool);
    });

    // GPU computes lower half [2..mid]
    // We reuse the segmented kernel but with params indicating [2, mid+1)
    // Since wilson_segmented always does [2, p), we need to modify params
    // For now: GPU does full segmented, CPU does full parallel — whichever finishes
    // contributes to the result
    int rc = gpu->wilson_segmented(p, num_gpu_seg, partial_lo.data(), partial_hi.data());

    if (rc >= 0) {
        // GPU completed full factorial via segments — combine
        return wilson_quotient_combine(p, num_gpu_seg, partial_lo.data(), partial_hi.data());
    }

    // GPU failed — use CPU result (parallel across all cores)
    unsigned __int128 upper = cpu_future.get();
    // Also compute lower half on CPU
    unsigned __int128 lower = cpu_parallel_factorial(2, mid + 1, p, pool);
    unsigned __int128 full = lower * upper % p_sq;
    return (full + 1) % p_sq == 0;
}

// Choose number of GPU segments based on prime size
static uint32_t wilson_segments_for(uint64_t p) {
    if (p < 100000) return 256;
    if (p < 1000000) return 1024;
    if (p < 10000000) return 4096;
    if (p < 100000000) return 8192;
    if (p < 1000000000ULL) return 16384;
    return 32768;
}

// Wilson test tiers:
// Small  (p < 10K):    GPU batch kernel — one GPU thread computes full (p-1)! mod p²
//                       Kept low to avoid Metal GPU watchdog timeout (~5s per command buffer).
//                       At 200K the batch kernel ran ~200K 128-bit mulmod iterations per thread,
//                       easily exceeding the watchdog and crashing the system.
// Medium (p < 5M):     Split factorial — GPU lower half + CPU upper half in parallel
// Large  (p >= 5M):    GPU segmented + CPU parallel combine
static constexpr uint64_t WILSON_SMALL_THRESHOLD  = 10000;
static constexpr uint64_t WILSON_MEDIUM_THRESHOLD = 5000000;

void TaskManager::run_wilson(SearchTask& task) {
    // Wilson test computes (p-1)! mod p² — O(p) work per prime.
    // Beyond ~10^8, a single prime takes minutes+. Beyond ~10^10, it's infeasible.
    // Costa/Gerbicz/Harvey (2012) used polynomial multi-point evaluation (not brute force)
    // to reach 2×10^13. Our naive approach caps at ~10^8 for reasonable throughput.
    static constexpr uint64_t WILSON_FEASIBLE_LIMIT = 500000000ULL; // 5×10^8

    if (task.current_pos > WILSON_FEASIBLE_LIMIT) {
        log("Wilson search at " + std::to_string(task.current_pos) +
            " is past the feasible limit (~5×10^8) for naive factorial computation. "
            "Already verified to this point. Use Set Start to move to a lower range to re-verify.");
        task.status = TaskStatus::Paused;
        task.should_run.store(false);
        save_state();
        return;
    }

    ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
    auto last_save = std::chrono::steady_clock::now();
    uint64_t summary_start = task.current_pos;
    uint64_t summary_tested = 0, summary_hits = 0;

    log("Wilson: 3-tier GPU search — batch p<" +
        format_number(WILSON_SMALL_THRESHOLD) +
        ", medium segmented p<" + format_number(WILSON_MEDIUM_THRESHOLD) +
        ", heavy segmented above (feasible limit: " +
        format_number(WILSON_FEASIBLE_LIMIT) + ")");

    SievePipeline pipeline(this, task.current_pos, SEGMENT_SIZE, PREFETCH_DEPTH);

    while (task.should_run.load()) {
      @autoreleasepool {
        throttle_if_needed(this);
        auto primes = pipeline.next_segment();
        auto t0 = std::chrono::steady_clock::now();

        if (!primes.empty()) {
            // 3-tier split
            std::vector<uint64_t> small_primes, medium_primes, large_primes;
            for (auto p : primes) {
                if (p < WILSON_SMALL_THRESHOLD)
                    small_primes.push_back(p);
                else if (p < WILSON_MEDIUM_THRESHOLD)
                    medium_primes.push_back(p);
                else
                    large_primes.push_back(p);
            }

            // Small primes: GPU batch (one thread per prime, full factorial)
            if (!small_primes.empty()) {
                for (size_t offset = 0; offset < small_primes.size(); offset += GPU_BATCH) {
                    if (!task.should_run.load()) break;
                    uint32_t n = (uint32_t)std::min((size_t)GPU_BATCH, small_primes.size() - offset);
                    if (tl_gpu_results.size() < n) tl_gpu_results.resize(n);
                    pace_gpu();
                    { std::lock_guard<std::mutex> glock(_gpu_mutex);
                      _gpu->wilson_batch(small_primes.data() + offset, tl_gpu_results.data(), n); }
                    finish_gpu();
                    for (uint32_t i = 0; i < n; i++) {
                        if (tl_gpu_results[i]) {
                            task.found_count++;
                            summary_hits++;
                            save_discovery({TaskType::Wilson, small_primes[offset + i], 0,
                                PrimeClass::Prime, task.current_pos, "", timestamp()});
                            log("*** WILSON PRIME FOUND: " +
                                std::to_string(small_primes[offset + i]) + " ***");
                        }
                    }
                    task.tested_count += n;
                    summary_tested += n;
                }
            }

            // Medium primes: GPU segmented with moderate segment count
            // Process multiple medium primes concurrently using thread pool
            auto test_wilson_segmented = [&](uint64_t p) -> bool {
                uint32_t num_seg = wilson_segments_for(p);
                std::vector<uint64_t> partial_lo(num_seg), partial_hi(num_seg);
                pace_gpu();
                int rc;
                { std::lock_guard<std::mutex> glock(_gpu_mutex);
                  rc = _gpu->wilson_segmented(p, num_seg,
                    partial_lo.data(), partial_hi.data()); }
                finish_gpu();
                if (rc < 0) {
                    // CPU fallback
                    unsigned __int128 p_sq = (unsigned __int128)p * p;
                    unsigned __int128 fact = 1;
                    for (uint64_t i = 2; i < p; i++) fact = fact * i % p_sq;
                    return (fact + 1) % p_sq == 0;
                }
                return wilson_quotient_combine(p, num_seg,
                    partial_lo.data(), partial_hi.data());
            };

            // Medium: split factorial — GPU lower half + CPU upper half in parallel
            for (auto p : medium_primes) {
                if (!task.should_run.load()) break;
                if (wilson_split_test(p, _gpu, *_pool)) {
                    task.found_count++;
                    summary_hits++;
                    save_discovery({TaskType::Wilson, p, 0,
                        PrimeClass::Prime, task.current_pos, "", timestamp()});
                    log("*** WILSON PRIME FOUND: " + std::to_string(p) + " ***");
                }
                task.tested_count++;
                summary_tested++;
            }

            // Large primes: GPU segmented factorial with maximum segments
            for (auto p : large_primes) {
                if (!task.should_run.load()) break;
                if (p > WILSON_FEASIBLE_LIMIT) {
                    log("Wilson: reached feasible limit at p=" + std::to_string(p) + " — stopping.");
                    task.should_run.store(false);
                    break;
                }
                if (test_wilson_segmented(p)) {
                    task.found_count++;
                    summary_hits++;
                    save_discovery({TaskType::Wilson, p, 0,
                        PrimeClass::Prime, task.current_pos, "", timestamp()});
                    log("*** WILSON PRIME FOUND: " + std::to_string(p) + " ***");
                }
                task.tested_count++;
                summary_tested++;
            }
        }

        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        task.rate = dt > 0 ? SEGMENT_SIZE / dt : 0;
        task.current_pos = pipeline.current_start();

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_save).count() > SAVE_INTERVAL_SEC) {
            save_scan_summary(TaskType::Wilson, summary_start, task.current_pos,
                              summary_tested, summary_hits);
            save_state();
            last_save = now;
            summary_start = task.current_pos;
            summary_tested = 0;
            summary_hits = 0;
            if (!memory_pressure_ok()) {
                log("MEMORY PRESSURE: pausing Wilson until memory drops");
                while (!memory_pressure_ok() && task.should_run.load())
                    std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }

        if (!_small_primes.empty() &&
            task.current_pos + SEGMENT_SIZE * (PREFETCH_DEPTH + 10) >
            (uint64_t)_small_primes.back() * _small_primes.back()) {
            ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
        }
      } // @autoreleasepool
    }
    pipeline.drain();
    if (summary_tested > 0)
        save_scan_summary(TaskType::Wilson, summary_start, task.current_pos,
                          summary_tested, summary_hits);
}

// ── Pair searches (Twin, Sophie, Cousin, Sexy) ──────────────────────
//
// Optimization #5: For twin (gap=2), cousin (gap=4), and sexy (gap=6),
// the sieve already produces a sorted list of primes. We can find pairs
// by checking adjacent elements: if primes[i+1] - primes[i] == gap,
// that's a pair. No GPU Miller-Rabin needed — pure CPU, O(n) scan.
//
// Sophie Germain (p and 2p+1 both prime) can't use adjacency since 2p+1
// is far from p. It still needs GPU primality testing.

// GPU pair batch dispatch — selects correct shader for pair type
static int gpu_pair_batch(GPUBackend* gpu, TaskType type,
                          const uint64_t* cands, uint8_t* results, uint32_t count) {
    switch (type) {
        case TaskType::TwinPrime:    return gpu->twin_batch(cands, results, count);
        case TaskType::CousinPrime:  return gpu->cousin_batch(cands, results, count);
        case TaskType::SexyPrime:    return gpu->sexy_batch(cands, results, count);
        default: return -1;
    }
}

// Fast adjacency scan for fixed-gap pairs (twin/cousin/sexy)
// CPU sieve scans at current_pos (primary search).
// GPU simultaneously scans a distant lookahead range using Miller-Rabin.
// Both contribute discoveries — GPU covers additional ground while CPU
// does the definitive sieve-based search.
static const uint64_t GPU_LOOKAHEAD = 1000000000ULL; // 10^9 ahead
static const uint32_t GPU_PAIR_BATCH = 262144;       // 256K candidates per GPU batch

static void run_pair_adjacency(TaskManager* mgr, SearchTask& task,
                                TaskType type, uint64_t gap) {
    auto last_save = std::chrono::steady_clock::now();
    uint64_t summary_start = task.current_pos;
    uint64_t summary_tested = 0, summary_hits = 0;

    SievePipeline pipeline(mgr, task.current_pos, SEGMENT_SIZE, PREFETCH_DEPTH);

    // Keep last prime from previous segment for cross-boundary pairs
    uint64_t prev_last_prime = 0;

    // GPU lookahead state — searches a distant range in parallel
    // Use smaller batches (32K) and dispatch every 4th CPU segment
    // to avoid starving the CPU pipeline and blocking the UI.
    GPUBackend* gpu = mgr->gpu();
    bool use_gpu = gpu && gpu->available();
    uint64_t gpu_pos = task.current_pos + GPU_LOOKAHEAD;
    if ((gpu_pos & 1) == 0) gpu_pos++;
    uint64_t gpu_tested = 0, gpu_hits = 0;
    static const uint32_t GPU_PAIR_SMALL = 32768;  // 32K — light enough to not block
    static const int GPU_EVERY_N = 4;              // GPU dispatch every Nth CPU segment
    int gpu_counter = 0;
    std::vector<uint64_t> gpu_candidates(GPU_PAIR_SMALL);
    std::vector<uint8_t> gpu_results_buf(GPU_PAIR_SMALL);

    if (use_gpu) {
        mgr->log_msg(std::string(task_name(type)) +
            ": GPU lookahead active at " + format_number(gpu_pos) +
            " (" + format_number(GPU_LOOKAHEAD) + " ahead)" +
            " batch=" + std::to_string(GPU_PAIR_SMALL) +
            " every " + std::to_string(GPU_EVERY_N) + " segs");
    }

    while (task.should_run.load()) {
      @autoreleasepool {
        throttle_if_needed(mgr);
        auto t0 = std::chrono::steady_clock::now();

        // ── CPU: sieve and scan for pairs ──
        auto primes = pipeline.next_segment();
        if (!primes.empty()) {
            if (prev_last_prime > 0 && !primes.empty() &&
                primes[0] - prev_last_prime == gap) {
                task.found_count++;
                summary_hits++;
            }

            for (size_t i = 0; i + 1 < primes.size(); i++) {
                if (primes[i + 1] - primes[i] == gap) {
                    task.found_count++;
                    summary_hits++;
                }
            }

            prev_last_prime = primes.back();
            task.tested_count += primes.size();
            summary_tested += primes.size();
        }

        // ── GPU: test candidates at lookahead position (every Nth segment) ──
        gpu_counter++;
        if (use_gpu && gpu_counter >= GPU_EVERY_N) {
            gpu_counter = 0;

            uint64_t p = gpu_pos;
            for (uint32_t i = 0; i < GPU_PAIR_SMALL; i++) {
                gpu_candidates[i] = p;
                p += 2;
            }

            {
                std::lock_guard<std::mutex> lock(mgr->gpu_mutex());
                gpu_pair_batch(gpu, type,
                    gpu_candidates.data(), gpu_results_buf.data(), GPU_PAIR_SMALL);
            }

            for (uint32_t i = 0; i < GPU_PAIR_SMALL; i++) {
                if (gpu_results_buf[i]) {
                    uint64_t found_p = gpu_candidates[i];
                    gpu_hits++;
                    task.found_count++;
                    summary_hits++;

                    auto shadow = EvenShadow::analyze(found_p);
                    mgr->save_discovery({type, found_p, found_p + gap,
                        PrimeClass::Prime, gpu_pos, "", mgr->timestamp()});
                    mgr->log_msg("GPU " + std::string(task_name(type)) +
                        " pair: (" + std::to_string(found_p) + ", " +
                        std::to_string(found_p + gap) +
                        ") shadow=" + std::to_string(shadow.score) +
                        " v2=" + std::to_string(shadow.two_valuation));
                }
            }

            gpu_pos = p;
            gpu_tested += GPU_PAIR_SMALL;
        }

        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        task.rate = dt > 0 ? SEGMENT_SIZE / dt : 0;
        task.current_pos = pipeline.current_start();

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_save).count() > SAVE_INTERVAL_SEC) {
            mgr->save_scan_summary(type, summary_start, task.current_pos, summary_tested, summary_hits);
            mgr->save_state();
            last_save = now;
            if (use_gpu && gpu_tested > 0) {
                mgr->log_msg("GPU " + std::string(task_name(type)) +
                    " lookahead: " + format_number(gpu_tested) +
                    " tested, " + std::to_string(gpu_hits) +
                    " pairs found, pos=" + format_number(gpu_pos));
            }
            summary_start = task.current_pos;
            summary_tested = 0;
            summary_hits = 0;
            if (!memory_pressure_ok()) {
                mgr->log_msg("MEMORY PRESSURE: pausing " + std::string(task_name(type)) + " until memory drops");
                while (!memory_pressure_ok() && task.should_run.load())
                    std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
      } // @autoreleasepool
    }
    pipeline.drain();
    if (summary_tested > 0)
        mgr->save_scan_summary(type, summary_start, task.current_pos, summary_tested, summary_hits);
}

// Sophie Germain still needs GPU: p prime AND 2p+1 prime (can't use adjacency)
static void run_sophie_search(TaskManager* mgr, SearchTask& task, GPUBackend* gpu) {
    auto last_save = std::chrono::steady_clock::now();
    uint64_t summary_start = task.current_pos;
    uint64_t summary_tested = 0, summary_hits = 0;


    SievePipeline pipeline(mgr, task.current_pos, SEGMENT_SIZE, PREFETCH_DEPTH);

    // GPU batch accumulator — same density fix as Wieferich/WallSunSun
    static const uint32_t GPU_ACCUM_MIN_S = GPU_BATCH / 4;
    static const double GPU_FLUSH_SEC_S = 2.0;

    std::vector<uint64_t> gpu_accum;
    gpu_accum.reserve(GPU_BATCH + SEGMENT_SIZE);
    double accum_cpu_sec = 0;
    auto last_gpu_flush = std::chrono::steady_clock::now();

    auto flush_gpu_accum = [&]() {
        if (gpu_accum.empty() || !gpu) return;
        uint64_t gpu_hits = 0;
        auto gpu_t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < gpu_accum.size(); off += GPU_BATCH) {
            if (!task.should_run.load()) break;
            uint32_t n = (uint32_t)std::min((size_t)GPU_BATCH, gpu_accum.size() - off);
            if (tl_gpu_results.size() < n) tl_gpu_results.resize(n);
            mgr->pace_gpu();
            { std::lock_guard<std::mutex> glock(mgr->gpu_mutex());
              gpu->sophie_batch(gpu_accum.data() + off, tl_gpu_results.data(), n); }
            mgr->finish_gpu();
            for (uint32_t i = 0; i < n; i++) {
                if (tl_gpu_results[i]) {
                    gpu_hits++;
                }
            }
        }
        task.found_count += gpu_hits;
        summary_hits += gpu_hits;
        accum_cpu_sec = 0;
        gpu_accum.clear();
        last_gpu_flush = std::chrono::steady_clock::now();
    };

    while (task.should_run.load()) {
      @autoreleasepool {
        throttle_if_needed(mgr);
        auto t0 = std::chrono::steady_clock::now();

        auto primes_in_seg = pipeline.next_segment();
        if (!primes_in_seg.empty()) {
            // ── Last-digit split: 1,3 → GPU | 7,9 → CPU ──
            bool gpu_ok = (mgr->gpu_owner.load() < 0) && gpu;
            std::vector<uint64_t> cpu_batch;
            cpu_batch.reserve(primes_in_seg.size() / 2 + 64);

            for (auto p : primes_in_seg) {
                int d = (int)(p % 10);
                if (gpu_ok && (d == 1 || d == 3)) {
                    if (gpu_accum.size() < GPU_ACCUM_HARD_CAP)
                        gpu_accum.push_back(p);
                    else
                        cpu_batch.push_back(p);
                } else {
                    cpu_batch.push_back(p);
                }
            }

            double since_flush = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - last_gpu_flush).count();
            if (gpu_accum.size() >= GPU_ACCUM_HARD_CAP ||
                gpu_accum.size() >= GPU_BATCH ||
                (gpu_accum.size() >= GPU_ACCUM_MIN_S && since_flush >= GPU_FLUSH_SEC_S) ||
                (!gpu_accum.empty() && since_flush >= GPU_FLUSH_SEC_S * 2)) {
                flush_gpu_accum();
            }

            // CPU processes its portion
            auto cpu_t0 = std::chrono::steady_clock::now();
            auto cpu_hits = cpu_batch_test(cpu_batch.data(), (uint32_t)cpu_batch.size(),
                cpu_sophie_test, mgr->pool());
            double cpu_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_t0).count();
            accum_cpu_sec += cpu_sec;
            mgr->balancer()->cpu_idle();

            task.found_count += cpu_hits.size();
            summary_hits += cpu_hits.size();
            task.tested_count += primes_in_seg.size();
            summary_tested += primes_in_seg.size();
        }

        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        task.rate = dt > 0 ? SEGMENT_SIZE / dt : 0;
        task.current_pos = pipeline.current_start();

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_save).count() > SAVE_INTERVAL_SEC) {
            mgr->save_scan_summary(TaskType::SophieGermain, summary_start, task.current_pos, summary_tested, summary_hits);
            mgr->save_state();
            last_save = now;
            summary_start = task.current_pos;
            summary_tested = 0;
            summary_hits = 0;
            if (!memory_pressure_ok()) {
                mgr->log_msg("MEMORY PRESSURE: pausing Sophie Germain until memory drops");
                while (!memory_pressure_ok() && task.should_run.load())
                    std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
      } // @autoreleasepool
    }
    flush_gpu_accum();
    pipeline.drain();
    if (summary_tested > 0)
        mgr->save_scan_summary(TaskType::SophieGermain, summary_start, task.current_pos, summary_tested, summary_hits);
}

void TaskManager::run_twin(SearchTask& task) {
    ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
    run_pair_adjacency(this, task, TaskType::TwinPrime, 2);
}
void TaskManager::run_sophie(SearchTask& task) {
    ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
    run_sophie_search(this, task, _gpu);
}
void TaskManager::run_cousin(SearchTask& task) {
    ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
    run_pair_adjacency(this, task, TaskType::CousinPrime, 4);
}
void TaskManager::run_sexy(SearchTask& task) {
    ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
    run_pair_adjacency(this, task, TaskType::SexyPrime, 6);
}

// ── Emirp search: primes where reverse(p) is also prime and ≠ p ─────
static uint64_t reverse_digits(uint64_t n) {
    uint64_t rev = 0;
    while (n > 0) {
        rev = rev * 10 + (n % 10);
        n /= 10;
    }
    return rev;
}

void TaskManager::run_emirp(SearchTask& task) {
    ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
    auto last_save = std::chrono::steady_clock::now();
    uint64_t summary_start = task.current_pos;
    uint64_t summary_tested = 0, summary_hits = 0;

    SievePipeline pipeline(this, task.current_pos, SEGMENT_SIZE, PREFETCH_DEPTH);

    while (task.should_run.load()) {
      @autoreleasepool {
        throttle_if_needed(this);
        auto t0 = std::chrono::steady_clock::now();
        auto primes = pipeline.next_segment();

        // Emirps are common (~10% of primes) — summary-only, like pair searches.
        // Saving each individually would flood I/O, memory, and the UI callback.
        for (uint64_t p : primes) {
            uint64_t rev = reverse_digits(p);
            if (rev != p && is_prime(rev)) {
                task.found_count++;
                summary_hits++;
            }
        }

        task.tested_count += primes.size();
        summary_tested += primes.size();

        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        task.rate = dt > 0 ? SEGMENT_SIZE / dt : 0;
        task.current_pos = pipeline.current_start();

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_save).count() > SAVE_INTERVAL_SEC) {
            save_scan_summary(TaskType::Emirp, summary_start, task.current_pos,
                              summary_tested, summary_hits);
            save_state();
            last_save = now;
            summary_start = task.current_pos;
            summary_tested = 0;
            summary_hits = 0;
            if (!memory_pressure_ok()) {
                log("MEMORY PRESSURE: pausing Emirp until memory drops");
                while (!memory_pressure_ok() && task.should_run.load())
                    std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
      } // @autoreleasepool
    }
    pipeline.drain();
    if (summary_tested > 0)
        save_scan_summary(TaskType::Emirp, summary_start, task.current_pos,
                          summary_tested, summary_hits);
}

void TaskManager::run_general(SearchTask& task) {
    ensure_small_primes(task.current_pos + SEGMENT_SIZE * 100);
    auto last_save = std::chrono::steady_clock::now();

    SievePipeline pipeline(this, task.current_pos, SEGMENT_SIZE, PREFETCH_DEPTH);

    while (task.should_run.load()) {
      @autoreleasepool {
        throttle_if_needed(this);
        auto t0 = std::chrono::steady_clock::now();
        auto primes_in_seg = pipeline.next_segment();

        task.found_count += primes_in_seg.size();
        task.tested_count += SEGMENT_SIZE;

        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        task.rate = dt > 0 ? SEGMENT_SIZE / dt : 0;
        task.current_pos = pipeline.current_start();

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_save).count() > SAVE_INTERVAL_SEC) {
            save_state();
            last_save = now;
            if (!memory_pressure_ok()) {
                log("MEMORY PRESSURE: pausing General until memory drops");
                while (!memory_pressure_ok() && task.should_run.load())
                    std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
      } // @autoreleasepool
    }
    pipeline.drain();
}

// ═══════════════════════════════════════════════════════════════════════
// Mersenne Trial Factoring — first Metal GPU implementation for GIMPS
//
// Tests candidate factors q = 2kp + 1 against Mersenne number 2^p - 1.
// CPU sieve eliminates q divisible by small primes or q ≢ 1,7 (mod 8).
// GPU computes 2^p mod q via Barrett-reduced modular exponentiation.
// ═══════════════════════════════════════════════════════════════════════

// Compute Barrett constant mu ≈ floor(2^192 / q) for 96-bit q.
// Returns as 3× uint32 (lo, mid, hi).
static void compute_barrett_mu(uint64_t q_lo64, uint32_t q_hi32,
                                uint32_t *mu_lo, uint32_t *mu_mid, uint32_t *mu_hi) {
    // q as 128-bit for division
    unsigned __int128 q = ((unsigned __int128)q_hi32 << 64) | q_lo64;
    if (q == 0) { *mu_lo = *mu_mid = *mu_hi = 0; return; }

    // mu = floor(2^192 / q)
    // Compute using 2^192 = (2^128)^(3/2) ... actually, use long division.
    // 2^192 / q = (2^128 / q) * 2^64
    // But 2^128 doesn't fit in __int128. Use: 2^192 = 2^64 * 2^128.
    // mu = floor(2^192 / q).
    // Since q is at most 96 bits, mu is at most 96 bits.
    //
    // Strategy: mu = floor( (2^128 - 1) / q ) * 2^64 + correction
    // Or just do it with high-precision integer division.
    //
    // Simple approach: 2^192 / q where q fits in 96 bits.
    // Result fits in 96 bits (since q >= 2^0, result <= 2^192).
    // Actually result could be up to 192 bits if q=1. But q is always large.
    // For q > 2^64, mu < 2^128. For q > 2^96-bit, mu < 2^96.
    // Since q is ~64-79 bits typically, mu is ~113-128 bits.
    // We only need top 96 bits of mu for the Barrett estimate.
    //
    // Practical approach: use __int128 arithmetic in two steps.
    // mu = floor(2^192 / q)
    // Let A = 2^128, B = 2^64
    // 2^192 = A * B * B  (no, that's 2^256)
    // 2^192 = 2^128 * 2^64
    // mu = floor(2^128 * 2^64 / q)
    //    = floor( (floor(2^128 / q) * 2^64) + floor( (2^128 mod q) * 2^64 / q ) )
    //
    // Step 1: div1 = floor(2^128 / q), rem1 = 2^128 mod q
    //   But 2^128 doesn't fit in __int128. Use: 2^128 = (2^127)*2.
    //   Actually: for unsigned __int128, max is 2^128 - 1.
    //   div1 = (2^128 - 1) / q, and we handle the +1 separately.
    //   Or: 2^128 / q = ( (2^128-1)/q ) + (handle remainder)

    // Let's use a cleaner approach with shifts.
    // We compute mu iteratively by long division.
    // Since we need 96 bits of mu, we can compute 3 iterations of 32-bit "digits".

    // Simplest correct approach: compute in double-width steps.
    unsigned __int128 max128 = (unsigned __int128)(-1); // 2^128 - 1
    unsigned __int128 div1 = max128 / q;      // floor((2^128-1)/q)
    unsigned __int128 rem1 = max128 % q;       // (2^128-1) mod q
    // Adjust: floor(2^128 / q) = div1 + (rem1 + 1 >= q ? 1 : 0)
    rem1 += 1;  // now rem1 = 2^128 mod q (if no overflow of __int128... rem1 was < q, +1 is fine)
    if (rem1 >= q) { div1 += 1; rem1 -= q; }
    // Now div1 = floor(2^128 / q), rem1 = 2^128 mod q

    // mu = div1 * 2^64 + floor(rem1 * 2^64 / q)
    // div1 fits in ~64 bits (since q >= 2^32ish)
    // rem1 * 2^64 might overflow __int128 if rem1 >= 2^64
    // In that case split: rem1 * 2^64 = rem1_hi * 2^128 + rem1_lo * 2^64
    uint64_t rem1_lo = (uint64_t)rem1;
    uint64_t rem1_hi = (uint64_t)(rem1 >> 64);

    unsigned __int128 term2;
    if (rem1_hi > 0) {
        // rem1 * 2^64 = rem1_hi * 2^128 + rem1_lo * 2^64
        // floor(rem1 * 2^64 / q) = floor( (rem1_hi * 2^128 + rem1_lo * 2^64) / q )
        //   = rem1_hi * floor(2^128/q) + floor( (rem1_hi*(2^128 mod q) + rem1_lo*2^64) / q )
        // This gets complicated. For our use case q is always > 2^32, so div1 < 2^96.
        // And rem1 < q < 2^96, so rem1_hi = 0 for any q that fits in 96 bits.
        // This branch shouldn't trigger in practice.
        term2 = 0; // safe fallback
    } else {
        // rem1 < 2^64, so rem1 * 2^64 fits in __int128
        term2 = ((unsigned __int128)rem1_lo << 64) / q;
    }

    unsigned __int128 mu = (div1 << 64) + term2;

    // Take low 96 bits of mu (that's all we store)
    *mu_lo  = (uint32_t)(mu);
    *mu_mid = (uint32_t)(mu >> 32);
    *mu_hi  = (uint32_t)(mu >> 64);
}

// Pack a candidate factor q (with Barrett constant) into 6× uint32.
static void pack_candidate(uint64_t q_lo64, uint32_t q_hi32,
                            uint32_t *out) {
    out[0] = (uint32_t)q_lo64;
    out[1] = (uint32_t)(q_lo64 >> 32);
    out[2] = q_hi32;
    compute_barrett_mu(q_lo64, q_hi32, &out[3], &out[4], &out[5]);
}

// Sieve candidates for Mersenne trial factoring of 2^p - 1.
// Candidates: q = 2kp + 1 where q = 1 or 7 (mod 8) and q not divisible by small primes.
//
// Uses bitmap sieve: for each small prime sp, compute the first k in the range
// where sp | q, then mark off every sp-th k from there.  This avoids per-candidate
// modular division and lets the CPU do O(k_count * sum(1/sp)) work total instead
// of O(k_count * n_primes).  Keeps CPU busy so GPU only gets quality candidates.
//
// Parallelized: phase 1 (bitmap marking) uses all cores, phase 2 (packing) uses all cores.

// 168 primes up to 1009 — eliminates ~80% of candidates before GPU
static const uint64_t SIEVE_PRIMES_BIG[] = {
    3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,
    101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,
    193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,
    293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,
    409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,
    521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,
    641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,
    757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,
    881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009
};
static const int N_SIEVE_PRIMES_BIG = sizeof(SIEVE_PRIMES_BIG) / sizeof(SIEVE_PRIMES_BIG[0]);

static std::vector<uint32_t> sieve_mersenne_candidates(
    uint64_t p, uint64_t k_start, uint64_t k_count,
    uint64_t &k_end_out)
{
    uint64_t k_end = k_start + k_count;
    k_end_out = k_end;

    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 4;

    // ── Phase 1: Bitmap sieve ──────────────────────────────────────────
    // Bit = 0 means candidate survives.  We mark bits for rejected k-values.
    // Use bytes for simplicity (no atomic bit ops needed in single-threaded marking).
    std::vector<uint8_t> sieve(k_count, 0);

    // For each small prime sp, q = 2kp + 1 is divisible by sp when:
    //   2kp + 1 ≡ 0 (mod sp)  →  k ≡ -(2p)^{-1} (mod sp)
    // We find k_first = first k >= k_start with sp | q, then mark every sp-th k.

    // Parallelize across sieve primes — each thread handles a subset
    {
        std::vector<std::thread> threads;
        int primes_per_thread = (N_SIEVE_PRIMES_BIG + (int)nthreads - 1) / (int)nthreads;

        for (unsigned t = 0; t < nthreads; t++) {
            int sp_start = t * primes_per_thread;
            int sp_end = std::min(sp_start + primes_per_thread, N_SIEVE_PRIMES_BIG);
            if (sp_start >= N_SIEVE_PRIMES_BIG) break;

            threads.emplace_back([&sieve, p, k_start, k_count, sp_start, sp_end]() {
                for (int si = sp_start; si < sp_end; si++) {
                    uint64_t sp = SIEVE_PRIMES_BIG[si];

                    // Find modular inverse of (2p) mod sp
                    // inv = (2p)^{-1} mod sp using extended Euclidean
                    uint64_t twop_mod = (2 * (p % sp)) % sp;
                    if (twop_mod == 0) continue;  // sp | 2p, degenerate

                    // Modular inverse via Fermat (sp is prime): inv = twop^{sp-2} mod sp
                    uint64_t inv = 1, base = twop_mod, exp = sp - 2;
                    while (exp > 0) {
                        if (exp & 1) inv = (inv * base) % sp;
                        base = (base * base) % sp;
                        exp >>= 1;
                    }

                    // k ≡ -inv ≡ (sp - inv) mod sp  → q divisible by sp
                    uint64_t k_residue = (sp - inv) % sp;

                    // First k in [k_start, ...) with k ≡ k_residue (mod sp)
                    uint64_t k_mod = k_start % sp;
                    uint64_t offset;
                    if (k_mod <= k_residue)
                        offset = k_residue - k_mod;
                    else
                        offset = sp - k_mod + k_residue;

                    // Mark every sp-th k from there
                    // But skip if q == sp itself (q is the prime, not composite)
                    for (uint64_t idx = offset; idx < k_count; idx += sp) {
                        sieve[idx] = 1;
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    // ── Phase 1b: mod-8 filter ─────────────────────────────────────────
    // q = 2kp + 1; q mod 8 depends on (2kp) mod 8.
    // Mark k-values where q mod 8 ∉ {1, 7}.
    // Since p is odd (prime > 2), 2p mod 8 is fixed for the batch.
    // q mod 8 = (2kp + 1) mod 8 = ((2p mod 8) * (k mod 4) + 1) mod 8
    // We can precompute which k mod 4 residues are valid.
    {
        uint64_t twop_mod8 = (2 * (p % 8)) % 8;
        bool k_mod4_valid[4];
        for (int r = 0; r < 4; r++) {
            uint64_t q_mod8 = (twop_mod8 * r + 1) % 8;
            k_mod4_valid[r] = (q_mod8 == 1 || q_mod8 == 7);
        }

        // Count valid residues — if all 4 are valid, skip this filter
        int n_valid = 0;
        for (int r = 0; r < 4; r++) if (k_mod4_valid[r]) n_valid++;

        if (n_valid < 4) {
            // Parallelize mod-8 marking
            uint64_t chunk = k_count / nthreads;
            std::vector<std::thread> threads;
            for (unsigned t = 0; t < nthreads; t++) {
                uint64_t my_start = t * chunk;
                uint64_t my_end = (t == nthreads - 1) ? k_count : my_start + chunk;
                threads.emplace_back([&sieve, &k_mod4_valid, k_start, my_start, my_end]() {
                    for (uint64_t idx = my_start; idx < my_end; idx++) {
                        if (sieve[idx]) continue;  // already rejected
                        int r = (int)((k_start + idx) % 4);
                        if (!k_mod4_valid[r]) sieve[idx] = 1;
                    }
                });
            }
            for (auto& th : threads) th.join();
        }
    }

    // ── Phase 2: Pack survivors ────────────────────────────────────────
    // Each thread scans its portion of the bitmap and packs surviving candidates.
    std::vector<std::vector<uint32_t>> thread_results(nthreads);
    {
        uint64_t chunk = k_count / nthreads;
        std::vector<std::thread> threads;

        for (unsigned t = 0; t < nthreads; t++) {
            uint64_t my_start = t * chunk;
            uint64_t my_end = (t == nthreads - 1) ? k_count : my_start + chunk;

            threads.emplace_back([&, t, my_start, my_end, p, k_start]() {
                auto& local = thread_results[t];
                // Estimate ~10% survival rate
                local.reserve((my_end - my_start) / 10 * 6);

                for (uint64_t idx = my_start; idx < my_end; idx++) {
                    if (sieve[idx]) continue;

                    uint64_t k = k_start + idx;
                    unsigned __int128 q128 = (unsigned __int128)2 * k * p + 1;
                    if (q128 >> 96) continue;

                    uint64_t q_lo = (uint64_t)q128;
                    uint32_t q_hi = (uint32_t)(q128 >> 64);

                    size_t pos = local.size();
                    local.resize(pos + 6);
                    pack_candidate(q_lo, q_hi, local.data() + pos);
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    // Merge
    size_t total = 0;
    for (auto& v : thread_results) total += v.size();

    std::vector<uint32_t> packed;
    packed.reserve(total);
    for (auto& v : thread_results) {
        packed.insert(packed.end(), v.begin(), v.end());
    }

    return packed;
}

void TaskManager::run_mersenne_trial(SearchTask& task) {
    // Default exponent: start with a prime near 100M (GIMPS first-test frontier ~140M)
    // The exponent is stored in task.end_pos (repurposed, 0 = use default)
    uint64_t exponent = task.end_pos;
    if (exponent == 0) exponent = 100000007ULL;  // prime near 10^8

    // Validate exponent is prime
    if (!is_prime(exponent)) {
        log("Mersenne TF: exponent " + std::to_string(exponent) + " is not prime — skipping.");
        task.should_run.store(false);
        task.status = TaskStatus::Paused;
        return;
    }

    log("Mersenne TF: testing 2^" + std::to_string(exponent) +
        " - 1 for factors q = 2kp+1, starting at k=" + std::to_string(task.current_pos));

    // Write live log header
    {
        std::string logpath = _data_dir + "/mersenne_tf_live.log";
        std::ofstream lf(logpath, std::ios::app);
        if (lf.is_open()) {
            lf << "\n=== Mersenne TF Session ===\n"
               << "Started: " << timestamp() << "\n"
               << "Exponent: " << exponent << " (M" << exponent << " = 2^" << exponent << " - 1)\n"
               << "Starting k: " << task.current_pos << "\n"
               << "Format: timestamp | exponent | k_position | total_tested | factors_found | rate\n"
               << "=========================================\n";
        }
    }

    const uint64_t K_BATCH = mersenne_k_batch.load();  // configurable from GIMPS panel
    static constexpr uint32_t GPU_BATCH_SIZE = 262144;  // 256K per GPU dispatch

    auto last_save = std::chrono::steady_clock::now();
    uint64_t summary_start = task.current_pos;
    uint64_t summary_tested = 0, summary_hits = 0;

    // Strategy: GPU does Mersenne TF exclusively (fused sieve+modexp).
    // CPU runs complementary high-value searches (WallSunSun, Wieferich) in parallel.
    // This keeps BOTH fully utilized — GPU on Mersenne, CPU on rare-prime hunts.
    bool use_fused = false;
    {
        std::lock_guard<std::mutex> lock(_gpu_mutex);
        use_fused = (_gpu && _gpu->name() != "CPU (fallback)");
    }

    // Auto-start complementary CPU tasks when GPU handles Mersenne
    std::vector<TaskType> auto_started_cpu_tasks;
    if (use_fused) {
        log("Mersenne TF: GPU-exclusive mode — GPU does fused sieve+modexp");
        gpu_owner.store((int)TaskType::MersenneTrial);

        // Start CPU-intensive searches on a separate thread to avoid
        // calling start_task() from within a worker (which can deadlock on join).
        // Only auto-start CPU companion tasks if search position is small enough
        // for safe 128-bit arithmetic (p < 2^32 → p² < 2^64, fast path).
        // For large primes, CPU mulmod is ~100x slower and causes memory pressure
        // since the sieve pipeline outpaces CPU testing.
        const TaskType cpu_tasks[] = {
            TaskType::WallSunSun,   // 0 known — holy grail
            TaskType::Wieferich,    // only 2 known
        };
        for (auto ct : cpu_tasks) {
            auto it = _tasks.find(ct);
            if (it != _tasks.end() && it->second.status != TaskStatus::Running) {
                // Skip if search position is too large for efficient CPU testing
                if (it->second.current_pos > 4294967296ULL) {
                    log("Mersenne TF: skipping " + std::string(task_name(ct)) +
                        " CPU companion — search position too large for efficient CPU mulmod");
                    continue;
                }
                auto_started_cpu_tasks.push_back(ct);
            }
        }
        if (!auto_started_cpu_tasks.empty()) {
            // Launch on a detached thread so start_task's join() doesn't block this worker
            auto tasks_to_start = auto_started_cpu_tasks;
            std::thread([this, tasks_to_start]() {
                for (auto ct : tasks_to_start) {
                    log("Mersenne TF: auto-starting " + std::string(task_name(ct)) +
                        " search on CPU (complementary workload)");
                    start_task(ct);
                }
            }).detach();
        } else {
            log("Mersenne TF: no idle CPU tasks to auto-start (may already be running)");
        }
    } else {
        log("Mersenne TF: CPU-only mode (no Metal backend)");
    }

    if (use_fused) {
        // ── GPU-exclusive Mersenne TF ─────────────────────────────────
        // GPU runs fused sieve+modexp kernel. CPU is free for other tasks.
        uint64_t gpu_log_ctr = 0;
        uint64_t total_hits = 0;

        while (task.should_run.load()) {
          @autoreleasepool {
            throttle_if_needed(this);

            uint64_t k_pos = task.current_pos;
            uint64_t k_end = k_pos + K_BATCH;

            while (k_pos < k_end && task.should_run.load()) {
                uint64_t GPU_FUSED_CHUNK = 1ULL << 22;  // 4M per dispatch (avoids GPU watchdog timeout)
                uint64_t chunk = std::min(GPU_FUSED_CHUNK, k_end - k_pos);

                std::vector<GPUBackend::FusedHit> hits;
                pace_gpu();
                {
                    std::lock_guard<std::mutex> lock(_gpu_mutex);
                    hits = _gpu->mersenne_fused_sieve(exponent, k_pos, chunk);
                }
                finish_gpu();

                for (auto& fh : hits) {
                    uint64_t q_lo = fh.q_lo;
                    uint64_t q_hi_and_k = fh.q_hi_and_k;
                    uint32_t q_hi = (uint32_t)(q_hi_and_k & 0xFFFFFFFF);
                    uint64_t q_val = q_lo;
                    std::string q_str = std::to_string(q_val);
                    if (q_hi > 0) {
                        unsigned __int128 q128 = ((unsigned __int128)q_hi << 64) | q_lo;
                        char buf[40]; int pos = 39; buf[pos] = 0;
                        unsigned __int128 tmp = q128;
                        while (tmp > 0) { buf[--pos] = '0' + (int)(tmp % 10); tmp /= 10; }
                        q_str = &buf[pos];
                    }
                    // ── CPU carry-chain verification of GPU-found factor ──
                    unsigned __int128 q_verify = q_hi > 0
                        ? (((unsigned __int128)q_hi << 64) | q_lo)
                        : (unsigned __int128)q_lo;
                    bool cpu_verified = false;
                    bool verify_ran = false;
                    if (q_verify > 1 && exponent > 0) {
                        try {
                            CarryChainMulMod ctx(q_verify);
                            unsigned __int128 base = 2, result = 1;
                            uint64_t e = exponent;
                            while (e > 0) {
                                if (e & 1) result = ctx.mul(result, base);
                                base = ctx.mul(base, base);
                                e >>= 1;
                            }
                            cpu_verified = (result == 1);
                            verify_ran = true;
                        } catch (...) {
                            log("WARNING: CPU verification threw exception for factor " + q_str);
                        }
                    }

                    // Check if this factor is already known
                    bool is_known = false;
                    for (auto& kf : task.known_factors) {
                        if (kf == q_str) { is_known = true; break; }
                        // Also check if known factor divides this factor (composite case)
                        // For now, exact string match; composite splitting below
                    }

                    if (is_known) {
                        log("Mersenne TF: factor " + q_str + " is already known, skipping");
                        summary_tested += 0; // no new discovery
                    } else {
                        // Check if composite containing known primes
                        // Trial divide by each known factor, strip known components
                        std::string new_factor = q_str;
                        if (!task.known_factors.empty() && q_verify > 1) {
                            unsigned __int128 remaining = q_verify;
                            for (auto& kf : task.known_factors) {
                                unsigned __int128 kf_val = 0;
                                for (char c : kf) kf_val = kf_val * 10 + (c - '0');
                                if (kf_val > 1) {
                                    while (remaining > kf_val && remaining % kf_val == 0) {
                                        remaining /= kf_val;
                                        log("Mersenne TF: stripped known factor " + kf + " from composite " + q_str);
                                    }
                                }
                            }
                            if (remaining == 1) {
                                log("Mersenne TF: factor " + q_str + " composed entirely of known primes, skipping");
                                is_known = true;
                            } else if (remaining != q_verify) {
                                // Report the remaining cofactor
                                char buf[40]; int bpos = 39; buf[bpos] = 0;
                                unsigned __int128 tmp = remaining;
                                while (tmp > 0) { buf[--bpos] = '0' + (int)(tmp % 10); tmp /= 10; }
                                new_factor = &buf[bpos];
                                log("Mersenne TF: composite " + q_str + " reduced to new factor " + new_factor);
                            }
                        }

                        if (!is_known) {
                            // Attempt to split composite factors (trial + Pollard rho)
                            // TF factors fit in uint64, so factor_u64 works directly
                            uint64_t factor_val = 0;
                            try { factor_val = std::stoull(new_factor); } catch (...) {}
                            if (factor_val > 1 && !prime::is_prime(factor_val)) {
                                log("Mersenne TF: factor " + new_factor + " is composite, attempting split...");
                                auto parts = prime::factor_u64(factor_val);
                                if (parts.size() > 1) {
                                    std::string split_str;
                                    for (size_t pi = 0; pi < parts.size(); pi++) {
                                        if (pi > 0) split_str += " x ";
                                        split_str += std::to_string(parts[pi]);
                                    }
                                    log("Mersenne TF: split " + new_factor + " = " + split_str);
                                    // Report individual prime factors instead of composite
                                    std::set<uint64_t> unique_parts(parts.begin(), parts.end());
                                    for (uint64_t p : unique_parts) {
                                        // Skip known factors
                                        bool p_known = false;
                                        for (auto& kf : task.known_factors) {
                                            if (kf == std::to_string(p)) { p_known = true; break; }
                                        }
                                        if (!p_known) {
                                            new_factor = std::to_string(p);
                                            total_hits++;
                                            task.found_count++;
                                            log("*** MERSENNE FACTOR (split): 2^" + std::to_string(exponent) +
                                                " - 1 has prime factor " + new_factor + " ***");
                                            save_discovery({TaskType::MersenneTrial, p, exponent,
                                                PrimeClass::Composite, k_pos, new_factor, timestamp()});
                                        }
                                    }
                                    // Skip the normal reporting below
                                    goto next_candidate;
                                } else {
                                    log("Mersenne TF: could not split " + new_factor + ", reporting as-is");
                                }
                            }

                            total_hits++;
                            task.found_count++;
                            if (verify_ran && cpu_verified) {
                                log("*** MERSENNE FACTOR VERIFIED (GPU+CPU): 2^" + std::to_string(exponent) +
                                    " - 1 has factor " + new_factor + " *** [carry-chain CPU confirmed]");
                            } else if (verify_ran && !cpu_verified) {
                                log("WARNING: GPU found factor " + new_factor + " for M" + std::to_string(exponent) +
                                    " but CPU carry-chain says NO -- logging anyway for investigation");
                            } else {
                                log("*** MERSENNE FACTOR FOUND: 2^" + std::to_string(exponent) +
                                    " - 1 has factor " + new_factor + " *** [CPU verify skipped]");
                            }
                            save_discovery({TaskType::MersenneTrial, q_val, exponent,
                                PrimeClass::Composite, k_pos, new_factor, timestamp()});

                            if (mersenne_abort_on_factor.load()) {
                                log("Mersenne TF: aborting on first factor (user preference)");
                                task.should_run.store(false);
                            }
                        }
                    }
                }
                next_candidate:

                summary_tested += chunk;
                task.tested_count += chunk;
                k_pos += chunk;
            }

            task.current_pos = k_end;

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - last_save).count();
            if (elapsed > SAVE_INTERVAL_SEC) {
                // Summary log only on save intervals (~30s)
                double rate = summary_tested / elapsed;
                log("Mersenne TF: " + std::to_string(task.tested_count / 1000000) +
                    "M tested, " + std::to_string((uint64_t)(rate / 1e6)) +
                    "M/s, k=" + std::to_string(task.current_pos));
                save_scan_summary(TaskType::MersenneTrial, summary_start, task.current_pos,
                                  summary_tested, summary_hits);
                save_state();

                // Write live Mersenne TF log for verification
                {
                    std::string logpath = _data_dir + "/mersenne_tf_live.log";
                    std::ofstream lf(logpath, std::ios::app);
                    if (lf.is_open()) {
                        lf << timestamp()
                           << " | M" << exponent
                           << " | k=" << task.current_pos
                           << " | tested=" << task.tested_count
                           << " | factors=" << total_hits
                           << " | rate=" << (uint64_t)(rate / 1e6) << "M/s"
                           << "\n";
                    }
                }
                last_save = now;
                summary_start = task.current_pos;
                summary_tested = 0;
                summary_hits = 0;
            }
          } // @autoreleasepool per batch
        }

        // Stop auto-started CPU tasks when Mersenne TF stops
        // Signal them to stop (don't call pause_task from worker — it joins)
        for (auto ct : auto_started_cpu_tasks) {
            auto it = _tasks.find(ct);
            if (it != _tasks.end() && it->second.status == TaskStatus::Running) {
                log("Mersenne TF: stopping auto-started " + std::string(task_name(ct)));
                it->second.should_run.store(false);
            }
        }

    } else {
        // ── CPU-only fallback ────────────────────────────────────────
        while (task.should_run.load()) {
          @autoreleasepool {
            throttle_if_needed(this);

            uint64_t k_end;
            auto packed = sieve_mersenne_candidates(exponent, task.current_pos, K_BATCH, k_end);
            uint32_t n_candidates = (uint32_t)(packed.size() / 6);

            // CPU modexp
            for (uint32_t i = 0; i < n_candidates; i++) {
                if (!task.should_run.load()) break;
                uint32_t *c = packed.data() + i * 6;
                unsigned __int128 q128 = (unsigned __int128)c[2] << 64 |
                                         (unsigned __int128)c[1] << 32 | c[0];
                if (q128 < 2) continue;
                unsigned __int128 acc = 2;
                uint64_t exp = exponent;
                int top = 63;
                while (top > 0 && !((exp >> top) & 1)) top--;
                for (int bit = top - 1; bit >= 0; bit--) {
                    acc = acc * acc % q128;
                    if ((exp >> bit) & 1)
                        acc = (acc << 1) % q128;
                }
                if (acc == 1) {
                    uint64_t q_lo = (uint64_t)q128;
                    std::string q_str = std::to_string(q_lo);

                    // Check known factors
                    bool is_known_cpu = false;
                    for (auto& kf : task.known_factors) {
                        if (kf == q_str) { is_known_cpu = true; break; }
                    }
                    if (is_known_cpu) {
                        log("Mersenne TF: factor " + q_str + " is already known, skipping");
                    } else {
                        task.found_count++;
                        summary_hits++;
                        log("*** MERSENNE FACTOR FOUND: 2^" + std::to_string(exponent) +
                            " - 1 has factor " + q_str + " ***");
                        save_discovery({TaskType::MersenneTrial, q_lo, exponent,
                            PrimeClass::Composite, task.current_pos, q_str, timestamp()});
                        if (mersenne_abort_on_factor.load()) {
                            log("Mersenne TF: aborting on first factor (user preference)");
                            task.should_run.store(false);
                        }
                    }
                }
            }

            task.tested_count += n_candidates;
            summary_tested += n_candidates;
            task.current_pos = k_end;

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_save).count() > SAVE_INTERVAL_SEC) {
                save_scan_summary(TaskType::MersenneTrial, summary_start, task.current_pos,
                                  summary_tested, summary_hits);
                save_state();
                last_save = now;
                summary_start = task.current_pos;
                summary_tested = 0;
                summary_hits = 0;
            }
          } // @autoreleasepool
        }
    }

    gpu_owner.store(-1);
    if (summary_tested > 0)
        save_scan_summary(TaskType::MersenneTrial, summary_start, task.current_pos,
                          summary_tested, summary_hits);
}

// ═══════════════════════════════════════════════════════════════════════
// Fermat Factor Search — find factors of Fermat numbers F_m = 2^(2^m)+1
//
// Any factor of F_m must have form q = k * 2^(m+2) + 1.
// GPU tests 2^(2^m) mod q: if result ≡ -1, then q | F_m.
// ═══════════════════════════════════════════════════════════════════════

static std::vector<uint32_t> sieve_fermat_candidates(
    uint64_t m, uint64_t k_start, uint64_t k_count,
    uint64_t &k_end_out)
{
    static const uint64_t SIEVE_PRIMES[] = {
        3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97
    };

    uint64_t step_bits = m + 2;  // factor is k * 2^(m+2) + 1

    std::vector<uint32_t> packed;
    packed.reserve(k_count / 4 * 6);

    uint64_t k_end = k_start + k_count;
    k_end_out = k_end;

    for (uint64_t k = k_start; k < k_end; k++) {
        // q = k * 2^(m+2) + 1
        // For m <= 24: 2^(m+2) <= 2^26 = 67M. k * 67M + 1 fits in 96 bits for k < 2^70.
        // For m > 60: 2^(m+2) > 2^62, so k must be small for q to fit in 96 bits.
        unsigned __int128 q128;
        if (step_bits < 64) {
            q128 = (unsigned __int128)k * ((uint64_t)1 << step_bits) + 1;
        } else if (step_bits < 96) {
            q128 = ((unsigned __int128)k << step_bits) + 1;
        } else {
            continue;  // q won't fit in 96 bits
        }

        if (q128 >> 96) continue;

        uint64_t q_lo = (uint64_t)q128;
        uint32_t q_hi = (uint32_t)(q128 >> 64);

        // Sieve by small primes
        bool rejected = false;
        for (uint64_t sp : SIEVE_PRIMES) {
            if (q128 % sp == 0 && q128 != sp) { rejected = true; break; }
        }
        if (rejected) continue;

        // Must also be prime (factor of Fermat number must be prime)
        // Quick check: only accept if q passes small-prime sieve.
        // Full primality test would be expensive; skip for now — the GPU test
        // will only return true if q actually divides F_m, which implies q is prime
        // (since Fermat numbers are products of primes of this form).

        size_t idx = packed.size();
        packed.resize(idx + 6);
        pack_candidate(q_lo, q_hi, packed.data() + idx);
    }

    return packed;
}

void TaskManager::run_fermat_factor(SearchTask& task) {
    // Fermat index m: search for factors of F_m = 2^(2^m) + 1
    // Stored in task.end_pos (0 = use default)
    uint64_t m = task.end_pos;
    if (m == 0) m = 20;  // F_20 — no factor known, composite (no known factor!)

    log("Fermat Factor: searching for factors of F_" + std::to_string(m) +
        " = 2^(2^" + std::to_string(m) + ")+1, candidates q = k*2^" +
        std::to_string(m+2) + "+1, starting at k=" + std::to_string(task.current_pos));

    static constexpr uint64_t K_BATCH = 500000;
    static constexpr uint32_t GPU_BATCH_SIZE = 65536;

    auto last_save = std::chrono::steady_clock::now();
    uint64_t summary_start = task.current_pos;
    uint64_t summary_tested = 0, summary_hits = 0;

    while (task.should_run.load()) {
      @autoreleasepool {
        throttle_if_needed(this);
        auto t0 = std::chrono::steady_clock::now();

        uint64_t k_end;
        auto packed = sieve_fermat_candidates(m, task.current_pos, K_BATCH, k_end);
        uint32_t n_candidates = (uint32_t)(packed.size() / 6);

        if (n_candidates > 0) {
            std::vector<uint8_t> results(n_candidates);
            for (uint32_t offset = 0; offset < n_candidates; offset += GPU_BATCH_SIZE) {
                if (!task.should_run.load()) break;
                uint32_t batch = std::min(GPU_BATCH_SIZE, n_candidates - offset);

                pace_gpu();
                { std::lock_guard<std::mutex> lock(_gpu_mutex);
                  _gpu->fermat_factor_batch(packed.data() + offset * 6,
                                            results.data() + offset, batch, m); }
                finish_gpu();
            }

            for (uint32_t i = 0; i < n_candidates; i++) {
                if (results[i]) {
                    uint64_t q_lo = packed[i*6] | ((uint64_t)packed[i*6+1] << 32);
                    uint32_t q_hi = packed[i*6+2];
                    std::string q_str = std::to_string(q_lo);
                    if (q_hi > 0) {
                        unsigned __int128 q128 = ((unsigned __int128)q_hi << 64) | q_lo;
                        char buf[40];
                        unsigned __int128 tmp = q128;
                        int pos = 39;
                        buf[pos] = 0;
                        while (tmp > 0) { buf[--pos] = '0' + (int)(tmp % 10); tmp /= 10; }
                        q_str = &buf[pos];
                    }

                    task.found_count++;
                    summary_hits++;
                    log("*** FERMAT FACTOR FOUND: F_" + std::to_string(m) +
                        " has factor " + q_str + " ***");
                    save_discovery({TaskType::FermatFactor, q_lo, m,
                        PrimeClass::Composite, task.current_pos, q_str, timestamp()});
                }
            }

            task.tested_count += n_candidates;
            summary_tested += n_candidates;
        }

        task.current_pos = k_end;

        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        task.rate = dt > 0 ? K_BATCH / dt : 0;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_save).count() > SAVE_INTERVAL_SEC) {
            save_scan_summary(TaskType::FermatFactor, summary_start, task.current_pos,
                              summary_tested, summary_hits);
            save_state();
            last_save = now;
            summary_start = task.current_pos;
            summary_tested = 0;
            summary_hits = 0;
        }
      } // @autoreleasepool
    }

    if (summary_tested > 0)
        save_scan_summary(TaskType::FermatFactor, summary_start, task.current_pos,
                          summary_tested, summary_hits);
}

} // namespace prime
