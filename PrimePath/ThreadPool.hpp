#pragma once
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>

// ObjC autorelease pool C API — works in both .cpp and .mm translation units.
// Prevents Metal command buffers and NSData objects from accumulating.
extern "C" void *objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void *pool);

namespace prime {

// ═══════════════════════════════════════════════════════════════════════
// ThreadPool — fixed-size pool for parallelising sieve + batch work
// ═══════════════════════════════════════════════════════════════════════

class ThreadPool {
public:
    explicit ThreadPool(int n_threads = 0) {
        if (n_threads <= 0)
            n_threads = (int)std::thread::hardware_concurrency();
        _n_threads = n_threads;
        for (int i = 0; i < n_threads; i++) {
            _workers.emplace_back([this]() { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _stop = true;
        }
        _cv.notify_all();
        for (auto& w : _workers) {
            if (w.joinable()) w.join();
        }
    }

    int size() const { return _n_threads; }

    // Submit a task, get a future
    template<typename F>
    auto submit(F&& f) -> std::future<decltype(f())> {
        using RetType = decltype(f());
        auto task = std::make_shared<std::packaged_task<RetType()>>(std::forward<F>(f));
        auto future = task->get_future();
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _tasks.emplace([task]() { (*task)(); });
        }
        _cv.notify_one();
        return future;
    }

    // Parallel for: split [start, end) across threads
    void parallel_for(uint64_t start, uint64_t end, std::function<void(uint64_t, uint64_t)> fn) {
        if (end <= start) return;
        uint64_t range = end - start;
        uint64_t chunk = range / _n_threads;
        if (chunk == 0) chunk = 1;

        std::vector<std::future<void>> futures;
        for (int t = 0; t < _n_threads; t++) {
            uint64_t lo = start + t * chunk;
            uint64_t hi = (t == _n_threads - 1) ? end : lo + chunk;
            if (lo >= end) break;
            futures.push_back(submit([fn, lo, hi]() { fn(lo, hi); }));
        }
        for (auto& f : futures) f.get();
    }

private:
    int _n_threads;
    std::vector<std::thread> _workers;
    std::queue<std::function<void()>> _tasks;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _stop = false;

    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _cv.wait(lock, [this]() { return _stop || !_tasks.empty(); });
                if (_stop && _tasks.empty()) return;
                task = std::move(_tasks.front());
                _tasks.pop();
            }
            // Drain ObjC autorelease pool per task to prevent Metal/NSData accumulation.
            // Uses C runtime API so this compiles in both .cpp and .mm translation units.
            void *pool = objc_autoreleasePoolPush();
            task();
            objc_autoreleasePoolPop(pool);
        }
    }
};

} // namespace prime
