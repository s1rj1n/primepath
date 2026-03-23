#pragma once
#include "../TaskManager.hpp"
#include "../GPUBackend.hpp"
#include "NetworkProtocol.hpp"
#include "TCPSocket.hpp"
#include "WorkSplitter.hpp"
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <chrono>
#include <memory>

namespace prime {

// Import network types into prime namespace
using net::CarriageInfo;
using net::WorkChunk;
using net::TCPSocket;
using net::DiscoveryMsg;

// ═══════════════════════════════════════════════════════════════════════
// CarriageClient — a worker node that connects to a Conductor
//
// The Carriage:
//   - Discovers Conductor via Bonjour (or manual IP)
//   - Connects over TCP, sends Hello with local machine info
//   - Receives work assignments and runs them on a local TaskManager
//   - Reports progress every 1s, discoveries immediately
//   - Responds to heartbeat pings
//   - Auto-reconnects on disconnect
// ═══════════════════════════════════════════════════════════════════════

using CarriageStatusCallback = std::function<void(const std::string& status)>;
using CarriageWorkCallback   = std::function<void(const WorkChunk& chunk)>;

class CarriageClient {
public:
    // data_dir: scratch folder for local TaskManager state
    CarriageClient(const std::string& data_dir);
    ~CarriageClient();

    // ── Connection ──────────────────────────────────────────────────
    // Auto-discover Conductor via Bonjour and connect
    void start_discovery();

    // Connect directly to a known Conductor
    void connect(const std::string& host, uint16_t port);

    // Disconnect and shut down
    void stop();

    bool is_connected() const { return _connected.load(); }
    bool is_working() const   { return _working.load(); }

    // ── Status ──────────────────────────────────────────────────────
    void set_status_callback(CarriageStatusCallback cb) { _on_status = cb; }
    void set_work_callback(CarriageWorkCallback cb)     { _on_work = cb; }

    // Current work progress
    std::string conductor_host() const;
    uint16_t    conductor_port() const;

private:
    std::string _data_dir;

    // ── Networking ──────────────────────────────────────────────────
    TCPSocket _socket;
    std::string _conductor_host;
    uint16_t    _conductor_port = 0;
    std::atomic<bool> _connected{false};
    std::atomic<bool> _working{false};
    std::atomic<bool> _should_run{false};

    mutable std::mutex _mutex;

    // ── Bonjour discovery ───────────────────────────────────────────
    void* _bonjour_browser = nullptr;  // BonjourBrowser* (ObjC, opaque)

    // ── Background threads ──────────────────────────────────────────
    std::thread _recv_thread;
    std::thread _progress_thread;
    std::thread _reconnect_thread;

    void recv_loop();
    void progress_loop();
    void reconnect_loop();

    // ── Message handling ────────────────────────────────────────────
    void handle_message(const std::string& json_body);
    void handle_assign_work(const std::string& json_body);
    void send_hello();
    void send_progress();
    void send_discovery(const DiscoveryMsg& disc);
    void send_work_done();

    // ── Local computation ───────────────────────────────────────────
    std::unique_ptr<TaskManager>  _task_mgr;
    std::unique_ptr<GPUBackend>   _gpu;
    WorkChunk                     _current_chunk;

    void start_local_work(const WorkChunk& chunk);
    void stop_local_work();

    // ── Machine info ────────────────────────────────────────────────
    static std::string local_hostname();
    static uint32_t    cpu_core_count();
    static uint64_t    total_memory_mb();

    // ── Callbacks ───────────────────────────────────────────────────
    CarriageStatusCallback _on_status;
    CarriageWorkCallback   _on_work;

    void report_status(const std::string& msg);
};

} // namespace prime
