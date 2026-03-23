#pragma once
#include "../TaskManager.hpp"
#include "NetworkProtocol.hpp"
#include "TCPSocket.hpp"
#include "WorkSplitter.hpp"
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <chrono>
#include <memory>

namespace prime {

// Import network types into prime namespace for convenience
using net::CarriageInfo;
using net::WorkChunk;
using net::TCPSocket;
using net::ProgressMsg;
using net::DiscoveryMsg;

// ═══════════════════════════════════════════════════════════════════════
// ConductorServer — the Engine that coordinates distributed prime search
//
// The Conductor is both coordinator AND worker:
//   - Accepts Carriage TCP connections (discovered via Bonjour)
//   - Distributes work chunks across all connected Carriages
//   - Keeps a local chunk for itself (runs its own TaskManager)
//   - Aggregates progress, discoveries, and handles failover
// ═══════════════════════════════════════════════════════════════════════

// Callbacks from the Conductor to the UI / owner
using CarriageConnectedCallback    = std::function<void(uint32_t carriage_id, const std::string& name)>;
using CarriageDisconnectedCallback = std::function<void(uint32_t carriage_id, const std::string& name)>;
using CarriageProgressCallback     = std::function<void(uint32_t carriage_id, const ProgressMsg& info)>;
using RemoteDiscoveryCallback      = std::function<void(uint32_t carriage_id, const DiscoveryMsg& disc)>;

class ConductorServer {
public:
    ConductorServer(uint16_t port, TaskManager* local_task_mgr);
    ~ConductorServer();

    // ── Lifecycle ───────────────────────────────────────────────────
    void start();   // opens TCP listener, publishes Bonjour, starts accept + heartbeat threads
    void stop();    // tears everything down gracefully

    bool is_running() const { return _running.load(); }
    uint16_t port() const { return _port; }

    // ── Work distribution ───────────────────────────────────────────
    // Splits [start, end) across all connected carriages + local.
    // Local chunk is started on _local_task_mgr; remote chunks sent as AssignWork.
    void distribute_task(TaskType type, uint64_t start, uint64_t end);

    // Assign a specific chunk to one carriage
    void assign_work(uint32_t carriage_id, const WorkChunk& chunk);

    // ── Status ──────────────────────────────────────────────────────
    std::vector<CarriageInfo> connected_carriages() const;
    size_t carriage_count() const;

    // ── Callbacks ───────────────────────────────────────────────────
    void set_on_carriage_connected(CarriageConnectedCallback cb)       { _on_connected = cb; }
    void set_on_carriage_disconnected(CarriageDisconnectedCallback cb) { _on_disconnected = cb; }
    void set_on_carriage_progress(CarriageProgressCallback cb)         { _on_progress = cb; }
    void set_on_remote_discovery(RemoteDiscoveryCallback cb)           { _on_discovery = cb; }

private:
    uint16_t _port;
    TaskManager* _local_task_mgr;   // local computation engine
    std::atomic<bool> _running{false};

    // ── Networking ──────────────────────────────────────────────────
    TCPSocket _server_socket;       // listening socket
    void* _bonjour_publisher = nullptr;  // BonjourPublisher* (ObjC, opaque in C++)

    // ── Carriage registry ───────────────────────────────────────────
    struct ConnectedCarriage {
        CarriageInfo info;
        std::unique_ptr<TCPSocket> socket;
        WorkChunk    assigned_chunk;
        bool         has_work = false;
        std::chrono::steady_clock::time_point last_pong;
        std::thread  recv_thread;
    };

    mutable std::mutex _carriages_mutex;
    std::map<uint32_t, ConnectedCarriage> _carriages;
    uint32_t _next_carriage_id = 1;

    // ── Background threads ──────────────────────────────────────────
    std::thread _accept_thread;
    std::thread _heartbeat_thread;

    void accept_loop();
    void heartbeat_loop();
    void receive_loop(uint32_t carriage_id);
    void handle_message(uint32_t carriage_id, const std::string& json_body);
    void disconnect_carriage(uint32_t carriage_id);
    void reassign_failed_work(const WorkChunk& chunk);

    // ── Callbacks ───────────────────────────────────────────────────
    CarriageConnectedCallback    _on_connected;
    CarriageDisconnectedCallback _on_disconnected;
    CarriageProgressCallback     _on_progress;
    RemoteDiscoveryCallback      _on_discovery;
};

} // namespace prime
