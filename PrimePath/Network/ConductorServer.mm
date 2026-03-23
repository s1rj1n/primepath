#include "ConductorServer.hpp"
#import "BonjourService.h"
#include <algorithm>

namespace prime {

using namespace net;

// ═══════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ═══════════════════════════════════════════════════════════════════════

ConductorServer::ConductorServer(uint16_t port, TaskManager* local_task_mgr)
    : _port(port)
    , _local_task_mgr(local_task_mgr)
{}

ConductorServer::~ConductorServer() {
    stop();
}

// ═══════════════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════════════

void ConductorServer::start() {
    if (_running.load()) return;

    if (!_server_socket.listen_on(_port)) {
        if (_local_task_mgr)
            _local_task_mgr->log_msg("[Conductor] Failed to bind TCP port " + std::to_string(_port));
        return;
    }

    _running.store(true);

    // Publish via Bonjour
    BonjourPublisher *publisher = [[BonjourPublisher alloc] initWithPort:_port];
    [publisher start];
    _bonjour_publisher = (__bridge_retained void*)publisher;

    _accept_thread = std::thread([this] { accept_loop(); });
    _heartbeat_thread = std::thread([this] { heartbeat_loop(); });

    if (_local_task_mgr)
        _local_task_mgr->log_msg("[Conductor] Started on port " + std::to_string(_port));
}

void ConductorServer::stop() {
    if (!_running.load()) return;
    _running.store(false);

    if (_bonjour_publisher) {
        BonjourPublisher *publisher = (__bridge_transfer BonjourPublisher*)_bonjour_publisher;
        [publisher stop];
        _bonjour_publisher = nullptr;
    }

    _server_socket.close();

    if (_accept_thread.joinable())
        _accept_thread.join();
    if (_heartbeat_thread.joinable())
        _heartbeat_thread.join();

    {
        std::lock_guard<std::mutex> lock(_carriages_mutex);
        for (auto& [id, c] : _carriages) {
            if (c.socket) c.socket->close();
            if (c.recv_thread.joinable())
                c.recv_thread.detach();
        }
        _carriages.clear();
    }

    if (_local_task_mgr)
        _local_task_mgr->log_msg("[Conductor] Stopped.");
}

// ═══════════════════════════════════════════════════════════════════════
// Accept loop
// ═══════════════════════════════════════════════════════════════════════

void ConductorServer::accept_loop() {
    while (_running.load()) {
        auto client = _server_socket.accept_connection();
        if (!client || !client->is_valid()) continue;

        // Read Hello message
        std::string hello_json = client->recv_message();
        if (hello_json.empty()) continue;

        auto msg_type = message_type_from_json(hello_json);
        if (msg_type != MessageType::Hello) {
            client->close();
            continue;
        }

        auto obj = json_parse(hello_json);
        CarriageInfo info = CarriageInfo::deserialize(obj);
        info.connected = true;

        uint32_t cid;
        {
            std::lock_guard<std::mutex> lock(_carriages_mutex);
            cid = _next_carriage_id++;

            ConnectedCarriage cc;
            cc.info = info;
            cc.socket = std::move(client);
            cc.last_pong = std::chrono::steady_clock::now();
            _carriages[cid] = std::move(cc);
            _carriages[cid].recv_thread = std::thread([this, cid] { receive_loop(cid); });
        }

        if (_local_task_mgr)
            _local_task_mgr->log_msg("[Conductor] Carriage connected: " + info.hostname
                                      + " (" + std::to_string(info.cores) + " cores, "
                                      + info.gpu_name + ")");
        if (_on_connected)
            _on_connected(cid, info.hostname);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Receive loop — one per connected carriage
// ═══════════════════════════════════════════════════════════════════════

void ConductorServer::receive_loop(uint32_t carriage_id) {
    while (_running.load()) {
        std::string json_body;
        {
            std::lock_guard<std::mutex> lock(_carriages_mutex);
            auto it = _carriages.find(carriage_id);
            if (it == _carriages.end()) return;
            json_body = it->second.socket->recv_message();
        }

        if (json_body.empty()) {
            disconnect_carriage(carriage_id);
            return;
        }

        handle_message(carriage_id, json_body);
    }
}

void ConductorServer::handle_message(uint32_t carriage_id, const std::string& json_body) {
    auto msg_type = message_type_from_json(json_body);
    auto obj = json_parse(json_body);

    switch (msg_type) {
        case MessageType::Pong: {
            std::lock_guard<std::mutex> lock(_carriages_mutex);
            auto it = _carriages.find(carriage_id);
            if (it != _carriages.end())
                it->second.last_pong = std::chrono::steady_clock::now();
            break;
        }

        case MessageType::Progress: {
            auto pm = ProgressMsg::deserialize(obj);
            if (_on_progress)
                _on_progress(carriage_id, pm);
            break;
        }

        case MessageType::DiscoveryReport: {
            auto dm = DiscoveryMsg::deserialize(obj);
            if (_on_discovery)
                _on_discovery(carriage_id, dm);
            break;
        }

        case MessageType::WorkDone: {
            auto wm = WorkDoneMsg::deserialize(obj);
            {
                std::lock_guard<std::mutex> lock(_carriages_mutex);
                auto it = _carriages.find(carriage_id);
                if (it != _carriages.end())
                    it->second.has_work = false;
            }
            if (_local_task_mgr)
                _local_task_mgr->log_msg("[Conductor] Carriage " + std::to_string(carriage_id)
                                          + " completed chunk: " + std::to_string(wm.tested)
                                          + " tested, " + std::to_string(wm.found) + " found");
            break;
        }

        default:
            break;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Heartbeat — ping every 5s, disconnect if no pong in 10s
// ═══════════════════════════════════════════════════════════════════════

void ConductorServer::heartbeat_loop() {
    uint64_t seq = 0;
    while (_running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        if (!_running.load()) break;

        auto now = std::chrono::steady_clock::now();
        std::vector<uint32_t> dead;

        {
            std::lock_guard<std::mutex> lock(_carriages_mutex);
            for (auto& [id, c] : _carriages) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - c.last_pong).count();
                if (elapsed > 10) {
                    dead.push_back(id);
                } else {
                    PingMsg ping{++seq};
                    c.socket->send_message(ping.serialize());
                }
            }
        }

        for (uint32_t id : dead) {
            if (_local_task_mgr)
                _local_task_mgr->log_msg("[Conductor] Carriage " + std::to_string(id)
                                          + " timed out.");
            disconnect_carriage(id);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Disconnect + failover
// ═══════════════════════════════════════════════════════════════════════

void ConductorServer::disconnect_carriage(uint32_t carriage_id) {
    std::string name;
    WorkChunk failed_chunk;
    bool had_work = false;

    {
        std::lock_guard<std::mutex> lock(_carriages_mutex);
        auto it = _carriages.find(carriage_id);
        if (it == _carriages.end()) return;

        name = it->second.info.hostname;
        had_work = it->second.has_work;
        failed_chunk = it->second.assigned_chunk;

        if (it->second.socket) it->second.socket->close();
        if (it->second.recv_thread.joinable())
            it->second.recv_thread.detach();
        _carriages.erase(it);
    }

    if (_on_disconnected)
        _on_disconnected(carriage_id, name);

    if (_local_task_mgr)
        _local_task_mgr->log_msg("[Conductor] Carriage '" + name + "' disconnected.");

    if (had_work)
        reassign_failed_work(failed_chunk);
}

void ConductorServer::reassign_failed_work(const WorkChunk& chunk) {
    if (_local_task_mgr)
        _local_task_mgr->log_msg("[Conductor] Reassigning failed chunk ["
                                  + std::to_string(chunk.range_start) + ", "
                                  + std::to_string(chunk.range_end) + ")");

    std::lock_guard<std::mutex> lock(_carriages_mutex);
    for (auto& [id, c] : _carriages) {
        if (!c.has_work) {
            assign_work(id, chunk);
            return;
        }
    }

    if (_local_task_mgr)
        _local_task_mgr->log_msg("[Conductor] No free carriages. Running failed chunk locally.");
}

// ═══════════════════════════════════════════════════════════════════════
// Work distribution
// ═══════════════════════════════════════════════════════════════════════

void ConductorServer::distribute_task(TaskType type, uint64_t start, uint64_t end) {
    std::lock_guard<std::mutex> lock(_carriages_mutex);

    int num_workers = static_cast<int>(_carriages.size()) + 1;
    auto chunks = split_range(type, start, end, num_workers);
    if (chunks.empty()) return;

    if (_local_task_mgr) {
        _local_task_mgr->log_msg("[Conductor] Distributing "
                                  + std::string(task_key(type))
                                  + " [" + std::to_string(start) + ", " + std::to_string(end)
                                  + ") across " + std::to_string(num_workers) + " workers.");
    }

    // First chunk goes to local Conductor
    if (_local_task_mgr) {
        auto& local = chunks[0];
        auto& tasks = _local_task_mgr->tasks();
        auto it = tasks.find(type);
        if (it != tasks.end()) {
            it->second.start_pos = local.range_start;
            it->second.current_pos = local.range_start;
            it->second.end_pos = local.range_end;
        }
        _local_task_mgr->start_task(type);
    }

    // Remaining chunks go to carriages
    size_t chunk_idx = 1;
    for (auto& [id, c] : _carriages) {
        if (chunk_idx >= chunks.size()) break;
        assign_work(id, chunks[chunk_idx]);
        chunk_idx++;
    }
}

void ConductorServer::assign_work(uint32_t carriage_id, const WorkChunk& chunk) {
    auto it = _carriages.find(carriage_id);
    if (it == _carriages.end()) return;

    WorkAssignment wa;
    wa.task_id = chunk.task_id;
    wa.type = chunk.type;
    wa.range_start = chunk.range_start;
    wa.range_end = chunk.range_end;

    it->second.socket->send_message(wa.serialize());
    it->second.assigned_chunk = chunk;
    it->second.has_work = true;

    if (_local_task_mgr)
        _local_task_mgr->log_msg("[Conductor] Assigned [" + std::to_string(chunk.range_start)
                                  + ", " + std::to_string(chunk.range_end)
                                  + ") to carriage " + std::to_string(carriage_id));
}

// ═══════════════════════════════════════════════════════════════════════
// Status queries
// ═══════════════════════════════════════════════════════════════════════

std::vector<CarriageInfo> ConductorServer::connected_carriages() const {
    std::lock_guard<std::mutex> lock(_carriages_mutex);
    std::vector<CarriageInfo> result;
    result.reserve(_carriages.size());
    for (auto& [id, c] : _carriages)
        result.push_back(c.info);
    return result;
}

size_t ConductorServer::carriage_count() const {
    std::lock_guard<std::mutex> lock(_carriages_mutex);
    return _carriages.size();
}

} // namespace prime
