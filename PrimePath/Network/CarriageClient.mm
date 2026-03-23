#include "CarriageClient.hpp"
#import "BonjourService.h"
#import <Foundation/Foundation.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <unistd.h>

namespace prime {

using namespace net;

// ═══════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ═══════════════════════════════════════════════════════════════════════

CarriageClient::CarriageClient(const std::string& data_dir)
    : _data_dir(data_dir)
{
    _gpu.reset(create_best_backend());
}

CarriageClient::~CarriageClient() {
    stop();
}

// ═══════════════════════════════════════════════════════════════════════
// Bonjour discovery
// ═══════════════════════════════════════════════════════════════════════

void CarriageClient::start_discovery() {
    if (_should_run.load()) return;
    _should_run.store(true);

    report_status("Searching for Conductor on local network...");

    BonjourBrowser *browser = [[BonjourBrowser alloc] init];

    __block CarriageClient* weakSelf = this;

    browser.onServiceFound = ^(NSString *hostname, uint16_t port) {
        std::string host = [hostname UTF8String];
        weakSelf->report_status("Found Conductor: " + host + ":" + std::to_string(port));
        weakSelf->connect(host, port);
    };

    browser.onServiceLost = ^(NSString *hostname) {
        std::string host = [hostname UTF8String];
        weakSelf->report_status("Lost Conductor: " + host);
    };

    [browser startBrowsing];
    _bonjour_browser = (__bridge_retained void*)browser;
}

// ═══════════════════════════════════════════════════════════════════════
// Connection
// ═══════════════════════════════════════════════════════════════════════

void CarriageClient::connect(const std::string& host, uint16_t port) {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_connected.load()) return;
        _conductor_host = host;
        _conductor_port = port;
    }

    report_status("Connecting to " + host + ":" + std::to_string(port) + "...");

    if (!_socket.connect_to(host, port)) {
        report_status("Failed to connect to " + host + ":" + std::to_string(port));
        if (_should_run.load() && !_reconnect_thread.joinable()) {
            _reconnect_thread = std::thread([this] { reconnect_loop(); });
        }
        return;
    }

    _connected.store(true);
    send_hello();
    report_status("Connected to Conductor at " + host + ":" + std::to_string(port));

    _recv_thread = std::thread([this] { recv_loop(); });
    _progress_thread = std::thread([this] { progress_loop(); });
}

void CarriageClient::stop() {
    _should_run.store(false);
    _connected.store(false);

    if (_bonjour_browser) {
        BonjourBrowser *browser = (__bridge_transfer BonjourBrowser*)_bonjour_browser;
        [browser stopBrowsing];
        _bonjour_browser = nullptr;
    }

    stop_local_work();
    _socket.close();

    if (_recv_thread.joinable()) _recv_thread.join();
    if (_progress_thread.joinable()) _progress_thread.join();
    if (_reconnect_thread.joinable()) _reconnect_thread.join();

    report_status("Carriage stopped.");
}

// ═══════════════════════════════════════════════════════════════════════
// Receive loop
// ═══════════════════════════════════════════════════════════════════════

void CarriageClient::recv_loop() {
    while (_connected.load() && _should_run.load()) {
        std::string json_body = _socket.recv_message();
        if (json_body.empty()) {
            _connected.store(false);
            report_status("Lost connection to Conductor.");
            stop_local_work();
            if (_should_run.load() && !_reconnect_thread.joinable()) {
                _reconnect_thread = std::thread([this] { reconnect_loop(); });
            }
            return;
        }
        handle_message(json_body);
    }
}

void CarriageClient::handle_message(const std::string& json_body) {
    auto msg_type = message_type_from_json(json_body);

    switch (msg_type) {
        case MessageType::Ping: {
            auto obj = json_parse(json_body);
            auto ping = PingMsg::deserialize(obj);
            PongMsg pong{ping.seq};
            std::lock_guard<std::mutex> lock(_mutex);
            _socket.send_message(pong.serialize());
            break;
        }

        case MessageType::AssignWork: {
            handle_assign_work(json_body);
            break;
        }

        case MessageType::CancelWork: {
            stop_local_work();
            report_status("Work cancelled by Conductor.");
            break;
        }

        default:
            break;
    }
}

void CarriageClient::handle_assign_work(const std::string& json_body) {
    auto obj = json_parse(json_body);
    auto wa = WorkAssignment::deserialize(obj);

    WorkChunk chunk;
    chunk.task_id = wa.task_id;
    chunk.type = wa.type;
    chunk.range_start = wa.range_start;
    chunk.range_end = wa.range_end;

    report_status("Received work: " + std::string(task_key(chunk.type))
                  + " [" + std::to_string(chunk.range_start) + ", "
                  + std::to_string(chunk.range_end) + ")");

    if (_on_work)
        _on_work(chunk);

    start_local_work(chunk);
}

// ═══════════════════════════════════════════════════════════════════════
// Progress reporting
// ═══════════════════════════════════════════════════════════════════════

void CarriageClient::progress_loop() {
    while (_connected.load() && _should_run.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!_connected.load() || !_should_run.load()) break;
        if (_working.load()) send_progress();
    }
}

void CarriageClient::send_progress() {
    if (!_task_mgr) return;

    ProgressMsg pm;
    pm.task_id = _current_chunk.task_id;

    auto& tasks = _task_mgr->tasks();
    auto it = tasks.find(_current_chunk.type);
    if (it != tasks.end()) {
        pm.current_pos = it->second.current_pos;
        pm.tested = it->second.tested_count;
        pm.rate = it->second.rate;
    }

    std::lock_guard<std::mutex> lock(_mutex);
    _socket.send_message(pm.serialize());
}

// ═══════════════════════════════════════════════════════════════════════
// Hello
// ═══════════════════════════════════════════════════════════════════════

void CarriageClient::send_hello() {
    CarriageInfo info;
    info.hostname = local_hostname();
    info.cores = static_cast<int>(cpu_core_count());
    info.gpu_name = (_gpu && _gpu->available()) ? _gpu->name() : "none";
    info.version = "0.5";
    info.connected = true;

    _socket.send_message(info.serialize());
}

// ═══════════════════════════════════════════════════════════════════════
// Discovery / WorkDone
// ═══════════════════════════════════════════════════════════════════════

void CarriageClient::send_discovery(const DiscoveryMsg& disc) {
    std::lock_guard<std::mutex> lock(_mutex);
    _socket.send_message(disc.serialize());
}

void CarriageClient::send_work_done() {
    WorkDoneMsg wm;
    wm.task_id = _current_chunk.task_id;
    wm.range_start = _current_chunk.range_start;
    wm.range_end = _current_chunk.range_end;

    if (_task_mgr) {
        auto& tasks = _task_mgr->tasks();
        auto it = tasks.find(_current_chunk.type);
        if (it != tasks.end()) {
            wm.tested = it->second.tested_count;
            wm.found = it->second.found_count;
        }
    }

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _socket.send_message(wm.serialize());
    }

    _working.store(false);
    report_status("Work chunk completed.");
}

// ═══════════════════════════════════════════════════════════════════════
// Local computation
// ═══════════════════════════════════════════════════════════════════════

void CarriageClient::start_local_work(const WorkChunk& chunk) {
    stop_local_work();

    _current_chunk = chunk;
    _working.store(true);

    _task_mgr = std::make_unique<TaskManager>(_data_dir);
    _task_mgr->set_gpu(_gpu.get());

    _task_mgr->init_defaults();

    auto& tasks = _task_mgr->tasks();
    auto it = tasks.find(chunk.type);
    if (it != tasks.end()) {
        it->second.start_pos = chunk.range_start;
        it->second.current_pos = chunk.range_start;
        it->second.end_pos = chunk.range_end;
    }

    _task_mgr->start_task(chunk.type);

    // Monitor for completion
    std::thread([this, chunk_type = chunk.type] {
        while (_working.load() && _should_run.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (!_task_mgr) break;

            auto& tasks = _task_mgr->tasks();
            auto it = tasks.find(chunk_type);
            if (it == tasks.end()) break;

            if (it->second.end_pos > 0 && it->second.current_pos >= it->second.end_pos) {
                send_work_done();
                break;
            }
        }
    }).detach();
}

void CarriageClient::stop_local_work() {
    _working.store(false);
    if (_task_mgr) {
        _task_mgr->stop_all();
        _task_mgr.reset();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Auto-reconnect
// ═══════════════════════════════════════════════════════════════════════

void CarriageClient::reconnect_loop() {
    while (_should_run.load() && !_connected.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        if (!_should_run.load()) break;

        std::string host;
        uint16_t port;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            host = _conductor_host;
            port = _conductor_port;
        }

        if (host.empty() || port == 0) continue;

        report_status("Reconnecting to " + host + ":" + std::to_string(port) + "...");

        if (_socket.connect_to(host, port)) {
            _connected.store(true);
            send_hello();
            report_status("Reconnected to Conductor.");

            if (_recv_thread.joinable()) _recv_thread.join();
            _recv_thread = std::thread([this] { recv_loop(); });

            if (_progress_thread.joinable()) _progress_thread.join();
            _progress_thread = std::thread([this] { progress_loop(); });

            return;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Machine info helpers
// ═══════════════════════════════════════════════════════════════════════

std::string CarriageClient::local_hostname() {
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0) return std::string(buf);
    return "unknown";
}

uint32_t CarriageClient::cpu_core_count() {
    int count = 0;
    size_t size = sizeof(count);
    sysctlbyname("hw.ncpu", &count, &size, nullptr, 0);
    return (count > 0) ? static_cast<uint32_t>(count) : 1;
}

uint64_t CarriageClient::total_memory_mb() {
    int64_t mem = 0;
    size_t size = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &size, nullptr, 0);
    return static_cast<uint64_t>(mem / (1024 * 1024));
}

// ═══════════════════════════════════════════════════════════════════════
// Status reporting
// ═══════════════════════════════════════════════════════════════════════

std::string CarriageClient::conductor_host() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _conductor_host;
}

uint16_t CarriageClient::conductor_port() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _conductor_port;
}

void CarriageClient::report_status(const std::string& msg) {
    if (_on_status) _on_status(msg);
}

} // namespace prime
