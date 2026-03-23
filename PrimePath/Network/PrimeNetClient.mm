#import <Foundation/Foundation.h>
#include "PrimeNetClient.hpp"
#include <fstream>
#include <sstream>
#include <random>
#include <map>
#include <chrono>
#include <iomanip>
#include <sys/utsname.h>

namespace primenet {

// ═══════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════

PrimeNetClient::PrimeNetClient(const std::string& data_dir, LogFn log)
    : _data_dir(data_dir), _log(log)
{
    load_state();
}

void PrimeNetClient::set_username(const std::string& user) {
    std::lock_guard<std::mutex> lock(_mutex);
    _state.username = user;
    save_state();
}

// ═══════════════════════════════════════════════════════════════════════
// PrimeNet v5 API -- Register Machine (t=uc)
// ═══════════════════════════════════════════════════════════════════════

bool PrimeNetClient::register_machine() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_state.username.empty()) {
        _log("PrimeNet: username not set. Set your mersenne.org username first.");
        return false;
    }

    if (_state.guid.empty()) {
        _state.guid = generate_guid();
    }

    if (_state.computer_name.empty()) {
        char hostname[256] = {};
        gethostname(hostname, sizeof(hostname));
        _state.computer_name = hostname;
    }

    std::string url = "https://v5.mersenne.org/v5server/"
        "?px=GIMPS&v=0.95&t=uc"
        "&g=" + _state.guid +
        "&hg=" + url_encode(_state.computer_name) +
        "&wg=" + url_encode(_state.computer_name) +
        "&u=" + url_encode(_state.username) +
        "&un=" + url_encode(machine_description()) +
        "&ss=&sh=";

    _log("PrimeNet: registering machine as '" + _state.computer_name +
         "' for user '" + _state.username + "'...");

    std::string response = http_get(url);
    if (response.empty()) {
        _log("PrimeNet: registration failed -- no response from server.");
        return false;
    }

    auto kv = parse_response(response);

    // Check for error
    if (kv.count("pnErrorResult") && kv["pnErrorResult"] != "0") {
        std::string code = kv["pnErrorResult"];
        std::string detail = kv.count("pnErrorDetail") ? kv["pnErrorDetail"] : "(no detail)";
        _log("PrimeNet: registration error " + code + " -- " + detail);
        // Error code 1 means "already registered" which is fine
        if (code == "1") {
            _log("PrimeNet: machine already registered (OK).");
            _state.registered = true;
            save_state();
            return true;
        }
        return false;
    }

    _state.registered = true;
    save_state();
    _log("PrimeNet: registration successful. GUID=" + _state.guid);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// Set Work Preference (t=po) -- request trial factoring
// ═══════════════════════════════════════════════════════════════════════

bool PrimeNetClient::set_work_preference() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (!_state.registered) {
        _log("PrimeNet: not registered. Register first.");
        return false;
    }

    // w=2 = trial factoring
    std::string url = "https://v5.mersenne.org/v5server/"
        "?px=GIMPS&v=0.95&t=po"
        "&g=" + _state.guid +
        "&c=0&w=2"
        "&ss=&sh=";

    _log("PrimeNet: setting work preference to Trial Factoring...");
    std::string response = http_get(url);
    if (response.empty()) {
        _log("PrimeNet: failed to set work preference.");
        return false;
    }

    auto kv = parse_response(response);
    if (kv.count("pnErrorResult") && kv["pnErrorResult"] != "0") {
        _log("PrimeNet: preference error -- " +
             (kv.count("pnErrorDetail") ? kv["pnErrorDetail"] : kv["pnErrorResult"]));
        return false;
    }

    _log("PrimeNet: work preference set to Trial Factoring.");
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// Get Assignment (t=ga)
// ═══════════════════════════════════════════════════════════════════════

Assignment PrimeNetClient::get_assignment() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (!_state.registered) {
        _log("PrimeNet: not registered.");
        return {};
    }

    std::string url = "https://v5.mersenne.org/v5server/"
        "?px=GIMPS&v=0.95&t=ga"
        "&g=" + _state.guid +
        "&c=0&ss=&sh=";

    _log("PrimeNet: requesting trial factoring assignment...");
    std::string response = http_get(url);
    if (response.empty()) {
        _log("PrimeNet: no response from server.");
        return {};
    }

    auto kv = parse_response(response);
    if (kv.count("pnErrorResult") && kv["pnErrorResult"] != "0") {
        _log("PrimeNet: assignment error -- " +
             (kv.count("pnErrorDetail") ? kv["pnErrorDetail"] : kv["pnErrorResult"]));
        return {};
    }

    Assignment a;
    if (kv.count("k")) a.key = kv["k"];
    if (kv.count("n")) a.exponent = std::stoull(kv["n"]);
    if (kv.count("sf")) a.bit_lo = std::stod(kv["sf"]);
    if (kv.count("ef")) a.bit_hi = std::stod(kv["ef"]);

    if (a.exponent > 0) {
        a.valid = true;
        _state.assignments.push_back(a);
        save_state();
        _log("PrimeNet: got assignment -- M" + std::to_string(a.exponent) +
             " TF from " + std::to_string((int)a.bit_lo) +
             " to " + std::to_string((int)a.bit_hi) + " bits");
    } else {
        _log("PrimeNet: server returned no assignment.");
    }

    return a;
}

// ═══════════════════════════════════════════════════════════════════════
// Submit Result (t=ar)
// ═══════════════════════════════════════════════════════════════════════

bool PrimeNetClient::submit_result(const TFResult& result) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (!_state.registered) {
        _log("PrimeNet: not registered.");
        return false;
    }

    // r=1: factor found, r=4: no factor
    int result_type = result.factor_found ? 1 : 4;

    std::string msg;
    if (result.factor_found) {
        msg = "M" + std::to_string(result.exponent) + " has a factor: " + result.factor;
    } else {
        msg = "M" + std::to_string(result.exponent) + " no factor from 2^" +
              std::to_string((int)result.bit_lo) + " to 2^" + std::to_string((int)result.bit_hi);
    }

    std::string akey = result.assignment_key.empty() ? "0" : result.assignment_key;

    std::string url = "https://v5.mersenne.org/v5server/"
        "?px=GIMPS&v=0.95&t=ar"
        "&g=" + _state.guid +
        "&k=" + akey +
        "&r=" + std::to_string(result_type) +
        "&d=1"
        "&n=" + std::to_string(result.exponent) +
        "&sf=" + std::to_string(result.bit_lo) +
        "&ef=" + std::to_string(result.bit_hi);

    if (result.factor_found) {
        url += "&f=" + result.factor;
    }

    url += "&m=" + url_encode(msg) + "&ss=&sh=";

    _log("PrimeNet: submitting result -- " + msg);
    std::string response = http_get(url);
    if (response.empty()) {
        _log("PrimeNet: no response from server.");
        return false;
    }

    auto kv = parse_response(response);
    if (kv.count("pnErrorResult") && kv["pnErrorResult"] != "0") {
        _log("PrimeNet: submit error -- " +
             (kv.count("pnErrorDetail") ? kv["pnErrorDetail"] : kv["pnErrorResult"]));
        return false;
    }

    _log("PrimeNet: result submitted successfully.");

    // Also write to local file
    write_result_json(result);

    // Remove completed assignment
    if (!result.assignment_key.empty()) {
        remove_assignment(result.assignment_key);
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// Convenience: fetch_work (register + get assignment)
// ═══════════════════════════════════════════════════════════════════════

Assignment PrimeNetClient::fetch_work() {
    // Register if needed (unlock mutex between calls)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_state.registered) {
            // Must unlock before calling register_machine which also locks
        }
    }

    if (!_state.registered) {
        if (!register_machine()) return {};
        set_work_preference();
    }

    return get_assignment();
}

// ═══════════════════════════════════════════════════════════════════════
// Local results.json.txt (mfaktc-compatible JSON)
// ═══════════════════════════════════════════════════════════════════════

void PrimeNetClient::write_result_json(const TFResult& result) {
    std::string path = _data_dir + "/results.json.txt";

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
    gmtime_r(&tt, &tm_buf);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_buf);

    std::ostringstream js;
    js << "{\"exponent\":" << result.exponent
       << ",\"worktype\":\"TF\""
       << ",\"status\":\"" << (result.factor_found ? "F" : "NF") << "\""
       << ",\"bitlo\":" << (int)result.bit_lo
       << ",\"bithi\":" << (int)result.bit_hi
       << ",\"rangecomplete\":" << (result.range_complete ? "true" : "false");

    if (result.factor_found) {
        js << ",\"factors\":[\"" << result.factor << "\"]";
    }

    js << ",\"program\":{\"name\":\"PrimePath\",\"version\":\"1.0\",\"kernel\":\"Metal96bit\"}"
       << ",\"timestamp\":\"" << ts << "\""
       << ",\"user\":\"" << _state.username << "\""
       << ",\"computer\":\"" << _state.computer_name << "\"";

    if (!result.assignment_key.empty() && result.assignment_key != "0") {
        js << ",\"aid\":\"" << result.assignment_key << "\"";
    }

    js << "}";

    std::ofstream f(path, std::ios::app);
    if (f.is_open()) {
        f << js.str() << "\n";
        _log("PrimeNet: result written to results.json.txt");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// State Persistence
// ═══════════════════════════════════════════════════════════════════════

void PrimeNetClient::load_state() {
    std::string path = _data_dir + "/primenet_state.txt";
    std::ifstream f(path);
    if (!f.is_open()) return;

    std::string line;
    Assignment current_assignment;
    bool in_assignment = false;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        if (key == "username") _state.username = val;
        else if (key == "guid") _state.guid = val;
        else if (key == "computer") _state.computer_name = val;
        else if (key == "registered") _state.registered = (val == "1");
        else if (key == "assignment_key") {
            if (in_assignment && current_assignment.exponent > 0) {
                current_assignment.valid = true;
                _state.assignments.push_back(current_assignment);
            }
            current_assignment = {};
            current_assignment.key = val;
            in_assignment = true;
        }
        else if (key == "assignment_exponent" && in_assignment) current_assignment.exponent = std::stoull(val);
        else if (key == "assignment_bitlo" && in_assignment) current_assignment.bit_lo = std::stod(val);
        else if (key == "assignment_bithi" && in_assignment) current_assignment.bit_hi = std::stod(val);
    }

    if (in_assignment && current_assignment.exponent > 0) {
        current_assignment.valid = true;
        _state.assignments.push_back(current_assignment);
    }
}

void PrimeNetClient::save_state() {
    std::string path = _data_dir + "/primenet_state.txt";
    std::ofstream f(path);
    if (!f.is_open()) return;

    f << "# PrimePath PrimeNet state\n"
      << "username=" << _state.username << "\n"
      << "guid=" << _state.guid << "\n"
      << "computer=" << _state.computer_name << "\n"
      << "registered=" << (_state.registered ? "1" : "0") << "\n";

    for (auto& a : _state.assignments) {
        f << "assignment_key=" << a.key << "\n"
          << "assignment_exponent=" << a.exponent << "\n"
          << "assignment_bitlo=" << a.bit_lo << "\n"
          << "assignment_bithi=" << a.bit_hi << "\n";
    }
}

void PrimeNetClient::remove_assignment(const std::string& key) {
    auto& as = _state.assignments;
    as.erase(std::remove_if(as.begin(), as.end(),
        [&key](const Assignment& a) { return a.key == key; }), as.end());
    save_state();
}

// ═══════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════

std::string PrimeNetClient::generate_guid() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 15);
    const char hex[] = "0123456789ABCDEF";
    std::string guid;
    guid.reserve(32);
    for (int i = 0; i < 32; i++) guid += hex[dist(gen)];
    return guid;
}

std::string PrimeNetClient::url_encode(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out << c;
        } else {
            out << '%' << std::hex << std::uppercase
                << std::setfill('0') << std::setw(2) << (int)c;
        }
    }
    return out.str();
}

std::string PrimeNetClient::http_get(const std::string& url) {
    @autoreleasepool {
        NSString *nsURL = [NSString stringWithUTF8String:url.c_str()];
        NSURL *reqURL = [NSURL URLWithString:nsURL];
        if (!reqURL) return "";

        NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:reqURL];
        req.HTTPMethod = @"GET";
        req.timeoutInterval = 30.0;
        [req setValue:@"PrimePath/1.0" forHTTPHeaderField:@"User-Agent"];

        __block NSData *responseData = nil;
        __block NSError *responseError = nil;
        __block BOOL done = NO;

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        NSURLSessionDataTask *task = [[NSURLSession sharedSession]
            dataTaskWithRequest:req
            completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
                responseData = data;
                responseError = error;
                done = YES;
                dispatch_semaphore_signal(sem);
            }];
        [task resume];
        dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC));

        if (responseError) {
            _log("PrimeNet HTTP error: " +
                 std::string(responseError.localizedDescription.UTF8String));
            return "";
        }

        if (responseData) {
            return std::string((const char *)responseData.bytes, responseData.length);
        }
        return "";
    }
}

std::map<std::string, std::string> PrimeNetClient::parse_response(const std::string& body) {
    std::map<std::string, std::string> result;
    std::istringstream stream(body);
    std::string line;
    while (std::getline(stream, line)) {
        if (line == "=END=" || line.empty()) continue;
        auto eq = line.find('=');
        if (eq != std::string::npos) {
            result[line.substr(0, eq)] = line.substr(eq + 1);
        }
    }
    return result;
}

std::string PrimeNetClient::machine_description() {
    struct utsname un;
    uname(&un);
    std::string desc = "PrimePath/1.0 Metal GPU ";
    desc += un.machine;  // e.g. "arm64"
    desc += " macOS ";
    desc += un.release;
    return desc;
}

} // namespace primenet
