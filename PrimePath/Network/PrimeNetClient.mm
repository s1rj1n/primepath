#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#include "PrimeNetClient.hpp"
#include <fstream>
#include <sstream>
#include <random>
#include <map>
#include <chrono>
#include <iomanip>
#include <sys/utsname.h>
#include <sys/sysctl.h>
#include <zlib.h>

namespace primenet {

// Forward declaration — defined near the worktodo.txt section
static bool parse_worktodo_line(const std::string& raw, Assignment& out);

// Returns Apple Silicon GPU core count from IORegistry, or 0 if unavailable.
static int detect_gpu_core_count() {
    int cores = 0;
    io_iterator_t iter;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
            IOServiceMatching("AGXAccelerator"), &iter) != KERN_SUCCESS)
        return 0;
    io_object_t svc;
    while ((svc = IOIteratorNext(iter))) {
        CFTypeRef prop = IORegistryEntrySearchCFProperty(svc, kIOServicePlane,
            CFSTR("gpu-core-count"), kCFAllocatorDefault,
            kIORegistryIterateRecursively | kIORegistryIterateParents);
        if (prop) {
            if (CFGetTypeID(prop) == CFNumberGetTypeID())
                CFNumberGetValue((CFNumberRef)prop, kCFNumberIntType, &cores);
            CFRelease(prop);
        }
        IOObjectRelease(svc);
        if (cores > 0) break;
    }
    IOObjectRelease(iter);
    return cores;
}

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

    // Hardware GUID: MD5-like hash of hostname
    std::string hw_guid = _state.guid;  // reuse GUID for hardware ID

    // Get system info for registration
    int64_t ram_mb = 0;
    int ncpu = 1;
    {
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        uint64_t memsize = 0;
        size_t len = sizeof(memsize);
        if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0)
            ram_mb = (int64_t)(memsize / (1024 * 1024));
        ncpu = (int)[[NSProcessInfo processInfo] processorCount];
    }

    // Dynamic system info for registration
    std::string mdesc = machine_description();
    char chip_buf[256] = {};
    size_t chip_len = sizeof(chip_buf);
    sysctlbyname("machdep.cpu.brand_string", chip_buf, &chip_len, NULL, 0);
    std::string chip_name = chip_buf[0] ? chip_buf : "Apple Silicon";

    std::string url = "https://v5.mersenne.org/v5server/"
        "?px=GIMPS&v=0.95&t=uc"
        "&g=" + _state.guid +
        "&hg=" + hw_guid +
        "&wg="
        "&a=" + url_encode("macOS,PrimePath,1.2.1 (github.com/s1rj1n/primepath)") +
        "&c=" + url_encode(chip_name + " (Metal GPU)") +
        "&f=" + url_encode("Metal,NEON,AES") +
        "&L1=192&L2=4096&L3=0"
        "&np=" + std::to_string(ncpu) +
        "&hp=1"
        "&m=" + std::to_string(ram_mb) +
        "&s=3200"
        "&h=24&r=1000"
        "&u=" + url_encode(_state.username) +
        "&cn=" + url_encode(_state.computer_name) +
        "&ss=19191919&sh=ABCDABCDABCDABCDABCDABCDABCDABCD";

    _log("PrimeNet: registering machine as '" + _state.computer_name +
         "' for user '" + _state.username + "'...");
    _log("PrimeNet: URL=" + url);

    std::string response = http_get(url);
    _log("PrimeNet: raw response: [" + response + "]");
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
        // Error code 1 or already registered is fine
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
        "&ss=19191919&sh=ABCDABCDABCDABCDABCDABCDABCDABCD";

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

    // Prefer worktodo.txt (AutoPrimeNet) if it has entries.
    // This lets users run PrimePath through AutoPrimeNet without
    // needing to register the machine directly with PrimeNet.
    {
        std::string wt_path = _data_dir + "/worktodo.txt";
        std::ifstream probe(wt_path);
        if (probe.is_open()) {
            probe.close();
            // Reuse private logic by re-reading (mutex already held,
            // don't call the public next_worktodo which may re-lock).
            std::ifstream f(wt_path);
            std::string line;
            while (std::getline(f, line)) {
                Assignment a;
                if (parse_worktodo_line(line, a)) {
                    _log("PrimeNet: using worktodo.txt assignment -- M" +
                         std::to_string(a.exponent) + " TF [" +
                         std::to_string((int)a.bit_lo) + "," +
                         std::to_string((int)a.bit_hi) + "]");
                    return a;
                }
            }
        }
    }

    if (!_state.registered) {
        _log("PrimeNet: not registered and no worktodo.txt entries.");
        return {};
    }

    std::string url = "https://v5.mersenne.org/v5server/"
        "?px=GIMPS&v=0.95&t=ga"
        "&g=" + _state.guid +
        "&c=0&ss=19191919&sh=ABCDABCDABCDABCDABCDABCDABCDABCD";

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

    // Build JSON result line (used for both &m= and results.json.txt)
    std::string json = build_result_json(result);

    // AutoPrimeNet mode: if not registered, just append to results.json.txt
    // and remove the completed line from worktodo.txt. AutoPrimeNet will
    // pick it up and submit to mersenne.org on its next cycle.
    if (!_state.registered) {
        _log("PrimeNet: not registered — writing result for AutoPrimeNet to submit.");
        write_result_json(result, json);

        Assignment done;
        done.key = result.assignment_key;
        done.exponent = result.exponent;
        done.bit_lo = result.bit_lo;
        done.bit_hi = result.bit_hi;

        // remove_worktodo takes the mutex via the public API; we already
        // hold it, so do the work inline.
        std::string wt_path = _data_dir + "/worktodo.txt";
        std::ifstream in(wt_path);
        if (in.is_open()) {
            std::vector<std::string> keep;
            std::string line;
            bool removed = false;
            while (std::getline(in, line)) {
                Assignment a;
                if (!removed && parse_worktodo_line(line, a)) {
                    bool match =
                        (!done.key.empty() && a.key == done.key) ||
                        (done.key.empty() &&
                         a.exponent == done.exponent &&
                         (int)a.bit_lo == (int)done.bit_lo &&
                         (int)a.bit_hi == (int)done.bit_hi);
                    if (match) { removed = true; continue; }
                }
                keep.push_back(line);
            }
            in.close();
            std::string tmp = wt_path + ".tmp";
            std::ofstream out(tmp, std::ios::trunc);
            if (out.is_open()) {
                for (const auto& l : keep) out << l << "\n";
                out.close();
                std::rename(tmp.c_str(), wt_path.c_str());
                if (removed)
                    _log("PrimeNet: removed completed assignment from worktodo.txt");
            }
        }
        return true;
    }

    // r=1: factor found, r=4: no factor
    int result_type = result.factor_found ? 1 : 4;
    std::string akey = result.assignment_key.empty() ? "0" : result.assignment_key;

    std::string url = "https://v5.mersenne.org/v5server/"
        "?px=GIMPS&v=0.95&t=ar"
        "&g=" + _state.guid +
        "&k=" + akey +
        "&r=" + std::to_string(result_type) +
        "&d=1"
        "&n=" + std::to_string(result.exponent) +
        "&sf=" + std::to_string((int)result.bit_lo) +
        "&ef=" + std::to_string((int)result.bit_hi);

    if (result.factor_found && !result.factors.empty()) {
        url += "&f=" + result.factors.front();
    }

    url += "&m=" + url_encode(json) + "&ss=19191919&sh=ABCDABCDABCDABCDABCDABCDABCDABCD";

    _log("PrimeNet: submitting JSON result: " + json);
    _log("PrimeNet: URL=" + url);
    std::string response = http_get(url);
    _log("PrimeNet: raw response: [" + response + "]");
    if (response.empty()) {
        _log("PrimeNet: no response from server — check network connectivity.");
        return false;
    }

    auto kv = parse_response(response);
    if (kv.count("pnErrorResult") && kv["pnErrorResult"] != "0") {
        _log("PrimeNet: submit error -- " +
             (kv.count("pnErrorDetail") ? kv["pnErrorDetail"] : kv["pnErrorResult"]));
        return false;
    }

    _log("PrimeNet: result submitted successfully.");

    // Also write same JSON to local file
    write_result_json(result, json);

    // Remove completed assignment from in-memory state
    if (!result.assignment_key.empty()) {
        remove_assignment(result.assignment_key);
    }

    // Also remove it from worktodo.txt if the assignment came from there
    {
        std::string wt_path = _data_dir + "/worktodo.txt";
        std::ifstream in(wt_path);
        if (in.is_open()) {
            std::vector<std::string> keep;
            std::string line;
            bool removed = false;
            while (std::getline(in, line)) {
                Assignment a;
                if (!removed && parse_worktodo_line(line, a)) {
                    bool match =
                        (!result.assignment_key.empty() && a.key == result.assignment_key) ||
                        (result.assignment_key.empty() &&
                         a.exponent == result.exponent &&
                         (int)a.bit_lo == (int)result.bit_lo &&
                         (int)a.bit_hi == (int)result.bit_hi);
                    if (match) { removed = true; continue; }
                }
                keep.push_back(line);
            }
            in.close();
            if (removed) {
                std::string tmp = wt_path + ".tmp";
                std::ofstream out(tmp, std::ios::trunc);
                if (out.is_open()) {
                    for (const auto& l : keep) out << l << "\n";
                    out.close();
                    std::rename(tmp.c_str(), wt_path.c_str());
                    _log("PrimeNet: removed completed assignment from worktodo.txt");
                }
            }
        }
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
// JSON result builder (shared by submit_result and write_result_json)
// ═══════════════════════════════════════════════════════════════════════

std::string PrimeNetClient::build_result_json(const TFResult& result) {
    // UTC timestamp
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
    gmtime_r(&tt, &tm_buf);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_buf);

    // OS info
    struct utsname un;
    uname(&un);

    // Hardware info
    char chip[256] = {};
    size_t chip_len = sizeof(chip);
    sysctlbyname("machdep.cpu.brand_string", chip, &chip_len, NULL, 0);
    std::string chip_str = chip[0] ? chip : "Apple Silicon";

    int ncpu = (int)[[NSProcessInfo processInfo] processorCount];
    int perf_cores = 0, eff_cores = 0;
    size_t sz = sizeof(int);
    sysctlbyname("hw.perflevel0.logicalcpu", &perf_cores, &sz, NULL, 0);
    sz = sizeof(int);
    sysctlbyname("hw.perflevel1.logicalcpu", &eff_cores, &sz, NULL, 0);

    uint64_t memsize = 0;
    sz = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &sz, NULL, 0);
    int ram_gb = (int)(memsize / (1024ULL * 1024 * 1024));

    int gpu_cores = detect_gpu_core_count();

    // Sort factors ascending for JSON and checksum
    std::vector<std::string> sorted_factors = result.factors;
    std::sort(sorted_factors.begin(), sorted_factors.end(),
        [](const std::string& a, const std::string& b) {
            if (a.size() != b.size()) return a.size() < b.size();
            return a < b;
        });

    std::vector<std::string> sorted_known = result.known_factors;
    std::sort(sorted_known.begin(), sorted_known.end(),
        [](const std::string& a, const std::string& b) {
            if (a.size() != b.size()) return a.size() < b.size();
            return a < b;
        });

    std::ostringstream js;
    js << "{\"timestamp\":\"" << ts << "\""
       << ",\"exponent\":" << result.exponent
       << ",\"worktype\":\"TF\""
       << ",\"status\":\"" << (result.factor_found ? "F" : "NF") << "\""
       << ",\"bitlo\":" << (int)result.bit_lo
       << ",\"bithi\":" << (int)result.bit_hi
       << ",\"rangecomplete\":" << (result.range_complete ? "true" : "false");

    if (result.factor_found && !sorted_factors.empty()) {
        js << ",\"factors\":[";
        for (size_t i = 0; i < sorted_factors.size(); i++) {
            if (i > 0) js << ",";
            js << "\"" << sorted_factors[i] << "\"";
        }
        js << "]";
    }

    if (!sorted_known.empty()) {
        js << ",\"known-factors\":[";
        for (size_t i = 0; i < sorted_known.size(); i++) {
            if (i > 0) js << ",";
            js << "\"" << sorted_known[i] << "\"";
        }
        js << "]";
    }

    js << ",\"program\":{\"name\":\"PrimePath\",\"version\":\"1.2.1\",\"kernel\":\"Metal96bit\"}"
       << ",\"os\":{\"os\":\"macOS\",\"version\":\"" << un.release << "\",\"architecture\":\"ARM_64\"}"
       << ",\"user\":\"" << _state.username << "\""
       << ",\"computer\":\"" << _state.computer_name << "\"";

    // Include AID only if non-empty, non-"0", and not all zeros
    auto aid_is_meaningful = [](const std::string& s) {
        if (s.empty() || s == "0") return false;
        for (char c : s) if (c != '0') return true;
        return false;
    };
    if (aid_is_meaningful(result.assignment_key)) {
        js << ",\"aid\":\"" << result.assignment_key << "\"";
    }

    js << ",\"hardware\":{\"chip\":\"" << chip_str << "\""
       << ",\"cpu_cores\":" << ncpu;
    if (perf_cores > 0)
        js << ",\"cpu_p_cores\":" << perf_cores
           << ",\"cpu_e_cores\":" << eff_cores;
    if (gpu_cores > 0)
        js << ",\"gpu_cores\":" << gpu_cores;
    js << ",\"cpu_ram_gb\":" << ram_gb
       << ",\"gpu_ram_gb\":" << ram_gb
       << "}";

    // CRC32 checksum (anti-tampering)
    // Build semicolon-separated string per James's spec:
    // exponent;worktype;factors;known-factors;bitlo;bithi;rangecomplete;
    // fft-length;error-code;program.name;program.version;program.kernel;
    // program.details;os.os;os.architecture;timestamp
    {
        std::ostringstream ck;
        ck << result.exponent << ";TF;";
        // factors: comma-separated ascending
        for (size_t i = 0; i < sorted_factors.size(); i++) {
            if (i > 0) ck << ",";
            ck << sorted_factors[i];
        }
        ck << ";";
        // known-factors: comma-separated ascending
        for (size_t i = 0; i < sorted_known.size(); i++) {
            if (i > 0) ck << ",";
            ck << sorted_known[i];
        }
        ck << ";"
           << (int)result.bit_lo << ";" << (int)result.bit_hi << ";"
           << (result.range_complete ? "1" : "0") << ";"
           << ";" // fft-length (empty for TF)
           << ";" // error-code (empty for TF)
           << "PrimePath;1.2.1;Metal96bit;"
           << ";" // program.details (empty)
           << "macOS;ARM_64;"
           << ts;

        std::string ck_str = ck.str();
        uLong crc = crc32(0L, Z_NULL, 0);
        crc = crc32(crc, (const Bytef*)ck_str.data(), (uInt)ck_str.size());

        char hex[9];
        snprintf(hex, sizeof(hex), "%08lX", (unsigned long)crc);
        js << ",\"checksum\":{\"version\":1,\"checksum\":\"" << hex << "\"}";
    }

    js << "}";
    return js.str();
}

// ═══════════════════════════════════════════════════════════════════════
// Local results.json.txt (same JSON as submitted to PrimeNet)
// ═══════════════════════════════════════════════════════════════════════

void PrimeNetClient::write_result_json(const TFResult& result, const std::string& json) {
    std::string path = _data_dir + "/results.json.txt";
    std::string line = json.empty() ? build_result_json(result) : json;

    std::ofstream f(path, std::ios::app);
    if (f.is_open()) {
        f << line << "\n";
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
// worktodo.txt reader (AutoPrimeNet / mfaktc interop)
// ═══════════════════════════════════════════════════════════════════════
//
// Lines accepted:
//   Factor=<AID>,<exponent>,<bitlo>,<bithi>      (mfaktc/mfakto, AutoPrimeNet)
//   Factor=<exponent>,<bitlo>,<bithi>            (AID omitted)
// Blank lines and lines starting with '#', ';', or "//" are ignored.

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static bool parse_worktodo_line(const std::string& raw, Assignment& out) {
    std::string line = trim(raw);
    if (line.empty()) return false;
    if (line[0] == '#' || line[0] == ';') return false;
    if (line.size() >= 2 && line[0] == '/' && line[1] == '/') return false;

    // Must start with "Factor="
    const std::string prefix = "Factor=";
    if (line.compare(0, prefix.size(), prefix) != 0) return false;
    std::string rest = line.substr(prefix.size());

    // Extract known-factors if trailing quoted string present:
    // Factor=AID,exp,lo,hi,"factor1,factor2"
    std::vector<std::string> known;
    auto qstart = rest.find('"');
    if (qstart != std::string::npos) {
        auto qend = rest.find('"', qstart + 1);
        if (qend != std::string::npos) {
            std::string kf_str = rest.substr(qstart + 1, qend - qstart - 1);
            std::istringstream kfs(kf_str);
            std::string kf;
            while (std::getline(kfs, kf, ',')) {
                std::string t = trim(kf);
                if (!t.empty()) known.push_back(t);
            }
        }
        // Remove the quoted portion (and preceding comma) from rest
        size_t comma = (qstart > 0 && rest[qstart - 1] == ',') ? qstart - 1 : qstart;
        rest = rest.substr(0, comma);
    }

    // Split remaining on commas
    std::vector<std::string> parts;
    std::string tok;
    std::istringstream ss(rest);
    while (std::getline(ss, tok, ',')) parts.push_back(trim(tok));

    if (parts.size() < 3) return false;

    // Detect AID: 32 hex chars or "N/A"
    bool has_aid = false;
    if (parts.size() >= 4) {
        const std::string& p0 = parts[0];
        if (p0 == "N/A") {
            // N/A = no assignment, skip AID entirely
            has_aid = true;
            out.key = "";
        } else if (p0.size() == 32) {
            bool all_hex = true;
            for (char c : p0) {
                if (!isxdigit((unsigned char)c)) { all_hex = false; break; }
            }
            if (all_hex) {
                has_aid = true;
                out.key = p0;
            }
        }
    }

    size_t idx = has_aid ? 1 : 0;
    if (!has_aid) out.key = "";

    if (idx + 2 >= parts.size()) return false;
    try {
        out.exponent = std::stoull(parts[idx]);
        out.bit_lo = std::stod(parts[idx + 1]);
        out.bit_hi = std::stod(parts[idx + 2]);
    } catch (...) {
        return false;
    }
    out.known_factors = known;
    out.valid = true;
    return true;
}

std::vector<Assignment> PrimeNetClient::read_worktodo() {
    std::vector<Assignment> out;
    std::string path = _data_dir + "/worktodo.txt";
    std::ifstream f(path);
    if (!f.is_open()) return out;
    std::string line;
    while (std::getline(f, line)) {
        Assignment a;
        if (parse_worktodo_line(line, a)) out.push_back(a);
    }
    return out;
}

bool PrimeNetClient::has_worktodo() {
    return !read_worktodo().empty();
}

Assignment PrimeNetClient::next_worktodo() {
    auto entries = read_worktodo();
    if (entries.empty()) {
        Assignment a;
        a.valid = false;
        return a;
    }
    _log("PrimeNet: loaded assignment from worktodo.txt: exp=" +
         std::to_string(entries.front().exponent) +
         " [" + std::to_string((int)entries.front().bit_lo) +
         "," + std::to_string((int)entries.front().bit_hi) + "]");
    return entries.front();
}

void PrimeNetClient::remove_worktodo(const Assignment& done) {
    std::string path = _data_dir + "/worktodo.txt";
    std::ifstream f(path);
    if (!f.is_open()) return;

    std::vector<std::string> keep;
    std::string line;
    bool removed = false;
    while (std::getline(f, line)) {
        Assignment a;
        if (!removed && parse_worktodo_line(line, a)) {
            bool match = false;
            if (!done.key.empty() && a.key == done.key) {
                match = true;
            } else if (done.key.empty() &&
                       a.exponent == done.exponent &&
                       (int)a.bit_lo == (int)done.bit_lo &&
                       (int)a.bit_hi == (int)done.bit_hi) {
                match = true;
            }
            if (match) {
                removed = true;
                continue; // drop this line
            }
        }
        keep.push_back(line);
    }
    f.close();

    // Atomic rewrite via temp file
    std::string tmp = path + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out.is_open()) return;
    for (const auto& l : keep) out << l << "\n";
    out.close();
    std::rename(tmp.c_str(), path.c_str());

    if (removed) {
        _log("PrimeNet: removed completed assignment from worktodo.txt");
    }
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
        if (!reqURL) {
            _log("PrimeNet: invalid URL.");
            return "";
        }

        NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:reqURL];
        req.HTTPMethod = @"GET";
        req.timeoutInterval = 45.0;  // Longer than semaphore so NSURLSession always finishes first
        [req setValue:@"PrimePath/1.2.1" forHTTPHeaderField:@"User-Agent"];

        // Use heap-allocated storage so the completion handler is safe even if
        // the semaphore times out (shouldn't happen since request timeout < sem timeout)
        __block NSData * __strong capturedData = nil;
        __block NSError * __strong capturedError = nil;
        __block NSInteger capturedStatus = 0;

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        NSURLSessionDataTask *task = [[NSURLSession sharedSession]
            dataTaskWithRequest:req
            completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
                capturedData = data;
                capturedError = error;
                if ([response isKindOfClass:[NSHTTPURLResponse class]])
                    capturedStatus = ((NSHTTPURLResponse *)response).statusCode;
                dispatch_semaphore_signal(sem);
            }];
        [task resume];

        long waitResult = dispatch_semaphore_wait(sem,
            dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC));

        if (waitResult != 0) {
            [task cancel];
            _log("PrimeNet: request timed out (60s).");
            return "";
        }

        if (capturedError) {
            _log("PrimeNet HTTP error: " +
                 std::string(capturedError.localizedDescription.UTF8String));
            return "";
        }

        if (capturedStatus != 0 && capturedStatus != 200) {
            _log("PrimeNet HTTP status: " + std::to_string((long)capturedStatus));
        }

        if (!capturedData) {
            _log("PrimeNet: no data received from server.");
            return "";
        }

        return std::string((const char *)capturedData.bytes, capturedData.length);
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

    // Get chip name (e.g. "Apple M5")
    char chip[256] = {};
    size_t chip_len = sizeof(chip);
    sysctlbyname("machdep.cpu.brand_string", chip, &chip_len, NULL, 0);
    std::string chip_str = chip[0] ? chip : un.machine;

    // CPU cores (performance + efficiency)
    int ncpu = (int)[[NSProcessInfo processInfo] processorCount];
    int perf_cores = 0, eff_cores = 0;
    size_t sz = sizeof(int);
    sysctlbyname("hw.perflevel0.logicalcpu", &perf_cores, &sz, NULL, 0);
    sz = sizeof(int);
    sysctlbyname("hw.perflevel1.logicalcpu", &eff_cores, &sz, NULL, 0);

    // RAM
    uint64_t memsize = 0;
    sz = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &sz, NULL, 0);
    int ram_gb = (int)(memsize / (1024ULL * 1024 * 1024));

    // GPU cores (Metal) via IORegistry
    int gpu_cores = detect_gpu_core_count();

    std::ostringstream desc;
    desc << "PrimePath/1.2.1 | " << chip_str
         << " | " << ncpu << " CPU";
    if (perf_cores > 0)
        desc << " (" << perf_cores << "P+" << eff_cores << "E)";
    if (gpu_cores > 0)
        desc << " " << gpu_cores << " GPU cores";
    desc << " | " << ram_gb << "GB RAM"
         << " | Metal fused sieve+modexp"
         << " | macOS " << un.release;
    return desc.str();
}

} // namespace primenet
