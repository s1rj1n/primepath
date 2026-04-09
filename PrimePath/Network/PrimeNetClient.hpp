#pragma once
// ═══════════════════════════════════════════════════════════════════════
// PrimeNetClient -- GIMPS / PrimeNet v5 API integration
//
// Handles:
//   - Machine registration (t=uc)
//   - Work preference setting (t=po) for trial factoring
//   - Assignment fetching (t=ga)
//   - Result submission (t=ar) -- factor found or no-factor
//   - Local results.json.txt file writing (mfaktc-compatible format)
//
// API docs: https://v5.mersenne.org/v5design/v5webAPI_0.97.html
// ═══════════════════════════════════════════════════════════════════════

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <mutex>
#include <cstdint>

namespace primenet {

// ── Assignment from PrimeNet ────────────────────────────────────────

struct Assignment {
    std::string key;        // 32-char hex GUID from server
    uint64_t exponent = 0;  // Mersenne exponent p
    double bit_lo = 0;      // already factored to this many bits
    double bit_hi = 0;      // factor up to this many bits
    bool valid = false;
};

// ── Result to submit ────────────────────────────────────────────────

struct TFResult {
    uint64_t exponent;
    double bit_lo;
    double bit_hi;
    bool factor_found;
    std::string factor;     // decimal string if found
    std::string assignment_key;
    bool range_complete;
};

// ── Client state (persisted to primenet_state.txt) ──────────────────

struct ClientState {
    std::string username;       // GIMPS account name (e.g. "s1rj1n")
    std::string guid;           // 32-char hex machine GUID
    std::string computer_name;  // human-readable machine name
    bool registered = false;
    std::vector<Assignment> assignments;
};

// ── Logging callback ────────────────────────────────────────────────

using LogFn = std::function<void(const std::string&)>;

// ═══════════════════════════════════════════════════════════════════════
// PrimeNetClient
// ═══════════════════════════════════════════════════════════════════════

class PrimeNetClient {
public:
    PrimeNetClient(const std::string& data_dir, LogFn log);
    ~PrimeNetClient() = default;

    // ── Configuration ───────────────────────────────────────────────
    void set_username(const std::string& user);
    const std::string& username() const { return _state.username; }
    bool is_registered() const { return _state.registered; }
    const ClientState& state() const { return _state; }

    // ── PrimeNet v5 API operations ──────────────────────────────────

    // Register this machine with PrimeNet (t=uc)
    // Returns true on success
    bool register_machine();

    // Set work preferences to trial factoring (t=po)
    bool set_work_preference();

    // Fetch a trial factoring assignment (t=ga)
    // Returns assignment with valid=true on success
    Assignment get_assignment();

    // Submit a trial factoring result (t=ar)
    bool submit_result(const TFResult& result);

    // ── Local results file ──────────────────────────────────────────

    // Build JSON result line (used for both PrimeNet submission and local file)
    std::string build_result_json(const TFResult& result);

    // Append JSON result to results.json.txt (same JSON as submitted to PrimeNet)
    void write_result_json(const TFResult& result, const std::string& json = "");

    // ── State persistence ───────────────────────────────────────────
    void load_state();
    void save_state();

    // ── Convenience ─────────────────────────────────────────────────

    // Full workflow: register if needed, get assignment, return it
    Assignment fetch_work();

    // How many pending assignments do we have?
    int pending_count() const { return (int)_state.assignments.size(); }

    // Remove a completed assignment
    void remove_assignment(const std::string& key);

private:
    std::string _data_dir;
    ClientState _state;
    LogFn _log;
    std::mutex _mutex;

    // Generate a random 32-char hex GUID
    static std::string generate_guid();

    // URL-encode a string
    static std::string url_encode(const std::string& s);

    // Make an HTTP GET request, return response body
    std::string http_get(const std::string& url);

    // Parse PrimeNet v5 response (key=value lines ending with =END=)
    std::map<std::string, std::string> parse_response(const std::string& body);

    // Get machine description string
    std::string machine_description();
};

} // namespace primenet
