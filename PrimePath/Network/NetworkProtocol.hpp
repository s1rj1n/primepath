#pragma once
#include "../TaskManager.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace prime { namespace net {

// =========================================================================
// Default port for Conductor <-> Carriage communication
// =========================================================================
static constexpr uint16_t DEFAULT_PORT = 9807;

// =========================================================================
// Message types exchanged between Conductor and Carriage
// =========================================================================
enum class MessageType : uint8_t {
    // Conductor -> Carriage
    AssignWork  = 0x01,
    CancelWork  = 0x02,
    Ping        = 0x03,
    Hello       = 0x04,   // initial handshake from conductor

    // Carriage -> Conductor
    Progress    = 0x10,
    DiscoveryReport = 0x11,
    WorkDone    = 0x12,
    Pong        = 0x13,
};

inline const char* msg_type_name(MessageType t) {
    switch (t) {
        case MessageType::AssignWork:      return "AssignWork";
        case MessageType::CancelWork:      return "CancelWork";
        case MessageType::Ping:            return "Ping";
        case MessageType::Hello:           return "Hello";
        case MessageType::Progress:        return "Progress";
        case MessageType::DiscoveryReport: return "Discovery";
        case MessageType::WorkDone:        return "WorkDone";
        case MessageType::Pong:            return "Pong";
    }
    return "Unknown";
}

// =========================================================================
// Framing: 4-byte big-endian length prefix + JSON body
// =========================================================================

inline void encode_length(uint32_t len, uint8_t out[4]) {
    out[0] = (len >> 24) & 0xFF;
    out[1] = (len >> 16) & 0xFF;
    out[2] = (len >>  8) & 0xFF;
    out[3] =  len        & 0xFF;
}

inline uint32_t decode_length(const uint8_t buf[4]) {
    return (static_cast<uint32_t>(buf[0]) << 24)
         | (static_cast<uint32_t>(buf[1]) << 16)
         | (static_cast<uint32_t>(buf[2]) <<  8)
         |  static_cast<uint32_t>(buf[3]);
}

// =========================================================================
// Minimal hand-rolled JSON helpers (no external deps)
//
// Messages are flat key-value objects.  Values are strings or numbers.
// This is intentionally minimal -- enough for the protocol, not general.
// =========================================================================

// Escape a string for JSON output
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// Remove surrounding whitespace
inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Unescape a JSON string value (minimal: handles \\, \", \n, \r, \t)
inline std::string json_unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case '"':  out += '"';  ++i; break;
                case '\\': out += '\\'; ++i; break;
                case 'n':  out += '\n'; ++i; break;
                case 'r':  out += '\r'; ++i; break;
                case 't':  out += '\t'; ++i; break;
                default:   out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

// Simple flat JSON object: map of string -> string
// Numbers are stored as their string representation.
using JsonObj = std::vector<std::pair<std::string, std::string>>;

inline std::string json_serialize(const JsonObj& obj) {
    std::ostringstream os;
    os << "{";
    for (size_t i = 0; i < obj.size(); ++i) {
        if (i > 0) os << ",";
        os << "\"" << json_escape(obj[i].first) << "\":";
        // Try to detect numeric values -- emit without quotes
        const auto& v = obj[i].second;
        bool is_num = !v.empty();
        bool has_dot = false;
        for (size_t j = 0; j < v.size() && is_num; ++j) {
            char c = v[j];
            if (c == '-' && j == 0) continue;
            if (c == '.' && !has_dot) { has_dot = true; continue; }
            if (!std::isdigit(static_cast<unsigned char>(c))) is_num = false;
        }
        if (v == "true" || v == "false") {
            os << v;
        } else if (is_num && !v.empty()) {
            os << v;
        } else {
            os << "\"" << json_escape(v) << "\"";
        }
    }
    os << "}";
    return os.str();
}

// Minimal JSON parser for flat objects.  Returns key-value pairs.
inline JsonObj json_parse(const std::string& json) {
    JsonObj result;
    size_t pos = json.find('{');
    if (pos == std::string::npos) return result;
    ++pos;

    auto skip_ws = [&]() {
        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    };

    auto read_string = [&]() -> std::string {
        skip_ws();
        if (pos >= json.size() || json[pos] != '"')
            throw std::runtime_error("json_parse: expected '\"'");
        ++pos;
        std::string s;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                s += json[pos]; s += json[pos + 1];
                pos += 2;
            } else {
                s += json[pos++];
            }
        }
        if (pos < json.size()) ++pos; // skip closing "
        return json_unescape(s);
    };

    auto read_value = [&]() -> std::string {
        skip_ws();
        if (pos >= json.size()) return "";
        if (json[pos] == '"') return read_string();
        // number, bool, null -- read until , or }
        size_t start = pos;
        while (pos < json.size() && json[pos] != ',' && json[pos] != '}') ++pos;
        return trim(json.substr(start, pos - start));
    };

    while (pos < json.size()) {
        skip_ws();
        if (pos >= json.size() || json[pos] == '}') break;
        if (json[pos] == ',') { ++pos; continue; }
        std::string key = read_string();
        skip_ws();
        if (pos < json.size() && json[pos] == ':') ++pos;
        std::string val = read_value();
        result.push_back({key, val});
    }
    return result;
}

// Helper to look up a value in a JsonObj
inline std::string json_get(const JsonObj& obj, const std::string& key,
                             const std::string& def = "") {
    for (auto& kv : obj)
        if (kv.first == key) return kv.second;
    return def;
}

inline uint64_t json_get_u64(const JsonObj& obj, const std::string& key, uint64_t def = 0) {
    std::string v = json_get(obj, key);
    if (v.empty()) return def;
    return std::stoull(v);
}

inline double json_get_f64(const JsonObj& obj, const std::string& key, double def = 0.0) {
    std::string v = json_get(obj, key);
    if (v.empty()) return def;
    return std::stod(v);
}

inline bool json_get_bool(const JsonObj& obj, const std::string& key, bool def = false) {
    std::string v = json_get(obj, key);
    if (v == "true") return true;
    if (v == "false") return false;
    return def;
}

// =========================================================================
// WorkAssignment -- a unit of work sent from Conductor to Carriage
// =========================================================================

struct WorkAssignment {
    std::string task_id;
    TaskType    type        = TaskType::GeneralPrime;
    uint64_t    range_start = 0;
    uint64_t    range_end   = 0;

    std::string serialize() const {
        JsonObj obj = {
            {"msg",         std::to_string(static_cast<uint8_t>(MessageType::AssignWork))},
            {"task_id",     task_id},
            {"type",        task_key(type)},
            {"range_start", std::to_string(range_start)},
            {"range_end",   std::to_string(range_end)},
        };
        return json_serialize(obj);
    }

    static WorkAssignment deserialize(const JsonObj& obj) {
        WorkAssignment wa;
        wa.task_id     = json_get(obj, "task_id");
        wa.type        = task_from_key(json_get(obj, "type", "general"));
        wa.range_start = json_get_u64(obj, "range_start");
        wa.range_end   = json_get_u64(obj, "range_end");
        return wa;
    }
};

// =========================================================================
// CarriageInfo -- status of a connected Carriage node
// =========================================================================

struct CarriageInfo {
    std::string hostname;
    int         cores       = 0;
    std::string gpu_name;
    std::string version;
    double      cpu_pct     = 0.0;
    double      gpu_pct     = 0.0;
    double      rate        = 0.0;   // candidates/sec
    uint64_t    current_pos = 0;
    bool        connected   = false;

    std::string serialize() const {
        JsonObj obj = {
            {"msg",         std::to_string(static_cast<uint8_t>(MessageType::Hello))},
            {"hostname",    hostname},
            {"cores",       std::to_string(cores)},
            {"gpu_name",    gpu_name},
            {"version",     version},
            {"cpu_pct",     std::to_string(cpu_pct)},
            {"gpu_pct",     std::to_string(gpu_pct)},
            {"rate",        std::to_string(rate)},
            {"current_pos", std::to_string(current_pos)},
            {"connected",   connected ? "true" : "false"},
        };
        return json_serialize(obj);
    }

    static CarriageInfo deserialize(const JsonObj& obj) {
        CarriageInfo ci;
        ci.hostname    = json_get(obj, "hostname");
        ci.cores       = static_cast<int>(json_get_u64(obj, "cores"));
        ci.gpu_name    = json_get(obj, "gpu_name");
        ci.version     = json_get(obj, "version");
        ci.cpu_pct     = json_get_f64(obj, "cpu_pct");
        ci.gpu_pct     = json_get_f64(obj, "gpu_pct");
        ci.rate        = json_get_f64(obj, "rate");
        ci.current_pos = json_get_u64(obj, "current_pos");
        ci.connected   = json_get_bool(obj, "connected");
        return ci;
    }
};

// =========================================================================
// Lightweight message structs for the remaining types
// =========================================================================

struct CancelWorkMsg {
    std::string task_id;

    std::string serialize() const {
        JsonObj obj = {
            {"msg",     std::to_string(static_cast<uint8_t>(MessageType::CancelWork))},
            {"task_id", task_id},
        };
        return json_serialize(obj);
    }

    static CancelWorkMsg deserialize(const JsonObj& obj) {
        return { json_get(obj, "task_id") };
    }
};

struct PingMsg {
    uint64_t seq = 0;

    std::string serialize() const {
        JsonObj obj = {
            {"msg", std::to_string(static_cast<uint8_t>(MessageType::Ping))},
            {"seq", std::to_string(seq)},
        };
        return json_serialize(obj);
    }

    static PingMsg deserialize(const JsonObj& obj) {
        return { json_get_u64(obj, "seq") };
    }
};

struct PongMsg {
    uint64_t seq = 0;

    std::string serialize() const {
        JsonObj obj = {
            {"msg", std::to_string(static_cast<uint8_t>(MessageType::Pong))},
            {"seq", std::to_string(seq)},
        };
        return json_serialize(obj);
    }

    static PongMsg deserialize(const JsonObj& obj) {
        return { json_get_u64(obj, "seq") };
    }
};

struct ProgressMsg {
    std::string task_id;
    uint64_t    current_pos = 0;
    uint64_t    tested      = 0;
    double      rate        = 0.0;

    std::string serialize() const {
        JsonObj obj = {
            {"msg",         std::to_string(static_cast<uint8_t>(MessageType::Progress))},
            {"task_id",     task_id},
            {"current_pos", std::to_string(current_pos)},
            {"tested",      std::to_string(tested)},
            {"rate",        std::to_string(rate)},
        };
        return json_serialize(obj);
    }

    static ProgressMsg deserialize(const JsonObj& obj) {
        ProgressMsg pm;
        pm.task_id     = json_get(obj, "task_id");
        pm.current_pos = json_get_u64(obj, "current_pos");
        pm.tested      = json_get_u64(obj, "tested");
        pm.rate        = json_get_f64(obj, "rate");
        return pm;
    }
};

struct DiscoveryMsg {
    std::string task_id;
    TaskType    type       = TaskType::GeneralPrime;
    uint64_t    value      = 0;
    uint64_t    value2     = 0;
    std::string timestamp;

    std::string serialize() const {
        JsonObj obj = {
            {"msg",       std::to_string(static_cast<uint8_t>(MessageType::DiscoveryReport))},
            {"task_id",   task_id},
            {"type",      task_key(type)},
            {"value",     std::to_string(value)},
            {"value2",    std::to_string(value2)},
            {"timestamp", timestamp},
        };
        return json_serialize(obj);
    }

    static DiscoveryMsg deserialize(const JsonObj& obj) {
        DiscoveryMsg dm;
        dm.task_id   = json_get(obj, "task_id");
        dm.type      = task_from_key(json_get(obj, "type", "general"));
        dm.value     = json_get_u64(obj, "value");
        dm.value2    = json_get_u64(obj, "value2");
        dm.timestamp = json_get(obj, "timestamp");
        return dm;
    }
};

struct WorkDoneMsg {
    std::string task_id;
    uint64_t    range_start = 0;
    uint64_t    range_end   = 0;
    uint64_t    tested      = 0;
    uint64_t    found       = 0;

    std::string serialize() const {
        JsonObj obj = {
            {"msg",         std::to_string(static_cast<uint8_t>(MessageType::WorkDone))},
            {"task_id",     task_id},
            {"range_start", std::to_string(range_start)},
            {"range_end",   std::to_string(range_end)},
            {"tested",      std::to_string(tested)},
            {"found",       std::to_string(found)},
        };
        return json_serialize(obj);
    }

    static WorkDoneMsg deserialize(const JsonObj& obj) {
        WorkDoneMsg wm;
        wm.task_id     = json_get(obj, "task_id");
        wm.range_start = json_get_u64(obj, "range_start");
        wm.range_end   = json_get_u64(obj, "range_end");
        wm.tested      = json_get_u64(obj, "tested");
        wm.found       = json_get_u64(obj, "found");
        return wm;
    }
};

// =========================================================================
// Dispatch helper: determine MessageType from a raw JSON payload
// =========================================================================

inline MessageType message_type_from_json(const std::string& json_body) {
    auto obj = json_parse(json_body);
    uint8_t code = static_cast<uint8_t>(json_get_u64(obj, "msg"));
    return static_cast<MessageType>(code);
}

}} // namespace prime::net
