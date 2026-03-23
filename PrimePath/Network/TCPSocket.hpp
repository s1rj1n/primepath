#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <memory>

namespace prime { namespace net {

// Default network port (matches NetworkProtocol.hpp)
static constexpr uint16_t PRIMEPATH_PORT = 9807;

// =========================================================================
// TCPSocket -- POSIX BSD socket wrapper with length-prefixed framing
//
// Framing: 4-byte big-endian uint32 length prefix + raw bytes body.
// Designed for JSON-over-TCP protocol messages.
// =========================================================================

class TCPSocket {
public:
    TCPSocket();
    ~TCPSocket();

    // Non-copyable, movable
    TCPSocket(const TCPSocket&) = delete;
    TCPSocket& operator=(const TCPSocket&) = delete;
    TCPSocket(TCPSocket&& other) noexcept;
    TCPSocket& operator=(TCPSocket&& other) noexcept;

    // ── Server operations ────────────────────────────────────────────
    // Bind and listen on the given port.  Returns true on success.
    bool listen_on(uint16_t port, int backlog = 8);

    // Accept a new connection (blocking, respects timeout).
    // Returns a new TCPSocket for the accepted connection, or nullptr on timeout/error.
    std::unique_ptr<TCPSocket> accept_connection();

    // ── Client operations ────────────────────────────────────────────
    // Connect to host:port.  Returns true on success.
    bool connect_to(const std::string& host, uint16_t port);

    // ── Messaging (length-prefixed framing) ──────────────────────────
    // Send a complete message (prepends 4-byte length header).
    bool send_message(const std::string& payload);

    // Receive a complete message (reads 4-byte header, then body).
    // Returns empty string on disconnect or error.
    // Handles partial reads internally.
    std::string recv_message();

    // ── Configuration ────────────────────────────────────────────────
    // Set read/write timeout in milliseconds (0 = blocking forever).
    void set_timeout(int timeout_ms);

    // Set non-blocking mode.
    void set_nonblocking(bool enabled);

    // ── State ────────────────────────────────────────────────────────
    bool is_valid() const { return _fd >= 0; }
    int  fd() const { return _fd; }
    void close();

    // Remote address info (populated after accept or connect)
    const std::string& remote_host() const { return _remote_host; }
    uint16_t remote_port() const { return _remote_port; }

private:
    int         _fd          = -1;
    int         _timeout_ms  = 5000;  // default 5 second timeout
    std::string _remote_host;
    uint16_t    _remote_port = 0;

    // Construct from an already-accepted fd
    explicit TCPSocket(int fd, const std::string& host, uint16_t port);

    // Low-level helpers: read/write exactly n bytes, handling partial I/O.
    bool read_exact(void* buf, size_t n);
    bool write_exact(const void* buf, size_t n);
};

}} // namespace prime::net
