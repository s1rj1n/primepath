#include "TCPSocket.hpp"
#include "NetworkProtocol.hpp"

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <cerrno>
#include <cstring>

// macOS does not have MSG_NOSIGNAL; use SO_NOSIGPIPE on the socket instead.
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace prime { namespace net {

// =========================================================================
// Construction / destruction
// =========================================================================

TCPSocket::TCPSocket() = default;

TCPSocket::TCPSocket(int fd, const std::string& host, uint16_t port)
    : _fd(fd), _remote_host(host), _remote_port(port) {}

TCPSocket::~TCPSocket() {
    close();
}

TCPSocket::TCPSocket(TCPSocket&& other) noexcept
    : _fd(other._fd),
      _timeout_ms(other._timeout_ms),
      _remote_host(std::move(other._remote_host)),
      _remote_port(other._remote_port)
{
    other._fd = -1;
}

TCPSocket& TCPSocket::operator=(TCPSocket&& other) noexcept {
    if (this != &other) {
        close();
        _fd          = other._fd;
        _timeout_ms  = other._timeout_ms;
        _remote_host = std::move(other._remote_host);
        _remote_port = other._remote_port;
        other._fd    = -1;
    }
    return *this;
}

void TCPSocket::close() {
    if (_fd >= 0) {
        ::close(_fd);
        _fd = -1;
    }
}

// =========================================================================
// Server: listen + accept
// =========================================================================

bool TCPSocket::listen_on(uint16_t port, int backlog) {
    _fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_fd < 0) return false;

    // Allow address reuse to avoid "address already in use" on restart
    int opt = 1;
    ::setsockopt(_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (::bind(_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        close();
        return false;
    }

    if (::listen(_fd, backlog) < 0) {
        close();
        return false;
    }

    return true;
}

std::unique_ptr<TCPSocket> TCPSocket::accept_connection() {
    if (_fd < 0) return nullptr;

    // Poll for incoming connection with timeout
    struct pollfd pfd{};
    pfd.fd     = _fd;
    pfd.events = POLLIN;
    int ret = ::poll(&pfd, 1, _timeout_ms > 0 ? _timeout_ms : -1);
    if (ret <= 0) return nullptr;  // timeout or error

    struct sockaddr_in client_addr{};
    socklen_t addr_len = sizeof(client_addr);
    int client_fd = ::accept(_fd, reinterpret_cast<struct sockaddr*>(&client_addr), &addr_len);
    if (client_fd < 0) return nullptr;

    // Disable Nagle for lower latency on small messages
    int flag = 1;
    ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

#ifdef SO_NOSIGPIPE
    ::setsockopt(client_fd, SOL_SOCKET, SO_NOSIGPIPE, &flag, sizeof(flag));
#endif

    char host_buf[INET_ADDRSTRLEN];
    ::inet_ntop(AF_INET, &client_addr.sin_addr, host_buf, sizeof(host_buf));
    uint16_t client_port = ntohs(client_addr.sin_port);

    auto sock = std::unique_ptr<TCPSocket>(
        new TCPSocket(client_fd, std::string(host_buf), client_port));
    sock->set_timeout(_timeout_ms);
    return sock;
}

// =========================================================================
// Client: connect
// =========================================================================

bool TCPSocket::connect_to(const std::string& host, uint16_t port) {
    // Resolve hostname
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    if (::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0 || !res) {
        return false;
    }

    _fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (_fd < 0) {
        ::freeaddrinfo(res);
        return false;
    }

    // Disable Nagle
    int flag = 1;
    ::setsockopt(_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

#ifdef SO_NOSIGPIPE
    ::setsockopt(_fd, SOL_SOCKET, SO_NOSIGPIPE, &flag, sizeof(flag));
#endif

    if (::connect(_fd, res->ai_addr, res->ai_addrlen) < 0) {
        ::freeaddrinfo(res);
        close();
        return false;
    }

    ::freeaddrinfo(res);

    _remote_host = host;
    _remote_port = port;
    return true;
}

// =========================================================================
// Configuration
// =========================================================================

void TCPSocket::set_timeout(int timeout_ms) {
    _timeout_ms = timeout_ms;
    if (_fd < 0) return;

    struct timeval tv{};
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    ::setsockopt(_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

void TCPSocket::set_nonblocking(bool enabled) {
    if (_fd < 0) return;
    int flags = ::fcntl(_fd, F_GETFL, 0);
    if (flags < 0) return;
    if (enabled)
        flags |= O_NONBLOCK;
    else
        flags &= ~O_NONBLOCK;
    ::fcntl(_fd, F_SETFL, flags);
}

// =========================================================================
// Messaging: length-prefixed framing
// =========================================================================

bool TCPSocket::send_message(const std::string& payload) {
    if (_fd < 0) return false;

    uint32_t len = static_cast<uint32_t>(payload.size());
    uint8_t header[4];
    encode_length(len, header);

    if (!write_exact(header, 4)) return false;
    if (!write_exact(payload.data(), payload.size())) return false;
    return true;
}

std::string TCPSocket::recv_message() {
    if (_fd < 0) return "";

    uint8_t header[4];
    if (!read_exact(header, 4)) return "";

    uint32_t len = decode_length(header);
    if (len == 0) return "";
    if (len > 16 * 1024 * 1024) return "";  // sanity: max 16 MB message

    std::string body(len, '\0');
    if (!read_exact(&body[0], len)) return "";
    return body;
}

// =========================================================================
// Low-level helpers: handle partial reads/writes
// =========================================================================

bool TCPSocket::read_exact(void* buf, size_t n) {
    auto* ptr = static_cast<uint8_t*>(buf);
    size_t remaining = n;

    while (remaining > 0) {
        // Use poll to respect timeout
        struct pollfd pfd{};
        pfd.fd     = _fd;
        pfd.events = POLLIN;
        int poll_ret = ::poll(&pfd, 1, _timeout_ms > 0 ? _timeout_ms : -1);
        if (poll_ret <= 0) return false;  // timeout or error

        ssize_t got = ::recv(_fd, ptr, remaining, 0);
        if (got <= 0) return false;  // disconnect or error
        ptr       += got;
        remaining -= static_cast<size_t>(got);
    }
    return true;
}

bool TCPSocket::write_exact(const void* buf, size_t n) {
    auto* ptr = static_cast<const uint8_t*>(buf);
    size_t remaining = n;

    while (remaining > 0) {
        ssize_t sent = ::send(_fd, ptr, remaining, MSG_NOSIGNAL);
        if (sent <= 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Poll for writability
                struct pollfd pfd{};
                pfd.fd     = _fd;
                pfd.events = POLLOUT;
                int poll_ret = ::poll(&pfd, 1, _timeout_ms > 0 ? _timeout_ms : -1);
                if (poll_ret <= 0) return false;
                continue;
            }
            return false;
        }
        ptr       += sent;
        remaining -= static_cast<size_t>(sent);
    }
    return true;
}

}} // namespace prime::net
