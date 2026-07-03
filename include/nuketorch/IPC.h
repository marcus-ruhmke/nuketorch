#pragma once

#include <nuketorch/Errors.h>

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nuketorch {

/// Thrown when the peer closes the connection while reading or writing a framed message.
class IPCClosedError : public Error {
public:
    IPCClosedError() : Error(ErrorCode::worker_died, "peer closed connection") {}
};

/// Upper bound for one framed message (control text or serialized request/metrics).
/// Frame pixel data never travels through the socket, so this is generous.
inline constexpr uint32_t kMaxMessageBytes = 16u * 1024u * 1024u;

/// Maximum number of SCM_RIGHTS file descriptors accepted per message.
/// The kernel bound is SCM_MAX_FD (253); 64 leaves ample margin while
/// allowing plugins that ship a window of frames per request (e.g. temporal
/// models needing ~50 input planes). Client and worker must agree — rebuild
/// worker binaries when changing this.
inline constexpr size_t kMaxFdsPerMessage = 64;

/// Unix domain stream socket server: bind/listen on @p path, accept one client,
/// exchange length-prefixed messages with optional SCM_RIGHTS fd payloads.
///
/// All fds are opened close-on-exec. Sends use MSG_NOSIGNAL, so a dead peer
/// raises IPCClosedError instead of SIGPIPE. Timeouts are deadlines covering
/// the whole message, including reads that stall mid-frame. Not thread-safe.
class IPCServer {
public:
    explicit IPCServer(const std::string& path);
    ~IPCServer();

    IPCServer(const IPCServer&) = delete;
    IPCServer& operator=(const IPCServer&) = delete;

    /// Accept the pending client connection, verifying via SO_PEERCRED that the
    /// peer runs as our own uid (and as @p expected_pid when > 0). Throws
    /// TimeoutError if nothing connects in time, ProtocolError on a credential
    /// mismatch. No-op if a client is already connected.
    void acceptClient(int timeout_ms, pid_t expected_pid = -1);

    /// Receive one message (uint32 big-endian size + body). Accepts the client
    /// first if none is connected (uid check only).
    /// @param timeout_ms -1 = block indefinitely; otherwise a deadline in ms for the whole message.
    /// @param fds When non-null, any SCM_RIGHTS descriptors that arrived with the message are appended
    ///            (close-on-exec). When null, unexpected descriptors are closed and discarded.
    std::string receive(int timeout_ms = -1, std::vector<int>* fds = nullptr);

    /// Send one message (size prefix + body). Requires an accepted client.
    void send(const std::string& message);

    /// Send one message with SCM_RIGHTS file descriptors attached to its first byte.
    void sendWithFds(const std::string& message, const std::vector<int>& fds);

    /// Returns true if the active socket (or listening socket before accept) is readable within @p timeout_ms.
    bool hasData(int timeout_ms = 0);

private:
    std::string path_;
    int server_fd_;
    int client_fd_;

    void cleanup();
};

/// Client for a Unix domain stream socket created by IPCServer; connects to @p path with retries.
class IPCClient {
public:
    /// @param connect_timeout_ms Overall budget for the connect retry loop.
    explicit IPCClient(const std::string& path, int connect_timeout_ms = 5000);
    ~IPCClient();

    IPCClient(const IPCClient&) = delete;
    IPCClient& operator=(const IPCClient&) = delete;

    /// Send one length-prefixed message to the server.
    void send(const std::string& message);
    /// Send one message with SCM_RIGHTS file descriptors attached to its first byte.
    void sendWithFds(const std::string& message, const std::vector<int>& fds);
    /// Receive one length-prefixed message; see IPCServer::receive for @p timeout_ms / @p fds semantics.
    std::string receive(int timeout_ms = -1, std::vector<int>* fds = nullptr);

private:
    std::string path_;
    int fd_;

    void cleanup();
};

}  // namespace nuketorch
