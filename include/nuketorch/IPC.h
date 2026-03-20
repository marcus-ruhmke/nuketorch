#pragma once

#include <string>
#include <stdexcept>

namespace nuketorch {

/// Thrown when the peer closes the connection while reading a framed message.
class IPCClosedError : public std::runtime_error {
public:
    IPCClosedError() : std::runtime_error("peer closed connection") {}
};

/// Unix domain stream socket server: bind/listen on @p path, accept one client, exchange length-prefixed UTF-8 payloads.
class IPCServer {
public:
    explicit IPCServer(const std::string& path);
    ~IPCServer();

    /// Receive one message (uint32 big-endian size + body). Blocks until data is available or times out.
    /// @param timeout_ms -1 = block indefinitely; 0 = non-blocking-style (poll once); >0 wait up to that many ms
    std::string receive(int timeout_ms = -1);
    /// Send one message (size prefix + body). Requires an accepted client.
    void send(const std::string& message);
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
    explicit IPCClient(const std::string& path);
    ~IPCClient();

    /// Send one length-prefixed message to the server.
    void send(const std::string& message);
    /// Receive one length-prefixed message. @param timeout_ms -1 = block indefinitely
    std::string receive(int timeout_ms = -1);

private:
    std::string path_;
    int fd_;

    void cleanup();
};

}  // namespace nuketorch
