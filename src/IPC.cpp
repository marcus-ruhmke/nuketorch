#include <nuketorch/IPC.h>

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>

namespace nuketorch {
namespace {

using Clock = std::chrono::steady_clock;

void throwSysError(const std::string& prefix) {
    throw Error(ErrorCode::internal, prefix + ": " + std::strerror(errno));
}

/// Deadline for one whole framed message. -1 means "no deadline".
class Deadline {
public:
    explicit Deadline(int timeout_ms)
        : infinite_(timeout_ms < 0),
          end_(Clock::now() + std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms)) {}

    /// Remaining budget in ms (clamped to 0), or -1 when infinite.
    int remainingMs() const {
        if (infinite_) {
            return -1;
        }
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - Clock::now()).count();
        return left > 0 ? static_cast<int>(left) : 0;
    }

    bool expired() const { return !infinite_ && Clock::now() >= end_; }

private:
    bool infinite_;
    Clock::time_point end_;
};

/// Poll @p fd for readability until data arrives or the deadline expires.
/// Retries EINTR with the remaining budget. Returns false on expiry.
bool waitReadable(int fd, const Deadline& deadline) {
    while (true) {
        pollfd pfd{};
        pfd.fd = fd;
        pfd.events = POLLIN;
        const int r = poll(&pfd, 1, deadline.remainingMs());
        if (r > 0) {
            return true;
        }
        if (r == 0) {
            return false;
        }
        if (errno == EINTR) {
            if (deadline.expired()) {
                return false;
            }
            continue;
        }
        throwSysError("poll failed");
    }
}

/// Closes a set of received fds unless released to the caller.
class FdCollector {
public:
    ~FdCollector() {
        for (int fd : fds_) {
            ::close(fd);
        }
    }

    void add(int fd) { fds_.push_back(fd); }
    size_t count() const { return fds_.size(); }

    void releaseTo(std::vector<int>* out) {
        if (out) {
            out->insert(out->end(), fds_.begin(), fds_.end());
        } else {
            for (int fd : fds_) {
                ::close(fd);
            }
        }
        fds_.clear();
    }

private:
    std::vector<int> fds_;
};

void sendAll(int fd, const void* data, size_t size) {
    const char* p = static_cast<const char*>(data);
    size_t remaining = size;
    while (remaining > 0) {
        const ssize_t n = ::send(fd, p, remaining, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EPIPE || errno == ECONNRESET) {
                throw IPCClosedError();
            }
            throwSysError("send failed");
        }
        if (n == 0) {
            throw Error(ErrorCode::internal, "send returned 0 bytes");
        }
        p += n;
        remaining -= static_cast<size_t>(n);
    }
}

/// Send the 4-byte length prefix + body as one framed message. When @p fds is
/// non-empty, the descriptors ride as SCM_RIGHTS ancillary data on the first
/// byte of the frame (kernel guarantees delivery with that byte).
void sendMessage(int fd, const std::string& message, const std::vector<int>& fds) {
    if (message.size() > kMaxMessageBytes) {
        throw ProtocolError("outgoing message exceeds kMaxMessageBytes");
    }
    if (fds.size() > kMaxFdsPerMessage) {
        throw ProtocolError("too many fds for one message");
    }

    const uint32_t msg_size_network = htonl(static_cast<uint32_t>(message.size()));

    if (fds.empty()) {
        sendAll(fd, &msg_size_network, sizeof(msg_size_network));
        if (!message.empty()) {
            sendAll(fd, message.data(), message.size());
        }
        return;
    }

    iovec iov[2];
    iov[0].iov_base = const_cast<uint32_t*>(&msg_size_network);
    iov[0].iov_len = sizeof(msg_size_network);
    iov[1].iov_base = const_cast<char*>(message.data());
    iov[1].iov_len = message.size();

    std::vector<char> control(CMSG_SPACE(fds.size() * sizeof(int)), 0);
    msghdr msg{};
    msg.msg_iov = iov;
    msg.msg_iovlen = message.empty() ? 1 : 2;
    msg.msg_control = control.data();
    msg.msg_controllen = control.size();

    cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(fds.size() * sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), fds.data(), fds.size() * sizeof(int));

    ssize_t sent = 0;
    while (true) {
        sent = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
        if (sent >= 0) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno == EPIPE || errno == ECONNRESET) {
            throw IPCClosedError();
        }
        throwSysError("sendmsg failed");
    }

    // The ancillary data went out with the first sendmsg; finish any remaining
    // payload bytes with plain sends.
    const size_t total = sizeof(msg_size_network) + message.size();
    size_t done = static_cast<size_t>(sent);
    if (done < sizeof(msg_size_network)) {
        sendAll(fd, reinterpret_cast<const char*>(&msg_size_network) + done, sizeof(msg_size_network) - done);
        done = sizeof(msg_size_network);
    }
    if (done < total) {
        sendAll(fd, message.data() + (done - sizeof(msg_size_network)), total - done);
    }
}

/// Read exactly @p size bytes, polling before each chunk so a stalled peer hits
/// the deadline instead of blocking forever. Any SCM_RIGHTS descriptors that
/// arrive with the data are captured (close-on-exec) into @p collector.
void readAll(int fd, void* data, size_t size, const Deadline& deadline, FdCollector& collector) {
    char* p = static_cast<char*>(data);
    size_t remaining = size;
    while (remaining > 0) {
        if (!waitReadable(fd, deadline)) {
            throw TimeoutError("IPC receive timed out mid-message");
        }

        iovec iov{};
        iov.iov_base = p;
        iov.iov_len = remaining;

        char control[CMSG_SPACE(kMaxFdsPerMessage * sizeof(int))];
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);

        const ssize_t n = ::recvmsg(fd, &msg, MSG_CMSG_CLOEXEC);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            throwSysError("recvmsg failed");
        }
        if (n == 0) {
            throw IPCClosedError();
        }

        for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
                continue;
            }
            const size_t payload = cmsg->cmsg_len - CMSG_LEN(0);
            const size_t count = payload / sizeof(int);
            for (size_t i = 0; i < count; ++i) {
                int received_fd = -1;
                std::memcpy(&received_fd, CMSG_DATA(cmsg) + i * sizeof(int), sizeof(int));
                collector.add(received_fd);
            }
        }
        if ((msg.msg_flags & MSG_CTRUNC) != 0 || collector.count() > kMaxFdsPerMessage) {
            throw ProtocolError("too many file descriptors in message");
        }

        p += n;
        remaining -= static_cast<size_t>(n);
    }
}

std::string receiveMessage(int fd, int timeout_ms, std::vector<int>* fds) {
    const Deadline deadline(timeout_ms);
    FdCollector collector;

    uint32_t msg_size_network = 0;
    readAll(fd, &msg_size_network, sizeof(msg_size_network), deadline, collector);
    const uint32_t msg_size = ntohl(msg_size_network);
    if (msg_size > kMaxMessageBytes) {
        throw ProtocolError("incoming message size " + std::to_string(msg_size) +
                            " exceeds limit " + std::to_string(kMaxMessageBytes));
    }

    std::string msg(msg_size, '\0');
    if (msg_size > 0) {
        readAll(fd, msg.data(), msg_size, deadline, collector);
    }
    collector.releaseTo(fds);
    return msg;
}

sockaddr_un makeAddress(const std::string& path) {
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        throw Error(ErrorCode::invalid_argument, "socket path too long: " + path);
    }
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    return addr;
}

}  // namespace

IPCServer::IPCServer(const std::string& path)
    : path_(path), server_fd_(-1), client_fd_(-1) {
    unlink(path_.c_str());

    server_fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (server_fd_ < 0) {
        throwSysError("server socket failed");
    }

    sockaddr_un addr = makeAddress(path_);
    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        cleanup();
        throwSysError("server bind failed");
    }

    // Belt and braces alongside the SO_PEERCRED check in acceptClient(): only
    // our own uid may connect.
    (void)chmod(path_.c_str(), 0600);

    if (listen(server_fd_, 1) < 0) {
        cleanup();
        throwSysError("server listen failed");
    }
}

IPCServer::~IPCServer() {
    cleanup();
}

void IPCServer::acceptClient(int timeout_ms, pid_t expected_pid) {
    if (client_fd_ >= 0) {
        return;
    }
    const Deadline deadline(timeout_ms);
    if (!waitReadable(server_fd_, deadline)) {
        throw TimeoutError("server accept timed out");
    }
    const int fd = accept4(server_fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd < 0) {
        throwSysError("server accept failed");
    }

    ucred cred{};
    socklen_t cred_len = sizeof(cred);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &cred_len) < 0) {
        ::close(fd);
        throwSysError("SO_PEERCRED failed");
    }
    if (cred.uid != geteuid()) {
        ::close(fd);
        throw ProtocolError("rejected connection from uid " + std::to_string(cred.uid));
    }
    if (expected_pid > 0 && cred.pid != expected_pid) {
        ::close(fd);
        throw ProtocolError("rejected connection from unexpected pid " + std::to_string(cred.pid) +
                            " (expected " + std::to_string(expected_pid) + ")");
    }
    client_fd_ = fd;
}

std::string IPCServer::receive(int timeout_ms, std::vector<int>* fds) {
    if (client_fd_ < 0) {
        acceptClient(timeout_ms);
    }
    return receiveMessage(client_fd_, timeout_ms, fds);
}

void IPCServer::send(const std::string& message) {
    if (client_fd_ < 0) {
        throw Error(ErrorCode::internal, "server has no client connection");
    }
    sendMessage(client_fd_, message, {});
}

void IPCServer::sendWithFds(const std::string& message, const std::vector<int>& fds) {
    if (client_fd_ < 0) {
        throw Error(ErrorCode::internal, "server has no client connection");
    }
    sendMessage(client_fd_, message, fds);
}

bool IPCServer::hasData(int timeout_ms) {
    if (client_fd_ < 0 && server_fd_ < 0) {
        return false;
    }
    const int fd = client_fd_ >= 0 ? client_fd_ : server_fd_;
    return waitReadable(fd, Deadline(timeout_ms < 0 ? -1 : timeout_ms));
}

void IPCServer::cleanup() {
    if (client_fd_ >= 0) {
        close(client_fd_);
        client_fd_ = -1;
    }
    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }
    if (!path_.empty()) {
        unlink(path_.c_str());
    }
}

IPCClient::IPCClient(const std::string& path, int connect_timeout_ms)
    : path_(path), fd_(-1) {
    fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd_ < 0) {
        throwSysError("client socket failed");
    }

    sockaddr_un addr = makeAddress(path_);
    const Deadline deadline(connect_timeout_ms);
    while (true) {
        if (connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            return;
        }
        if (errno != ENOENT && errno != ECONNREFUSED && errno != EINTR) {
            cleanup();
            throwSysError("client connect failed");
        }
        if (deadline.expired()) {
            cleanup();
            throw TimeoutError("client connect timed out: " + path_);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

IPCClient::~IPCClient() {
    cleanup();
}

void IPCClient::send(const std::string& message) {
    sendMessage(fd_, message, {});
}

void IPCClient::sendWithFds(const std::string& message, const std::vector<int>& fds) {
    sendMessage(fd_, message, fds);
}

std::string IPCClient::receive(int timeout_ms, std::vector<int>* fds) {
    return receiveMessage(fd_, timeout_ms, fds);
}

void IPCClient::cleanup() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

}  // namespace nuketorch
