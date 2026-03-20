#include <nuketorch/IPC.h>

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>

namespace nuketorch {
namespace {

void throwSysError(const std::string& prefix) {
    throw std::runtime_error(prefix + ": " + std::strerror(errno));
}

bool waitReadable(int fd, int timeout_ms) {
    if (timeout_ms < 0) {
        return true;
    }
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    const int r = poll(&pfd, 1, timeout_ms);
    if (r < 0) {
        if (errno == EINTR) {
            return false;
        }
        throwSysError("poll failed");
    }
    return r > 0;
}

void writeAll(int fd, const void* data, size_t size) {
    const char* p = static_cast<const char*>(data);
    size_t remaining = size;
    while (remaining > 0) {
        const ssize_t n = write(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            throwSysError("write failed");
        }
        if (n == 0) {
            throw std::runtime_error("write returned 0 bytes");
        }
        p += n;
        remaining -= static_cast<size_t>(n);
    }
}

void readAll(int fd, void* data, size_t size) {
    char* p = static_cast<char*>(data);
    size_t remaining = size;
    while (remaining > 0) {
        const ssize_t n = read(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            throwSysError("read failed");
        }
        if (n == 0) {
            throw IPCClosedError();
        }
        p += n;
        remaining -= static_cast<size_t>(n);
    }
}

sockaddr_un makeAddress(const std::string& path) {
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        throw std::runtime_error("socket path too long: " + path);
    }
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    return addr;
}

}  // namespace

IPCServer::IPCServer(const std::string& path)
    : path_(path), server_fd_(-1), client_fd_(-1) {
    unlink(path_.c_str());

    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throwSysError("server socket failed");
    }

    sockaddr_un addr = makeAddress(path_);
    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        cleanup();
        throwSysError("server bind failed");
    }

    if (listen(server_fd_, 1) < 0) {
        cleanup();
        throwSysError("server listen failed");
    }
}

IPCServer::~IPCServer() {
    cleanup();
}

std::string IPCServer::receive(int timeout_ms) {
    if (client_fd_ < 0) {
        if (!waitReadable(server_fd_, timeout_ms)) {
            throw std::runtime_error("server accept timed out");
        }
        client_fd_ = accept(server_fd_, nullptr, nullptr);
        if (client_fd_ < 0) {
            throwSysError("server accept failed");
        }
    } else {
        if (!waitReadable(client_fd_, timeout_ms)) {
            throw std::runtime_error("server receive timed out");
        }
    }

    uint32_t msg_size_network = 0;
    readAll(client_fd_, &msg_size_network, sizeof(msg_size_network));
    const uint32_t msg_size = ntohl(msg_size_network);

    std::string msg(msg_size, '\0');
    if (msg_size > 0) {
        readAll(client_fd_, msg.data(), msg_size);
    }
    return msg;
}

void IPCServer::send(const std::string& message) {
    if (client_fd_ < 0) {
        throw std::runtime_error("server has no client connection");
    }
    const uint32_t msg_size_network = htonl(static_cast<uint32_t>(message.size()));
    writeAll(client_fd_, &msg_size_network, sizeof(msg_size_network));
    if (!message.empty()) {
        writeAll(client_fd_, message.data(), message.size());
    }
}

bool IPCServer::hasData(int timeout_ms) {
    if (client_fd_ < 0 && server_fd_ < 0) {
        return false;
    }
    int fd = client_fd_ >= 0 ? client_fd_ : server_fd_;

    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    const int poll_result = poll(&pfd, 1, timeout_ms);
    if (poll_result < 0) {
        if (errno == EINTR) {
            return false;
        }
        throwSysError("poll failed");
    }
    return poll_result > 0;
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

IPCClient::IPCClient(const std::string& path)
    : path_(path), fd_(-1) {
    fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd_ < 0) {
        throwSysError("client socket failed");
    }

    sockaddr_un addr = makeAddress(path_);
    bool connected = false;
    for (int retry = 0; retry < 100; ++retry) {
        if (connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            connected = true;
            break;
        }
        if (errno != ENOENT && errno != ECONNREFUSED) {
            cleanup();
            throwSysError("client connect failed");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (!connected) {
        cleanup();
        throw std::runtime_error("client connect timed out: " + path_);
    }
}

IPCClient::~IPCClient() {
    cleanup();
}

void IPCClient::send(const std::string& message) {
    const uint32_t msg_size_network = htonl(static_cast<uint32_t>(message.size()));
    writeAll(fd_, &msg_size_network, sizeof(msg_size_network));
    if (!message.empty()) {
        writeAll(fd_, message.data(), message.size());
    }
}

std::string IPCClient::receive(int timeout_ms) {
    if (!waitReadable(fd_, timeout_ms)) {
        throw std::runtime_error("client receive timed out");
    }
    uint32_t msg_size_network = 0;
    readAll(fd_, &msg_size_network, sizeof(msg_size_network));
    const uint32_t msg_size = ntohl(msg_size_network);

    std::string msg(msg_size, '\0');
    if (msg_size > 0) {
        readAll(fd_, msg.data(), msg_size);
    }
    return msg;
}

void IPCClient::cleanup() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

}  // namespace nuketorch
