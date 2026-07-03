#include <gtest/gtest.h>

#include <nuketorch/Errors.h>
#include <nuketorch/IPC.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using nuketorch::IPCClient;
using nuketorch::IPCServer;

namespace {

std::string uniqueSocketPath(const char* tag) {
    return "/tmp/nuketorch_ipc_" + std::string(tag) + "_" + std::to_string(getpid()) + ".sock";
}

/// Raw connector for protocol-abuse tests (partial frames, oversized lengths).
class RawClient {
public:
    explicit RawClient(const std::string& path) {
        fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        EXPECT_GE(fd_, 0);
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
        EXPECT_EQ(connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    }
    ~RawClient() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }
    void write(const void* data, size_t size) {
        EXPECT_EQ(::write(fd_, data, size), static_cast<ssize_t>(size));
    }

private:
    int fd_ = -1;
};

}  // namespace

TEST(IPCTest, ClientServerCommunication) {
    const std::string socket_path = uniqueSocketPath("basic");

    std::thread server_thread([&]() {
        IPCServer server(socket_path);
        std::string received = server.receive(-1);
        EXPECT_EQ(received, "{\"shm_in\": \"/in1\", \"width\": 1920}");
        server.send("SUCCESS");
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    IPCClient client(socket_path);
    client.send("{\"shm_in\": \"/in1\", \"width\": 1920}");

    std::string response = client.receive(5000);
    EXPECT_EQ(response, "SUCCESS");

    server_thread.join();
}

TEST(IPCTest, ServerReceiveTimeoutBeforeClientConnect) {
    const std::string socket_path = uniqueSocketPath("timeout");
    IPCServer server(socket_path);
    EXPECT_THROW(server.receive(50), nuketorch::TimeoutError);
}

TEST(IPCTest, FdPassingRoundTrip) {
    const std::string socket_path = uniqueSocketPath("fdpass");

    std::thread server_thread([&]() {
        IPCServer server(socket_path);
        std::vector<int> fds;
        const std::string msg = server.receive(5000, &fds);
        EXPECT_EQ(msg, "here-is-a-buffer");
        ASSERT_EQ(fds.size(), 1u);

        auto buf = nuketorch::SharedMemoryBuffer::adopt(fds[0], 16);
        const float* data = static_cast<const float*>(buf.data());
        EXPECT_FLOAT_EQ(data[0], 42.5f);
        server.send("got-it");
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto shm = nuketorch::SharedMemoryBuffer::create(16);
    static_cast<float*>(shm.data())[0] = 42.5f;

    IPCClient client(socket_path);
    client.sendWithFds("here-is-a-buffer", {shm.fd()});
    EXPECT_EQ(client.receive(5000), "got-it");

    server_thread.join();
}

TEST(IPCTest, UnclaimedFdsAreClosedNotLeaked) {
    const std::string socket_path = uniqueSocketPath("fddrop");

    std::thread server_thread([&]() {
        IPCServer server(socket_path);
        // Receiver passes no fd vector: descriptors must be silently closed.
        EXPECT_EQ(server.receive(5000, nullptr), "dropping");
        server.send("done");
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto shm = nuketorch::SharedMemoryBuffer::create(16);
    IPCClient client(socket_path);
    client.sendWithFds("dropping", {shm.fd()});
    EXPECT_EQ(client.receive(5000), "done");

    server_thread.join();
}

TEST(IPCTest, OversizedMessageIsRejected) {
    const std::string socket_path = uniqueSocketPath("oversize");
    IPCServer server(socket_path);

    RawClient raw(socket_path);
    const uint32_t huge = htonl(nuketorch::kMaxMessageBytes + 1);
    raw.write(&huge, sizeof(huge));

    EXPECT_THROW(server.receive(2000), nuketorch::ProtocolError);
}

TEST(IPCTest, PartialMessageHitsDeadline) {
    const std::string socket_path = uniqueSocketPath("partial");
    IPCServer server(socket_path);

    // Send only half of the length prefix, then stall: the read deadline must
    // fire instead of blocking forever.
    RawClient raw(socket_path);
    const char half[2] = {0, 0};
    raw.write(half, sizeof(half));

    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_THROW(server.receive(200), nuketorch::TimeoutError);
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 5000);
}

TEST(IPCTest, TruncatedBodyHitsDeadline) {
    const std::string socket_path = uniqueSocketPath("truncbody");
    IPCServer server(socket_path);

    // Announce a 100-byte body but deliver only 3 bytes.
    RawClient raw(socket_path);
    const uint32_t len = htonl(100);
    raw.write(&len, sizeof(len));
    raw.write("abc", 3);

    EXPECT_THROW(server.receive(200), nuketorch::TimeoutError);
}
