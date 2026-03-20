#include <gtest/gtest.h>

#include <nuketorch/IPC.h>

#include <unistd.h>

#include <chrono>
#include <string>
#include <thread>

using nuketorch::IPCClient;
using nuketorch::IPCServer;

TEST(IPCTest, ClientServerCommunication) {
    const std::string socket_path =
        "/tmp/nuketorch_ipc_test_" + std::to_string(getpid()) + ".sock";

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
    const std::string socket_path =
        "/tmp/nuketorch_ipc_timeout_" + std::to_string(getpid()) + ".sock";
    IPCServer server(socket_path);
    EXPECT_THROW(server.receive(50), std::runtime_error);
}
