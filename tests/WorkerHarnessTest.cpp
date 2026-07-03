#include <gtest/gtest.h>

#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/IPC.h>
#include <nuketorch/Protocol.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <vector>

#ifndef HARNESS_SMOKE_BIN
#define HARNESS_SMOKE_BIN "./HarnessSmokeWorker"
#endif

#ifndef FAKE_WORKER_BIN
#define FAKE_WORKER_BIN "./FakeWorker"
#endif

namespace {

const std::string kReady = "READY|" + std::to_string(nuketorch::kProtocolVersion);

std::string uniqueSocketPath(const char* tag) {
    return "/tmp/nuketorch_harness_" + std::string(tag) + "_" + std::to_string(getpid()) + ".sock";
}

pid_t spawnWorker(const char* binary, const std::string& socket_path) {
    const pid_t child = fork();
    if (child == 0) {
        execl(binary, binary, socket_path.c_str(), nullptr);
        _exit(127);
    }
    return child;
}

void expectCleanExit(pid_t child) {
    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

}  // namespace

TEST(WorkerHarnessTest, HandshakePingAndShutdown) {
    const std::string socket_path = uniqueSocketPath("smoke");
    nuketorch::IPCServer server(socket_path);

    const pid_t child = spawnWorker(HARNESS_SMOKE_BIN, socket_path);
    ASSERT_NE(child, -1);

    EXPECT_EQ(server.receive(5000), kReady);

    server.send("PING|1");
    EXPECT_EQ(server.receive(5000), "R|1|PONG");

    server.send("GPUINFO|2");
    EXPECT_EQ(server.receive(5000), "R|2|OK|test-gpu");

    server.send("QUIT|3");
    EXPECT_EQ(server.receive(5000), "R|3|BYE");

    expectCleanExit(child);
}

TEST(WorkerHarnessTest, ProcessCallsCallbackAndReturnsMetrics) {
    const std::string socket_path = uniqueSocketPath("proc");
    nuketorch::IPCServer server(socket_path);

    const pid_t child = spawnWorker(FAKE_WORKER_BIN, socket_path);
    ASSERT_NE(child, -1);

    ASSERT_EQ(server.receive(5000), kReady);

    const int w = 2;
    const int h = 1;
    const int c = 3;
    const size_t count = static_cast<size_t>(w) * h * c;
    const size_t bytes = count * sizeof(float);

    std::vector<float> in0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> in1 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    auto b0 = nuketorch::SharedMemoryBuffer::create(bytes);
    auto b1 = nuketorch::SharedMemoryBuffer::create(bytes);
    auto bo = nuketorch::SharedMemoryBuffer::create(bytes);
    auto cancel = nuketorch::SharedMemoryBuffer::create(sizeof(uint32_t));
    std::memcpy(b0.data(), in0.data(), bytes);
    std::memcpy(b1.data(), in1.data(), bytes);
    std::memset(cancel.data(), 0, sizeof(uint32_t));

    nuketorch::InferenceRequest req;
    req.request_id = 7;
    req.num_inputs = 2;
    req.header.model_path = "unused.pt";
    req.header.width = w;
    req.header.height = h;
    req.header.channels = c;
    req.params["timestep"] = "0.25";

    server.sendWithFds(nuketorch::serialize(req), {b0.fd(), b1.fd(), bo.fd(), cancel.fd()});

    const std::string reply = server.receive(5000);
    const std::string prefix = "R|7|OK";
    ASSERT_GE(reply.size(), prefix.size());
    ASSERT_EQ(reply.substr(0, prefix.size()), prefix);

    nuketorch::InferenceMetrics metrics;
    std::string err;
    ASSERT_TRUE(nuketorch::parseInferenceOkResponse(reply.substr(4), metrics, err)) << err;
    EXPECT_DOUBLE_EQ(metrics.backend_forward_ms, 42.0);
    EXPECT_EQ(metrics.backend, "fake");
    EXPECT_EQ(metrics.device, "cpu");
    EXPECT_EQ(metrics.dtype, "float32");

    std::vector<float> out(count);
    std::memcpy(out.data(), bo.data(), bytes);
    for (size_t i = 0; i < count; ++i) {
        const float expected = ((in0[i] + in1[i]) * 0.5f) + 0.25f;
        EXPECT_FLOAT_EQ(out[i], expected);
    }

    server.send("QUIT|8");
    EXPECT_EQ(server.receive(5000), "R|8|BYE");
    expectCleanExit(child);
}

TEST(WorkerHarnessTest, FdCountMismatchIsBadRequest) {
    const std::string socket_path = uniqueSocketPath("badfd");
    nuketorch::IPCServer server(socket_path);

    const pid_t child = spawnWorker(FAKE_WORKER_BIN, socket_path);
    ASSERT_NE(child, -1);

    ASSERT_EQ(server.receive(5000), kReady);

    const size_t bytes = 4 * sizeof(float);
    auto b0 = nuketorch::SharedMemoryBuffer::create(bytes);

    nuketorch::InferenceRequest req;
    req.request_id = 9;
    req.num_inputs = 2;  // promises 2 inputs + output + cancel = 4 fds, sends 1
    req.header.model_path = "unused.pt";
    req.header.width = 2;
    req.header.height = 1;
    req.header.channels = 2;

    server.sendWithFds(nuketorch::serialize(req), {b0.fd()});

    const std::string reply = server.receive(5000);
    EXPECT_EQ(reply.rfind("R|9|ERR|bad_request|", 0), 0u) << reply;

    // The worker must survive a bad request and keep serving.
    server.send("PING|10");
    EXPECT_EQ(server.receive(5000), "R|10|PONG");

    server.send("QUIT|11");
    EXPECT_EQ(server.receive(5000), "R|11|BYE");
    expectCleanExit(child);
}

TEST(WorkerHarnessTest, InvalidDimensionsAreBadRequest) {
    const std::string socket_path = uniqueSocketPath("baddim");
    nuketorch::IPCServer server(socket_path);

    const pid_t child = spawnWorker(FAKE_WORKER_BIN, socket_path);
    ASSERT_NE(child, -1);

    ASSERT_EQ(server.receive(5000), kReady);

    nuketorch::InferenceRequest req;
    req.request_id = 12;
    req.num_inputs = 2;
    req.header.model_path = "unused.pt";
    req.header.width = 0;
    req.header.height = 1;
    req.header.channels = 1;

    server.send(nuketorch::serialize(req));
    const std::string reply = server.receive(5000);
    EXPECT_EQ(reply.rfind("R|12|ERR|bad_request|", 0), 0u) << reply;

    server.send("QUIT|13");
    EXPECT_EQ(server.receive(5000), "R|13|BYE");
    expectCleanExit(child);
}
