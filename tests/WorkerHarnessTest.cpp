#include <gtest/gtest.h>

#include <nuketorch/IPC.h>
#include <nuketorch/Protocol.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <cstring>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>
#include <vector>

#ifndef HARNESS_SMOKE_BIN
#define HARNESS_SMOKE_BIN "./HarnessSmokeWorker"
#endif

#ifndef FAKE_WORKER_BIN
#define FAKE_WORKER_BIN "./FakeWorker"
#endif

TEST(WorkerHarnessTest, HandshakePingAndShutdown) {
    const std::string socket_path =
        "/tmp/nuketorch_harness_smoke_" + std::to_string(getpid()) + ".sock";

    nuketorch::IPCServer server(socket_path);

    const pid_t child = fork();
    ASSERT_NE(child, -1);
    if (child == 0) {
        execl(HARNESS_SMOKE_BIN, HARNESS_SMOKE_BIN, socket_path.c_str(), nullptr);
        _exit(127);
    }

    const std::string ready = server.receive(-1);
    EXPECT_EQ(ready, "READY");

    server.send("PING");
    const std::string pong = server.receive(-1);
    EXPECT_EQ(pong, "PONG");

    server.send("QUIT");
    const std::string bye = server.receive(-1);
    EXPECT_EQ(bye, "BYE");

    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(WorkerHarnessTest, ProcessCallsCallback) {
    const std::string socket_path =
        "/tmp/nuketorch_harness_proc_" + std::to_string(getpid()) + ".sock";

    nuketorch::IPCServer server(socket_path);

    const pid_t child = fork();
    ASSERT_NE(child, -1);
    if (child == 0) {
        execl(FAKE_WORKER_BIN, FAKE_WORKER_BIN, socket_path.c_str(), nullptr);
        _exit(127);
    }

    ASSERT_EQ(server.receive(-1), "READY");

    const int w = 2;
    const int h = 1;
    const int c = 3;
    const size_t count = static_cast<size_t>(w) * h * c;
    const size_t bytes = count * sizeof(float);

    const std::string shm0 = "/nuketorch_hwtest_in0_" + std::to_string(getpid());
    const std::string shm1 = "/nuketorch_hwtest_in1_" + std::to_string(getpid());
    const std::string shmo = "/nuketorch_hwtest_out_" + std::to_string(getpid());

    std::vector<float> in0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> in1 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    {
        nuketorch::SharedMemoryBuffer b0(shm0, bytes, true);
        nuketorch::SharedMemoryBuffer b1(shm1, bytes, true);
        nuketorch::SharedMemoryBuffer bo(shmo, bytes, true);
        std::memcpy(b0.data(), in0.data(), bytes);
        std::memcpy(b1.data(), in1.data(), bytes);

        nuketorch::InferenceRequest req;
        req.shm_inputs = {shm0, shm1};
        req.shm_output = shmo;
        req.header.model_path = "unused.pt";
        req.header.width = w;
        req.header.height = h;
        req.header.channels = c;
        req.params["timestep"] = "0.25";

        server.send(nuketorch::serialize(req));
        const std::string reply = server.receive(-1);
        ASSERT_EQ(reply, "OK");

        std::vector<float> out(count);
        std::memcpy(out.data(), bo.data(), bytes);
        for (size_t i = 0; i < count; ++i) {
            const float expected = ((in0[i] + in1[i]) * 0.5f) + 0.25f;
            EXPECT_FLOAT_EQ(out[i], expected);
        }
    }

    server.send("QUIT");
    ASSERT_EQ(server.receive(-1), "BYE");

    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}
