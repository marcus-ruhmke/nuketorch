#include <gtest/gtest.h>

#include <nuketorch/Errors.h>
#include <nuketorch/IPC.h>
#include <nuketorch/InferenceClient.h>
#include <nuketorch/InferenceMetrics.h>

#include <unistd.h>

#include <chrono>
#include <string>
#include <vector>

#ifndef FAKE_WORKER_BIN
#define FAKE_WORKER_BIN "./FakeWorker"
#endif

namespace {

std::string uniqueSocketPath(const char* tag) {
    return "/tmp/nuketorch_client_" + std::string(tag) + "_" + std::to_string(getpid()) + ".sock";
}

nuketorch::FrameBuffers makeBuffers(const std::vector<float>& in0,
                                    const std::vector<float>& in1,
                                    std::vector<float>& out) {
    nuketorch::FrameBuffers buffers;
    buffers.inputs = {in0.data(), in1.data()};
    buffers.output = out.data();
    buffers.width = 2;
    buffers.height = 1;
    buffers.channels = 3;
    return buffers;
}

}  // namespace

TEST(InferenceClientTest, PingAndProcessRoundTrip) {
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocketPath("roundtrip"), 2);
    client.start();

    EXPECT_TRUE(client.ping());

    std::vector<float> in0{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> in1{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(6, 0.0f);
    nuketorch::FrameBuffers buffers = makeBuffers(in0, in1, out);

    nuketorch::InferenceConfig cfg;
    cfg.model_path = "unused.pt";
    cfg.params["timestep"] = "0.25";

    nuketorch::InferenceMetrics metrics;
    client.processFrame(buffers, cfg, nullptr, &metrics);
    client.stop();

    EXPECT_GT(metrics.total_ms, 0.0);
    EXPECT_GT(metrics.shm_write_ms, 0.0);
    EXPECT_GT(metrics.round_trip_ms, 0.0);
    EXPECT_GT(metrics.shm_read_ms, 0.0);
    EXPECT_DOUBLE_EQ(metrics.backend_forward_ms, 42.0);
    EXPECT_EQ(metrics.backend, "fake");

    for (size_t i = 0; i < out.size(); ++i) {
        const float expected = ((in0[i] + in1[i]) * 0.5f) + 0.25f;
        EXPECT_FLOAT_EQ(out[i], expected);
    }
}

TEST(InferenceClientTest, SingleInputModelRejectsWrongBufferCount) {
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocketPath("single"), 1);
    client.start();

    std::vector<float> in0{1.0f};
    std::vector<float> out(1, 0.0f);

    nuketorch::FrameBuffers buffers;
    buffers.inputs = {in0.data()};
    buffers.output = out.data();
    buffers.width = 1;
    buffers.height = 1;
    buffers.channels = 1;

    nuketorch::InferenceConfig cfg;
    cfg.model_path = "unused.pt";

    // FakeWorker expects 2 inputs; its callback throws, surfaced as WorkerReportedError.
    EXPECT_THROW(client.processFrame(buffers, cfg), nuketorch::WorkerReportedError);

    // The worker survives a failed callback.
    EXPECT_TRUE(client.ping());
    client.stop();
}

TEST(InferenceClientTest, RestartAfterWorkerCrashMidFrame) {
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocketPath("crash"), 2);
    client.start();

    std::vector<float> in0{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> in1{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(6, 0.0f);
    nuketorch::FrameBuffers buffers = makeBuffers(in0, in1, out);

    nuketorch::InferenceConfig crash_cfg;
    crash_cfg.model_path = "unused.pt";
    crash_cfg.params["crash_now"] = "1";

    try {
        client.processFrame(buffers, crash_cfg);
        FAIL() << "expected WorkerDiedError";
    } catch (const nuketorch::WorkerDiedError& e) {
        // The captured stderr tail must make the crash self-diagnosing.
        EXPECT_NE(std::string(e.what()).find("crashing on request"), std::string::npos)
            << e.what();
    }

    EXPECT_FALSE(client.ping());

    // A fresh start on the same socket path must fully recover.
    client.start();
    EXPECT_TRUE(client.ping());

    nuketorch::InferenceConfig cfg;
    cfg.model_path = "unused.pt";
    cfg.params["timestep"] = "0.5";
    client.processFrame(buffers, cfg);
    for (size_t i = 0; i < out.size(); ++i) {
        const float expected = ((in0[i] + in1[i]) * 0.5f) + 0.5f;
        EXPECT_FLOAT_EQ(out[i], expected);
    }
    client.stop();
}

TEST(InferenceClientTest, FrameTimeoutKillsWorkerAndRestartWorks) {
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocketPath("timeout"), 2);
    client.start();

    std::vector<float> in0{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> in1{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(6, 0.0f);
    nuketorch::FrameBuffers buffers = makeBuffers(in0, in1, out);

    nuketorch::InferenceConfig slow_cfg;
    slow_cfg.model_path = "unused.pt";
    slow_cfg.params["sleep_ms"] = "10000";
    slow_cfg.frame_timeout_ms = 300;

    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_THROW(client.processFrame(buffers, slow_cfg), nuketorch::TimeoutError);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
    EXPECT_LT(elapsed_ms, 5000);

    // The hung worker was killed; a restart recovers.
    EXPECT_FALSE(client.ping());
    client.start();

    nuketorch::InferenceConfig cfg;
    cfg.model_path = "unused.pt";
    client.processFrame(buffers, cfg);
    client.stop();
}

TEST(InferenceClientTest, CooperativeCancelStopsQuickly) {
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocketPath("cancel"), 2);
    client.start();

    std::vector<float> in0{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> in1{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(6, 0.0f);
    nuketorch::FrameBuffers buffers = makeBuffers(in0, in1, out);

    nuketorch::InferenceConfig cfg;
    cfg.model_path = "unused.pt";
    cfg.params["sleep_ms"] = "30000";

    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_THROW(client.processFrame(buffers, cfg, []() { return true; }),
                 nuketorch::CancelledError);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
    // The worker polls the shared cancel flag every ~10 ms; without cooperative
    // cancellation this would take the full 30 s.
    EXPECT_LT(elapsed_ms, 5000);

    // Cancel is not fatal: the same worker keeps serving.
    EXPECT_TRUE(client.ping());
    nuketorch::InferenceConfig ok_cfg;
    ok_cfg.model_path = "unused.pt";
    client.processFrame(buffers, ok_cfg);
    client.stop();
}

TEST(InferenceClientTest, MappedFrameZeroCopyRoundTrip) {
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocketPath("mapped"), 2);
    client.start();

    nuketorch::MappedFrame frame = client.mapFrame(2, 1, 3);
    ASSERT_EQ(frame.inputs.size(), 2u);
    ASSERT_NE(frame.output, nullptr);

    for (int i = 0; i < 6; ++i) {
        frame.inputs[0][i] = static_cast<float>(i + 1);
        frame.inputs[1][i] = static_cast<float>(i + 7);
    }

    nuketorch::InferenceConfig cfg;
    cfg.model_path = "unused.pt";
    cfg.params["timestep"] = "0.25";

    nuketorch::InferenceMetrics metrics;
    client.processMappedFrame(cfg, nullptr, &metrics);

    for (int i = 0; i < 6; ++i) {
        const float expected = ((float(i + 1) + float(i + 7)) * 0.5f) + 0.25f;
        EXPECT_FLOAT_EQ(frame.output[i], expected);
    }
    EXPECT_DOUBLE_EQ(metrics.backend_forward_ms, 42.0);
    EXPECT_GT(metrics.round_trip_ms, 0.0);

    client.stop();
}

TEST(InferenceClientTest, ProcessMappedFrameWithoutMapThrows) {
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocketPath("nomapped"), 2);
    nuketorch::InferenceConfig cfg;
    EXPECT_THROW(client.processMappedFrame(cfg), nuketorch::Error);
}

TEST(InferenceClientTest, SpawnFailureIsReportedWithReason) {
    nuketorch::InferenceClient client("/nonexistent/worker/binary",
                                      uniqueSocketPath("nospawn"), 2);
    try {
        client.start();
        FAIL() << "expected SpawnError or WorkerDiedError";
    } catch (const nuketorch::Error& e) {
        // glibc posix_spawn reports exec failure directly (SpawnError); other
        // platforms surface it as an immediate child exit.
        EXPECT_TRUE(e.code() == nuketorch::ErrorCode::spawn_failed ||
                    e.code() == nuketorch::ErrorCode::worker_died)
            << e.what();
    }
}

TEST(InferenceClientTest, RepeatedAndGrowingFramesReuseWorker) {
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocketPath("reuse"), 2);
    client.start();

    nuketorch::InferenceConfig cfg;
    cfg.model_path = "unused.pt";
    cfg.params["timestep"] = "0.25";

    // Two identical frames: the second hits the worker-side mapping cache.
    for (int round = 0; round < 2; ++round) {
        std::vector<float> in0{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> in1{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        std::vector<float> out(6, 0.0f);
        nuketorch::FrameBuffers buffers = makeBuffers(in0, in1, out);
        client.processFrame(buffers, cfg);
        for (size_t i = 0; i < out.size(); ++i) {
            EXPECT_FLOAT_EQ(out[i], ((in0[i] + in1[i]) * 0.5f) + 0.25f) << "round " << round;
        }
    }

    // A larger frame forces the client to reallocate segments; the worker must
    // map the new ones, not reuse stale cache entries.
    const size_t count = 4 * 2 * 3;
    std::vector<float> big0(count), big1(count), big_out(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        big0[i] = static_cast<float>(i);
        big1[i] = static_cast<float>(2 * i);
    }
    nuketorch::FrameBuffers big;
    big.inputs = {big0.data(), big1.data()};
    big.output = big_out.data();
    big.width = 4;
    big.height = 2;
    big.channels = 3;
    client.processFrame(big, cfg);
    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(big_out[i], ((big0[i] + big1[i]) * 0.5f) + 0.25f);
    }

    client.stop();
}

TEST(InferenceClientTest, ConstructorRejectsTooManyInputs) {
    // [inputs..., output, cancel] must fit the per-message SCM_RIGHTS cap; fail
    // at construction, not on the first frame.
    const int too_many =
        static_cast<int>(nuketorch::kMaxFdsPerMessage) - 2 + 1;
    EXPECT_THROW(
        nuketorch::InferenceClient(FAKE_WORKER_BIN, uniqueSocketPath("many"), too_many),
        nuketorch::Error);
    // The largest allowed count constructs fine.
    EXPECT_NO_THROW(nuketorch::InferenceClient(FAKE_WORKER_BIN, uniqueSocketPath("max"),
                                               too_many - 1));
}

TEST(InferenceClientTest, StopWithoutStartIsHarmless) {
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocketPath("nostart"), 2);
    client.stop();
    EXPECT_FALSE(client.ping());
}
