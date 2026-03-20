#include <gtest/gtest.h>

#include <nuketorch/InferenceClient.h>

#include <unistd.h>

#include <string>
#include <vector>

#ifndef FAKE_WORKER_BIN
#define FAKE_WORKER_BIN "./FakeWorker"
#endif

TEST(InferenceClientTest, PingAndProcessRoundTrip) {
    const std::string socket_path =
        "/tmp/nuketorch_infer_client_" + std::to_string(getpid()) + ".sock";
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, socket_path, 2);
    client.start();

    EXPECT_TRUE(client.ping());

    std::vector<float> in0{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> in1{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(6, 0.0f);

    nuketorch::FrameBuffers buffers;
    buffers.inputs = {in0.data(), in1.data()};
    buffers.output = out.data();
    buffers.width = 2;
    buffers.height = 1;
    buffers.channels = 3;

    nuketorch::InferenceConfig cfg;
    cfg.model_path = "unused.pt";
    cfg.params["timestep"] = "0.25";

    client.processFrame(buffers, cfg);
    client.stop();

    for (size_t i = 0; i < out.size(); ++i) {
        const float expected = ((in0[i] + in1[i]) * 0.5f) + 0.25f;
        EXPECT_FLOAT_EQ(out[i], expected);
    }
}

TEST(InferenceClientTest, SingleInputModelRejectsWrongBufferCount) {
    const std::string socket_path =
        "/tmp/nuketorch_infer_single_" + std::to_string(getpid()) + ".sock";
    nuketorch::InferenceClient client(FAKE_WORKER_BIN, socket_path, 1);
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

    // FakeWorker expects 2 inputs; worker will fail or callback throws
    EXPECT_THROW(client.processFrame(buffers, cfg), std::runtime_error);

    client.stop();
}
