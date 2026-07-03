#include <gtest/gtest.h>

#include <nuketorch/nuketorch_c.h>

#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#ifndef FAKE_WORKER_BIN
#define FAKE_WORKER_BIN "./FakeWorker"
#endif

TEST(NuketorchCApiTest, CreateAndDestroy) {
    nuketorch_client_t c =
        nuketorch_client_create("/bin/true", "/tmp/nuketorch_c_create_destroy.sock", 2);
    ASSERT_NE(c, nullptr);
    nuketorch_client_destroy(c);
}

TEST(NuketorchCApiTest, CreateNullPathReturnsNull) {
    EXPECT_EQ(nuketorch_client_create(nullptr, "/tmp/x.sock", 2), nullptr);
    EXPECT_EQ(nuketorch_client_create(FAKE_WORKER_BIN, nullptr, 2), nullptr);
}

TEST(NuketorchCApiTest, StartPingStop) {
    const std::string socket_path =
        "/tmp/nuketorch_capi_ping_" + std::to_string(getpid()) + ".sock";
    nuketorch_client_t c = nuketorch_client_create(FAKE_WORKER_BIN, socket_path.c_str(), 2);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(nuketorch_client_start(c), 0);
    EXPECT_EQ(nuketorch_client_ping(c), 0);
    EXPECT_EQ(nuketorch_client_stop(c), 0);
    nuketorch_client_destroy(c);
}

TEST(NuketorchCApiTest, GetGpuInfo) {
    const std::string socket_path =
        "/tmp/nuketorch_capi_gpu_" + std::to_string(getpid()) + ".sock";
    nuketorch_client_t c = nuketorch_client_create(FAKE_WORKER_BIN, socket_path.c_str(), 2);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(nuketorch_client_start(c), 0);

    char buf[256];
    ASSERT_EQ(nuketorch_client_get_gpu_info(c, buf, sizeof(buf)), 0);
    EXPECT_NE(std::strstr(buf, "FakeWorker"), nullptr);

    nuketorch_client_destroy(c);
}

TEST(NuketorchCApiTest, ProcessFrameRoundTrip) {
    const std::string socket_path =
        "/tmp/nuketorch_capi_frame_" + std::to_string(getpid()) + ".sock";
    nuketorch_client_t c = nuketorch_client_create(FAKE_WORKER_BIN, socket_path.c_str(), 2);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(nuketorch_client_start(c), 0);

    std::vector<float> in0{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> in1{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(6, 0.0f);
    const float* inputs[] = {in0.data(), in1.data()};

    nuketorch_frame_buffers fb{};
    fb.inputs = inputs;
    fb.num_inputs = 2;
    fb.output = out.data();
    fb.width = 2;
    fb.height = 1;
    fb.channels = 3;

    nuketorch_param params[] = {{"timestep", "0.25"}};
    nuketorch_inference_config cfg{};
    cfg.model_path = "unused.pt";
    cfg.use_gpu = 1;
    cfg.mixed_precision = 1;
    cfg.debug = 0;
    cfg.params = params;
    cfg.num_params = 1;

    nuketorch_inference_metrics metrics{};
    ASSERT_EQ(nuketorch_client_process_frame(c, &fb, &cfg, nullptr, nullptr, &metrics), 0);
    EXPECT_GT(metrics.total_ms, 0.0);
    EXPECT_GT(metrics.shm_write_ms, 0.0);
    EXPECT_GT(metrics.round_trip_ms, 0.0);
    EXPECT_GT(metrics.shm_read_ms, 0.0);
    EXPECT_DOUBLE_EQ(metrics.backend_forward_ms, 42.0);

    for (size_t i = 0; i < out.size(); ++i) {
        const float expected = ((in0[i] + in1[i]) * 0.5f) + 0.25f;
        EXPECT_FLOAT_EQ(out[i], expected);
    }

    nuketorch_client_destroy(c);
}

static int alwaysAbort(void*) {
    return 1;
}

TEST(NuketorchCApiTest, ProcessFrameWithAbort) {
    const std::string socket_path =
        "/tmp/nuketorch_capi_abort_" + std::to_string(getpid()) + ".sock";
    nuketorch_client_t c = nuketorch_client_create(FAKE_WORKER_BIN, socket_path.c_str(), 2);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(nuketorch_client_start(c), 0);

    std::vector<float> in0{1.0f};
    std::vector<float> in1{2.0f};
    std::vector<float> out(1, 0.0f);
    const float* inputs[] = {in0.data(), in1.data()};

    nuketorch_frame_buffers fb{};
    fb.inputs = inputs;
    fb.num_inputs = 2;
    fb.output = out.data();
    fb.width = 1;
    fb.height = 1;
    fb.channels = 1;

    nuketorch_inference_config cfg{};
    cfg.model_path = "unused.pt";
    cfg.use_gpu = 1;
    cfg.mixed_precision = 1;
    cfg.debug = 0;
    cfg.params = nullptr;
    cfg.num_params = 0;

    EXPECT_EQ(nuketorch_client_process_frame(c, &fb, &cfg, alwaysAbort, nullptr, nullptr), -1);
    EXPECT_NE(std::strlen(nuketorch_client_last_error(c)), 0u);
    EXPECT_EQ(nuketorch_client_last_error_code(c), NUKETORCH_ERRC_CANCELLED);

    nuketorch_client_destroy(c);
}

TEST(NuketorchCApiTest, FrameTimeoutReportsTimeoutCode) {
    const std::string socket_path =
        "/tmp/nuketorch_capi_timeout_" + std::to_string(getpid()) + ".sock";
    nuketorch_client_t c = nuketorch_client_create(FAKE_WORKER_BIN, socket_path.c_str(), 2);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(nuketorch_client_start(c), 0);

    std::vector<float> in0{1.0f};
    std::vector<float> in1{2.0f};
    std::vector<float> out(1, 0.0f);
    const float* inputs[] = {in0.data(), in1.data()};

    nuketorch_frame_buffers fb{};
    fb.inputs = inputs;
    fb.num_inputs = 2;
    fb.output = out.data();
    fb.width = 1;
    fb.height = 1;
    fb.channels = 1;

    nuketorch_param params[] = {{"sleep_ms", "10000"}};
    nuketorch_inference_config cfg{};
    cfg.model_path = "unused.pt";
    cfg.frame_timeout_ms = 300;
    cfg.params = params;
    cfg.num_params = 1;

    EXPECT_EQ(nuketorch_client_process_frame(c, &fb, &cfg, nullptr, nullptr, nullptr), -1);
    EXPECT_EQ(nuketorch_client_last_error_code(c), NUKETORCH_ERRC_TIMEOUT);

    nuketorch_client_destroy(c);
}

TEST(NuketorchCApiTest, LastErrorAfterFailure) {
    const std::string socket_path =
        "/tmp/nuketorch_capi_err_" + std::to_string(getpid()) + ".sock";
    nuketorch_client_t c = nuketorch_client_create(FAKE_WORKER_BIN, socket_path.c_str(), 2);
    ASSERT_NE(c, nullptr);

    std::vector<float> in0{1.0f};
    std::vector<float> in1{2.0f};
    const float* inputs[] = {in0.data(), in1.data()};
    nuketorch_frame_buffers fb{};
    fb.inputs = inputs;
    fb.num_inputs = 2;
    fb.output = in0.data();
    fb.width = 1;
    fb.height = 1;
    fb.channels = 1;

    nuketorch_inference_config cfg{};
    cfg.model_path = "unused.pt";
    cfg.use_gpu = 1;
    cfg.mixed_precision = 1;
    cfg.debug = 0;

    EXPECT_EQ(nuketorch_client_process_frame(c, &fb, &cfg, nullptr, nullptr, nullptr), -1);
    EXPECT_NE(std::strlen(nuketorch_client_last_error(c)), 0u);
    EXPECT_EQ(nuketorch_client_last_error_code(c), NUKETORCH_ERRC_NOT_STARTED);

    nuketorch_client_destroy(c);
}

TEST(NuketorchCApiTest, MetricsStringFields) {
    const std::string socket_path =
        "/tmp/nuketorch_capi_metrics_str_" + std::to_string(getpid()) + ".sock";
    nuketorch_client_t c = nuketorch_client_create(FAKE_WORKER_BIN, socket_path.c_str(), 2);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(nuketorch_client_start(c), 0);

    std::vector<float> in0{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> in1{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(6, 0.0f);
    const float* inputs[] = {in0.data(), in1.data()};

    nuketorch_frame_buffers fb{};
    fb.inputs = inputs;
    fb.num_inputs = 2;
    fb.output = out.data();
    fb.width = 2;
    fb.height = 1;
    fb.channels = 3;

    nuketorch_inference_config cfg{};
    cfg.model_path = "unused.pt";
    cfg.use_gpu = 1;
    cfg.mixed_precision = 1;
    cfg.debug = 0;
    cfg.params = nullptr;
    cfg.num_params = 0;

    nuketorch_inference_metrics metrics{};
    ASSERT_EQ(nuketorch_client_process_frame(c, &fb, &cfg, nullptr, nullptr, &metrics), 0);
    EXPECT_STREQ(metrics.backend, "fake");
    EXPECT_STREQ(metrics.device, "cpu");
    EXPECT_STREQ(metrics.dtype, "float32");

    nuketorch_client_destroy(c);
}

TEST(NuketorchCApiTest, MappedFrameRoundTrip) {
    const std::string socket_path =
        "/tmp/nuketorch_capi_mapped_" + std::to_string(getpid()) + ".sock";
    nuketorch_client_t c = nuketorch_client_create(FAKE_WORKER_BIN, socket_path.c_str(), 2);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(nuketorch_client_start(c), 0);

    float* inputs[2] = {nullptr, nullptr};
    float* output = nullptr;
    ASSERT_EQ(nuketorch_client_map_frame(c, 2, 1, 3, inputs, 2, &output), 0)
        << nuketorch_client_last_error(c);
    ASSERT_NE(inputs[0], nullptr);
    ASSERT_NE(inputs[1], nullptr);
    ASSERT_NE(output, nullptr);

    for (int i = 0; i < 6; ++i) {
        inputs[0][i] = static_cast<float>(i + 1);
        inputs[1][i] = static_cast<float>(i + 7);
    }

    nuketorch_param params[] = {{"timestep", "0.25"}};
    nuketorch_inference_config cfg{};
    cfg.model_path = "unused.pt";
    cfg.use_gpu = 1;
    cfg.mixed_precision = 1;
    cfg.params = params;
    cfg.num_params = 1;

    nuketorch_inference_metrics metrics{};
    ASSERT_EQ(nuketorch_client_process_mapped_frame(c, &cfg, nullptr, nullptr, &metrics), 0)
        << nuketorch_client_last_error(c);
    EXPECT_STREQ(metrics.backend, "fake");
    for (int i = 0; i < 6; ++i) {
        // FakeWorker: (in0 + in1) / 2 + timestep; no flip on the mapped path.
        EXPECT_FLOAT_EQ(output[i], (inputs[0][i] + inputs[1][i]) * 0.5f + 0.25f);
    }

    // Too-small capacity is a typed error, not a crash.
    float* one[1] = {nullptr};
    EXPECT_EQ(nuketorch_client_map_frame(c, 2, 1, 3, one, 1, &output), -1);
    EXPECT_EQ(nuketorch_client_last_error_code(c), NUKETORCH_ERRC_INVALID_ARGUMENT);

    nuketorch_client_destroy(c);
}
