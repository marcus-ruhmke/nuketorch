#include <gtest/gtest.h>

#include <nuketorch/InferenceMetrics.h>

#include <cmath>
#include <string>

namespace {

void expectMetricsNear(const nuketorch::InferenceMetrics& a, const nuketorch::InferenceMetrics& b) {
    EXPECT_DOUBLE_EQ(a.backend_forward_ms, b.backend_forward_ms);
    EXPECT_DOUBLE_EQ(a.gpu_compute_ms, b.gpu_compute_ms);
    EXPECT_DOUBLE_EQ(a.tensor_prep_ms, b.tensor_prep_ms);
    EXPECT_DOUBLE_EQ(a.output_copy_ms, b.output_copy_ms);
    EXPECT_DOUBLE_EQ(a.model_load_ms, b.model_load_ms);
    EXPECT_EQ(a.backend, b.backend);
    EXPECT_EQ(a.device, b.device);
    EXPECT_EQ(a.dtype, b.dtype);
    EXPECT_EQ(a.peak_gpu_memory_bytes, b.peak_gpu_memory_bytes);
    EXPECT_DOUBLE_EQ(a.shm_write_ms, b.shm_write_ms);
    EXPECT_DOUBLE_EQ(a.round_trip_ms, b.round_trip_ms);
    EXPECT_DOUBLE_EQ(a.shm_read_ms, b.shm_read_ms);
    EXPECT_DOUBLE_EQ(a.total_ms, b.total_ms);
}

}  // namespace

TEST(InferenceMetricsTest, RoundTripAllFields) {
    nuketorch::InferenceMetrics m;
    m.backend_forward_ms = 12.5;
    m.gpu_compute_ms = 8.25;
    m.tensor_prep_ms = 1.0;
    m.output_copy_ms = 2.5;
    m.model_load_ms = 100.0;
    m.backend = "torchscript";
    m.device = "cuda:0";
    m.dtype = "float16";
    m.peak_gpu_memory_bytes = 42LL * 1024 * 1024;
    m.shm_write_ms = 3.3;
    m.round_trip_ms = 20.0;
    m.shm_read_ms = 4.4;
    m.total_ms = 30.0;

    const std::string blob = nuketorch::serializeMetrics(m);
    nuketorch::InferenceMetrics parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::parseMetricsPayload(blob, parsed, error)) << error;

    // Host-side fields are not on the wire; expect defaults after parse.
    nuketorch::InferenceMetrics expected = m;
    expected.shm_write_ms = -1;
    expected.round_trip_ms = -1;
    expected.shm_read_ms = -1;
    expected.total_ms = -1;
    expectMetricsNear(parsed, expected);
}

TEST(InferenceMetricsTest, RoundTripDefaultFields) {
    nuketorch::InferenceMetrics m;
    const std::string blob = nuketorch::serializeMetrics(m);
    nuketorch::InferenceMetrics parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::parseMetricsPayload(blob, parsed, error)) << error;

    nuketorch::InferenceMetrics expected;
    expectMetricsNear(parsed, expected);
}

TEST(InferenceMetricsTest, EmptyStringFields) {
    nuketorch::InferenceMetrics m;
    m.backend = "";
    m.device = "";
    m.dtype = "";

    const std::string blob = nuketorch::serializeMetrics(m);
    nuketorch::InferenceMetrics parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::parseMetricsPayload(blob, parsed, error)) << error;

    EXPECT_EQ(parsed.backend, "");
    EXPECT_EQ(parsed.device, "");
    EXPECT_EQ(parsed.dtype, "");
}

TEST(InferenceMetricsTest, NegativeOneTimings) {
    nuketorch::InferenceMetrics m;
    m.backend_forward_ms = -1;
    m.gpu_compute_ms = -1;
    m.tensor_prep_ms = -1;
    m.output_copy_ms = -1;
    m.model_load_ms = -1;

    const std::string blob = nuketorch::serializeMetrics(m);
    nuketorch::InferenceMetrics parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::parseMetricsPayload(blob, parsed, error)) << error;

    EXPECT_DOUBLE_EQ(parsed.backend_forward_ms, -1);
    EXPECT_DOUBLE_EQ(parsed.gpu_compute_ms, -1);
    EXPECT_DOUBLE_EQ(parsed.tensor_prep_ms, -1);
    EXPECT_DOUBLE_EQ(parsed.output_copy_ms, -1);
    EXPECT_DOUBLE_EQ(parsed.model_load_ms, -1);
}

TEST(InferenceMetricsTest, ParseInferenceOkResponseStripsOkPrefix) {
    nuketorch::InferenceMetrics sent;
    sent.backend_forward_ms = 42.0;
    sent.backend = "aotinductor";

    const std::string blob = nuketorch::serializeMetrics(sent);
    const std::string response = std::string("OK") + blob;

    nuketorch::InferenceMetrics parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::parseInferenceOkResponse(response, parsed, error)) << error;
    EXPECT_DOUBLE_EQ(parsed.backend_forward_ms, 42.0);
    EXPECT_EQ(parsed.backend, "aotinductor");
}

TEST(InferenceMetricsTest, ParseInferenceOkResponseBareOk) {
    nuketorch::InferenceMetrics parsed;
    parsed.backend_forward_ms = 99.0;
    std::string error;
    ASSERT_TRUE(nuketorch::parseInferenceOkResponse("OK", parsed, error)) << error;
    EXPECT_DOUBLE_EQ(parsed.backend_forward_ms, -1);
}

TEST(InferenceMetricsTest, ParseInferenceOkResponseRejectsNonOk) {
    nuketorch::InferenceMetrics parsed;
    std::string error;
    EXPECT_FALSE(nuketorch::parseInferenceOkResponse("FAIL", parsed, error));
    EXPECT_FALSE(error.empty());
}
