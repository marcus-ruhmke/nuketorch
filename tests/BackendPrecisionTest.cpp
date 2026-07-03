#include <gtest/gtest.h>

#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/TorchScriptBackend.h>

#include <torch/cuda.h>
#include <torch/script.h>

#include <unistd.h>

#include <cstdio>
#include <stdexcept>
#include <string>

using nuketorch::torch_worker::PrecisionMode;
using nuketorch::torch_worker::precisionModeFromParams;

TEST(BackendPrecisionTest, DefaultFollowsMixedPrecisionFlag) {
    EXPECT_EQ(precisionModeFromParams({}, true), PrecisionMode::half);
    EXPECT_EQ(precisionModeFromParams({}, false), PrecisionMode::full);
}

TEST(BackendPrecisionTest, ParsesAliasesCaseInsensitively) {
    EXPECT_EQ(precisionModeFromParams({{"precision", "AutoCast"}}, true), PrecisionMode::autocast);
    EXPECT_EQ(precisionModeFromParams({{"precision", "amp"}}, false), PrecisionMode::autocast);
    EXPECT_EQ(precisionModeFromParams({{"precision", "fp16"}}, false), PrecisionMode::half);
    EXPECT_EQ(precisionModeFromParams({{"precision", "float32"}}, true), PrecisionMode::full);
}

TEST(BackendPrecisionTest, RejectsUnknownValue) {
    EXPECT_THROW(precisionModeFromParams({{"precision", "quantum"}}, true), std::invalid_argument);
}

namespace {

/// Builds and saves a minimal TorchScript module whose forward is a matmul —
/// an op on autocast's FP16 list.
std::string saveMatmulModule() {
    torch::jit::Module m("MatmulModule");
    m.define(R"JIT(
        def forward(self, x):
            return torch.mm(x, x)
    )JIT");
    const std::string path =
        "/tmp/nuketorch_autocast_test_" + std::to_string(getpid()) + ".pt";
    m.save(path);
    return path;
}

}  // namespace

TEST(BackendPrecisionTest, AutocastRunsMatmulInHalfOnCuda) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    const std::string path = saveMatmulModule();

    nuketorch::torch_worker::TorchScriptBackend backend;
    backend.load(path, torch::Device(torch::kCUDA, 0), torch::kFloat32);

    const auto x = torch::randn(
        {8, 8}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    backend.setAutocast(false);
    const auto full = backend.forward({x});
    ASSERT_EQ(full.size(), 1u);
    EXPECT_EQ(full[0].scalar_type(), torch::kFloat32);

    backend.setAutocast(true);
    const auto amp = backend.forward({x});
    ASSERT_EQ(amp.size(), 1u);
    EXPECT_EQ(amp[0].scalar_type(), torch::kFloat16);
    EXPECT_TRUE(torch::allclose(amp[0].to(torch::kFloat32), full[0], /*rtol=*/1e-2, /*atol=*/1e-2));

    // The region must not leak: disabling restores FP32 execution.
    backend.setAutocast(false);
    const auto full_again = backend.forward({x});
    EXPECT_EQ(full_again[0].scalar_type(), torch::kFloat32);

    std::remove(path.c_str());
}
