#include <gtest/gtest.h>

#include <nuketorch/torch_worker/CustomLogTransfer.h>

#include <torch/torch.h>

#include <cmath>

namespace {

// Reference (Python parity) implementations — full closed form, no shortcuts.
torch::Tensor refForward(const torch::Tensor& x, double a, double b) {
    return 1.0 - torch::pow(1.0 + a * x, -b);
}

torch::Tensor refInverse(const torch::Tensor& y, double a, double b, double eps) {
    auto one_minus_y = (1.0 - y).clamp(eps, 1.0);
    return (torch::pow(one_minus_y, -1.0 / b) - 1.0) / a;
}

}  // namespace

TEST(CustomLogTransferTest, ForwardKnownSamplePointsDefaults) {
    // Defaults a=3, b=1: y = 3x / (1 + 3x).
    auto x = torch::tensor({0.0, 1.0 / 3.0, 1.0, 10.0}, torch::kFloat64);
    auto y = nuketorch::torch_worker::applyCustomLog(x);
    EXPECT_NEAR(y[0].item<double>(), 0.0, 1e-12);
    EXPECT_NEAR(y[1].item<double>(), 0.5, 1e-12);
    EXPECT_NEAR(y[2].item<double>(), 0.75, 1e-12);
    EXPECT_NEAR(y[3].item<double>(), 30.0 / 31.0, 1e-12);
}

TEST(CustomLogTransferTest, ForwardMonotonicOnSweep) {
    auto x = torch::linspace(0.0, 100.0, 1024, torch::kFloat32);
    auto y = nuketorch::torch_worker::applyCustomLog(x);
    auto diffs = y.slice(0, 1) - y.slice(0, 0, -1);
    // Strictly increasing (well above fp32 epsilon for this range).
    EXPECT_GT(diffs.min().item<float>(), 0.0f);
}

TEST(CustomLogTransferTest, RoundTripDefaults) {
    auto x = torch::linspace(0.0, 100.0, 1024, torch::kFloat32);
    auto y = nuketorch::torch_worker::applyCustomLog(x);
    auto x_round = nuketorch::torch_worker::applyCustomLogInverse(y);
    EXPECT_TRUE(torch::allclose(x_round, x, /*rtol=*/1e-4, /*atol=*/1e-4));
}

TEST(CustomLogTransferTest, RoundTripCustomParamsPowBranch) {
    // a=5.0, b=0.7 exercises the torch::pow branch (b != 1).
    const double a = 5.0;
    const double b = 0.7;
    auto x = torch::linspace(0.0, 100.0, 1024, torch::kFloat64);
    auto y = nuketorch::torch_worker::applyCustomLog(x, a, b);
    auto x_round = nuketorch::torch_worker::applyCustomLogInverse(y, a, b);
    EXPECT_TRUE(torch::allclose(x_round, x, /*rtol=*/1e-9, /*atol=*/1e-9));
}

TEST(CustomLogTransferTest, ShortcutMatchesReferenceForward) {
    // The b=1 algebraic shortcut must be numerically equivalent to the closed
    // form to fp32 tolerance.
    auto x = torch::linspace(0.0, 100.0, 4096, torch::kFloat32);
    auto y_fast = nuketorch::torch_worker::applyCustomLog(x);
    auto y_ref = refForward(x, 3.0, 1.0);
    EXPECT_TRUE(torch::allclose(y_fast, y_ref, /*rtol=*/1e-5, /*atol=*/1e-6));
}

TEST(CustomLogTransferTest, ShortcutMatchesReferenceInverse) {
    auto y = torch::linspace(0.0, 0.999, 4096, torch::kFloat32);
    auto x_fast = nuketorch::torch_worker::applyCustomLogInverse(y);
    auto x_ref = refInverse(y, 3.0, 1.0, 1e-9);
    EXPECT_TRUE(torch::allclose(x_fast, x_ref, /*rtol=*/1e-5, /*atol=*/1e-6));
}

TEST(CustomLogTransferTest, InverseOutOfRangeStaysFinite) {
    // Network can predict y >= 1 or slightly negative. Inverse must not blow up.
    auto y = torch::tensor({-0.1, 0.0, 0.5, 0.999999, 1.0, 1.5}, torch::kFloat32);
    auto x = nuketorch::torch_worker::applyCustomLogInverse(y);
    EXPECT_TRUE(torch::isfinite(x).all().item<bool>());
}

TEST(CustomLogTransferTest, DtypePreservedFloat32) {
    auto x = torch::linspace(0.0, 1.0, 16, torch::kFloat32);
    EXPECT_EQ(nuketorch::torch_worker::applyCustomLog(x).scalar_type(), torch::kFloat32);
    EXPECT_EQ(nuketorch::torch_worker::applyCustomLogInverse(x).scalar_type(), torch::kFloat32);
}

TEST(CustomLogTransferTest, DtypePreservedFloat64) {
    auto x = torch::linspace(0.0, 1.0, 16, torch::kFloat64);
    EXPECT_EQ(nuketorch::torch_worker::applyCustomLog(x).scalar_type(), torch::kFloat64);
    EXPECT_EQ(nuketorch::torch_worker::applyCustomLogInverse(x).scalar_type(), torch::kFloat64);
}

TEST(CustomLogTransferTest, ShapeAndDevicePreserved) {
    auto x = torch::rand({2, 3, 8, 8}, torch::kFloat32);
    auto y = nuketorch::torch_worker::applyCustomLog(x);
    EXPECT_EQ(y.sizes(), x.sizes());
    EXPECT_EQ(y.device(), x.device());
}

TEST(CustomLogTransferTest, RoundTripOnCuda) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto x_cpu = torch::linspace(0.0, 100.0, 1024, torch::kFloat32);
    auto x_cuda = x_cpu.to(torch::kCUDA);
    auto y_cuda = nuketorch::torch_worker::applyCustomLog(x_cuda);
    auto x_round_cuda = nuketorch::torch_worker::applyCustomLogInverse(y_cuda);
    EXPECT_TRUE(torch::allclose(x_round_cuda.cpu(), x_cpu, /*rtol=*/1e-4, /*atol=*/1e-4));
}
