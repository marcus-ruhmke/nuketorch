#include <gtest/gtest.h>

#include "TestPaths.h"

#include <nuketorch/Protocol.h>
#include <nuketorch/torch_worker/BackendCache.h>
#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/InferenceBackend.h>
#include <nuketorch/torch_worker/TorchScriptBackend.h>
#include <nuketorch/torch_worker/TorchWorkerUtils.h>

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

namespace nuketorch_fs = std::filesystem;

TEST(BackendParamsTest, DefaultBackendIsTorchScript) {
    std::unordered_map<std::string, std::string> empty;
    EXPECT_EQ(nuketorch::torch_worker::backendNameFromParams(empty),
              nuketorch::torch_worker::kBackendTorchScript);
}

TEST(BackendParamsTest, ParsesExplicitBackend) {
    std::unordered_map<std::string, std::string> p;
    p["backend"] = nuketorch::torch_worker::kBackendAOTInductor;
    EXPECT_EQ(nuketorch::torch_worker::backendNameFromParams(p),
              nuketorch::torch_worker::kBackendAOTInductor);
}

TEST(BackendParamsTest, NormalizesAlias) {
    std::unordered_map<std::string, std::string> p;
    p["backend"] = "AOTI";
    EXPECT_EQ(nuketorch::torch_worker::backendNameFromParams(p),
              nuketorch::torch_worker::kBackendAOTInductor);
}

TEST(BackendParamsTest, ParsesTensorRT) {
    std::unordered_map<std::string, std::string> p;
    p["backend"] = "trt";
    EXPECT_EQ(nuketorch::torch_worker::backendNameFromParams(p),
              nuketorch::torch_worker::kBackendTensorRT);
    p["backend"] = "tensorrt";
    EXPECT_EQ(nuketorch::torch_worker::backendNameFromParams(p),
              nuketorch::torch_worker::kBackendTensorRT);
}

#ifndef NUKETORCH_TORCH_WORKER_HAS_TENSORRT
TEST(InferenceBackendFactoryTest, TensorRTRaisesWhenBackendNotBuilt) {
    EXPECT_THROW(nuketorch::torch_worker::createBackend(nuketorch::torch_worker::kBackendTensorRT),
                 std::exception);
}
#endif

#ifdef NUKETORCH_TORCH_WORKER_HAS_TENSORRT
TEST(TensorRTBackendTest, FactoryCreatesTensorRT) {
    auto b = nuketorch::torch_worker::createBackend(nuketorch::torch_worker::kBackendTensorRT);
    ASSERT_NE(b, nullptr);
}

TEST(TensorRTBackendTest, LoadFailsForNonexistentEngine) {
    auto b = nuketorch::torch_worker::createBackend(nuketorch::torch_worker::kBackendTensorRT);
    EXPECT_THROW(b->load("/nonexistent/nuketorch_trt_missing.engine", torch::kCUDA, torch::kFloat32),
                 std::exception);
}

TEST(TensorRTBackendTest, LoadFailsForCorruptEngine) {
    const nuketorch_fs::path tmp =
        nuketorch_fs::temp_directory_path() / "nuketorch_trt_corrupt_test.engine";
    {
        std::ofstream out(tmp, std::ios::binary);
        out << "not a valid tensorrt engine";
    }
    auto b = nuketorch::torch_worker::createBackend(nuketorch::torch_worker::kBackendTensorRT);
    EXPECT_THROW(b->load(tmp.string(), torch::kCUDA, torch::kFloat32), std::exception);
    nuketorch_fs::remove(tmp);
}

TEST(TensorRTBackendTest, LoadFailsOnCpuDevice) {
    auto b = nuketorch::torch_worker::createBackend(nuketorch::torch_worker::kBackendTensorRT);
    EXPECT_THROW(b->load("/tmp/does_not_matter.engine", torch::kCPU, torch::kFloat32), std::exception);
}
#endif

TEST(TorchWorkerUtilsTest, PadTensorToMultipleReflect) {
    // Reflect padding requires pad amount < spatial extent; use sizes where the nnRetime-style pad is valid.
    auto t = torch::arange(100, torch::dtype(torch::kFloat32).device(torch::kCPU)).view({1, 1, 10, 10});
    auto padded = nuketorch::torch_worker::padTensorToMultiple(t, 8);
    EXPECT_EQ(padded.size(2), 16);
    EXPECT_EQ(padded.size(3), 16);
}

TEST(TorchWorkerUtilsTest, PadTensorToMultipleAlreadyAlignedReturnsUnchanged) {
    auto t = torch::ones({1, 1, 8, 8}, torch::kFloat32);
    auto out = nuketorch::torch_worker::padTensorToMultiple(t, 8);
    EXPECT_EQ(out.sizes(), t.sizes());
    EXPECT_EQ(out.data_ptr(), t.data_ptr());
}

TEST(TorchWorkerUtilsTest, PadTensorToMultipleTinySpatialDoesNotThrow) {
    // 1x1 spatial with factor 32 needs large pads; reflect would be invalid — implementation uses replicate.
    auto t = torch::ones({1, 1, 1, 1}, torch::kFloat32);
    auto padded = nuketorch::torch_worker::padTensorToMultiple(t, 32);
    EXPECT_EQ(padded.size(2), 32);
    EXPECT_EQ(padded.size(3), 32);
    EXPECT_TRUE(torch::allclose(padded.narrow(2, 0, 1).narrow(3, 0, 1), t));
}

TEST(TorchWorkerUtilsTest, ResolveDeviceCpuWhenDisabled) {
    auto d = nuketorch::torch_worker::resolveDevice(/*use_gpu=*/false);
    EXPECT_EQ(d.type(), torch::kCPU);
}

TEST(TorchScriptBackendIntegrationTest, LoadsFixtureAndDoubles) {
    nuketorch::torch_worker::TorchScriptBackend backend;
    backend.load(nuketorch::test::kTinyTorchScriptPt, torch::kCPU, torch::kFloat32);
    auto x = torch::ones({1, 1, 4, 4}, torch::kFloat32);
    auto y = backend.forward({x})[0];
    EXPECT_TRUE(torch::allclose(y, x * 2.0f));
}

TEST(BackendCacheTest, ReloadFailsForMissingArtifacts) {
    nuketorch::torch_worker::BackendCache cache;
    nuketorch::FrameHeader h;
    h.width = 1;
    h.height = 1;
    h.channels = 1;
    h.use_gpu = false;
    h.mixed_precision = false;
    std::unordered_map<std::string, std::string> params;

    h.model_path = "/nonexistent/nuketorch_torchscript_test.pt";
    params["backend"] = nuketorch::torch_worker::kBackendTorchScript;
    EXPECT_THROW(cache.ensure(h, params), std::exception);

    h.model_path = "/nonexistent/nuketorch_aoti_test.pt2";
    params["backend"] = nuketorch::torch_worker::kBackendAOTInductor;
    EXPECT_THROW(cache.ensure(h, params), std::exception);

    h.use_gpu = true;
    h.model_path = "/nonexistent/nuketorch_trt_test.engine";
    params["backend"] = nuketorch::torch_worker::kBackendTensorRT;
    EXPECT_THROW(cache.ensure(h, params), std::exception);
}

#ifdef NUKETORCH_TORCH_WORKER_HAS_TENSORRT
TEST(TensorRTBackendGPUIntegrationTest, TinyEngineDoublesWhenEnvSet) {
    const char* p = std::getenv("NUKETORCH_TRT_TEST_ENGINE");
    if (p == nullptr || p[0] == '\0') {
        GTEST_SKIP() << "Set NUKETORCH_TRT_TEST_ENGINE to a .engine path (see tests/fixtures/gen_tiny_trt_engine.py)";
    }
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto backend = nuketorch::torch_worker::createBackend(nuketorch::torch_worker::kBackendTensorRT);
    ASSERT_NE(backend, nullptr);
    backend->load(std::string(p), torch::kCUDA, torch::kFloat32);
    auto x = torch::ones({1, 1, 4, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto outs = backend->forward({x});
    ASSERT_EQ(outs.size(), 1u);
    EXPECT_TRUE(torch::allclose(outs[0], x * 2.0f));
}
#endif
