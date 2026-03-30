#pragma once

#include <nuketorch/torch_worker/InferenceBackend.h>

#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

namespace nuketorch::torch_worker {

/// Loads a TensorRT serialized engine (`.engine`) and runs inference via `enqueueV3`.
class TensorRTBackend : public InferenceBackend {
public:
    ~TensorRTBackend() override;

    void load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) override;
    std::vector<torch::Tensor> forward(const std::vector<torch::jit::IValue>& inputs) override;
    bool isLoaded() const override;
    const std::string& loadedPath() const override;

private:
    /// Frees TensorRT objects and stream; clears I/O names and path. Safe to call multiple times.
    void releaseTensorRtResources() noexcept;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    std::string path_;
    torch::Device device_ = torch::kCPU;
};

}  // namespace nuketorch::torch_worker
