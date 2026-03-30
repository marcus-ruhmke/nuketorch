#pragma once

#include <memory>
#include <string>
#include <vector>

#include <torch/script.h>

namespace nuketorch::torch_worker {

/// Pluggable inference runtime (TorchScript, AOTInductor `.pt2`, optional TensorRT `.engine`).
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual void load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) = 0;

    /// Runs the compiled model. One entry per output tensor (TorchScript may return a tuple).
    virtual std::vector<torch::Tensor> forward(const std::vector<torch::jit::IValue>& inputs) = 0;

    virtual bool isLoaded() const = 0;
    virtual const std::string& loadedPath() const = 0;
};

std::unique_ptr<InferenceBackend> createBackend(const std::string& canonical_backend_name);

}  // namespace nuketorch::torch_worker
