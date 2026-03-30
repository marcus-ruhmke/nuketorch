#pragma once

#include <memory>
#include <string>
#include <vector>

#include <torch/script.h>

namespace nuketorch::torch_worker {

/// Pluggable inference runtime (TorchScript `torch::jit::load` vs AOTInductor `.pt2` package).
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual void load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) = 0;

    /// Runs the compiled model. TorchScript accepts mixed tensors/scalars; AOTInductor converts scalars to tensors.
    virtual torch::Tensor forward(const std::vector<torch::jit::IValue>& inputs) = 0;

    virtual bool isLoaded() const = 0;
    virtual const std::string& loadedPath() const = 0;
};

std::unique_ptr<InferenceBackend> createBackend(const std::string& canonical_backend_name);

}  // namespace nuketorch::torch_worker
