#pragma once

#include <nuketorch/torch_worker/InferenceBackend.h>

#include <memory>

namespace nuketorch::torch_worker {

class TorchScriptBackend : public InferenceBackend {
public:
    void load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) override;
    std::vector<torch::Tensor> forward(const std::vector<torch::jit::IValue>& inputs) override;
    bool isLoaded() const override;
    const std::string& loadedPath() const override;

    /// When enabled and the model runs on CUDA, forward() executes under an
    /// autocast region: weights stay FP32 (pass dtype=kFloat32 to load()),
    /// eligible ops run in FP16.
    void setAutocast(bool enabled) override;

private:
    std::unique_ptr<torch::jit::script::Module> model_;
    std::string path_;
    torch::Device device_ = torch::kCPU;
    bool autocast_ = false;
};

}  // namespace nuketorch::torch_worker
