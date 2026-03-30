#pragma once

#include <nuketorch/torch_worker/InferenceBackend.h>

#include <memory>

namespace nuketorch::torch_worker {

class TorchScriptBackend : public InferenceBackend {
public:
    void load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) override;
    torch::Tensor forward(const std::vector<torch::jit::IValue>& inputs) override;
    bool isLoaded() const override;
    const std::string& loadedPath() const override;

private:
    std::unique_ptr<torch::jit::script::Module> model_;
    std::string path_;
};

}  // namespace nuketorch::torch_worker
