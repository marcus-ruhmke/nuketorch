#pragma once

#include <nuketorch/torch_worker/InferenceBackend.h>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <memory>

namespace nuketorch::torch_worker {

/// Loads a `.pt2` package produced by `torch._inductor.aoti_compile_and_package` (see PyTorch AOTInductor docs).
class AOTInductorBackend : public InferenceBackend {
public:
    ~AOTInductorBackend() override;

    void load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) override;
    std::vector<torch::Tensor> forward(const std::vector<torch::jit::IValue>& inputs) override;
    bool isLoaded() const override;
    const std::string& loadedPath() const override;

private:
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
    std::string path_;
    torch::Device device_ = torch::kCPU;
    torch::ScalarType dtype_ = torch::kFloat32;
};

}  // namespace nuketorch::torch_worker
