#pragma once

#include <nuketorch/torch_worker/InferenceBackend.h>

#include <memory>
#include <optional>

namespace nuketorch::torch_worker {

class TorchScriptBackend : public InferenceBackend {
public:
    /// Loads under GraphOptimizerEnabledGuard(precision().jit_optimize) when
    /// it is set; that flag then holds around every call into this module.
    void load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) override;
    std::vector<torch::Tensor> forward(const std::vector<torch::jit::IValue>& inputs) override;

    /// Calls run under InferenceMode, the load-time executor flag, the spec's
    /// TF32 and, on CUDA, an autocast region for methods the spec autocasts
    /// (weights stay as loaded: pass dtype=kFloat32 to load() for autocast).
    /// A method the spec does not autocast runs with CUDA autocast disabled,
    /// even inside an enclosing region.
    torch::jit::IValue runMethod(const std::string& name, std::vector<torch::jit::IValue> inputs) override;
    torch::jit::IValue runMethod(const std::string& name,
                                 std::vector<torch::jit::IValue> inputs,
                                 const PrecisionSpec& spec) override;
    bool hasMethod(const std::string& name) const override;

    bool isLoaded() const override;
    const std::string& loadedPath() const override;

private:
    std::unique_ptr<torch::jit::script::Module> model_;
    std::string path_;
    torch::Device device_ = torch::kCPU;
    /// Executor flag the module was loaded under; nullopt = the caller's.
    std::optional<bool> jit_optimize_;
};

}  // namespace nuketorch::torch_worker
