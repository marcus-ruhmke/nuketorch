#include <nuketorch/torch_worker/TorchScriptBackend.h>

#include <stdexcept>

namespace nuketorch::torch_worker {

void TorchScriptBackend::load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) {
    model_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path));
    model_->eval();
    model_->to(device);
    model_->to(dtype);
    path_ = model_path;
}

torch::Tensor TorchScriptBackend::forward(const std::vector<torch::jit::IValue>& inputs) {
    if (!model_) {
        throw std::runtime_error("TorchScriptBackend: not loaded");
    }
    torch::InferenceMode guard;
    return model_->forward(inputs).toTensor();
}

bool TorchScriptBackend::isLoaded() const {
    return model_ != nullptr;
}

const std::string& TorchScriptBackend::loadedPath() const {
    return path_;
}

}  // namespace nuketorch::torch_worker
