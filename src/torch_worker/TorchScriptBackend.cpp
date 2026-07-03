#include <nuketorch/torch_worker/TorchScriptBackend.h>

#include <stdexcept>

namespace nuketorch::torch_worker {

namespace {

std::vector<torch::Tensor> iValueToTensorOutputs(const torch::jit::IValue& out) {
    if (out.isTensor()) {
        return {out.toTensor()};
    }
    if (out.isTuple()) {
        const c10::ivalue::Tuple& tup = out.toTupleRef();
        std::vector<torch::Tensor> r;
        r.reserve(tup.elements().size());
        for (const auto& e : tup.elements()) {
            if (!e.isTensor()) {
                throw std::runtime_error("TorchScriptBackend: tuple output contains non-tensor");
            }
            r.push_back(e.toTensor());
        }
        return r;
    }
    if (out.isTensorList()) {
        return out.toTensorVector();
    }
    if (out.isList()) {
        const c10::List<torch::jit::IValue> list = out.toList();
        std::vector<torch::Tensor> r;
        r.reserve(list.size());
        for (size_t i = 0; i < list.size(); ++i) {
            const torch::jit::IValue e = list.get(i);
            if (!e.isTensor()) {
                throw std::runtime_error("TorchScriptBackend: list output contains non-tensor");
            }
            r.push_back(e.toTensor());
        }
        return r;
    }
    throw std::runtime_error("TorchScriptBackend: unsupported output type (need tensor, tuple, or list)");
}

}  // namespace

void TorchScriptBackend::load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) {
    // Load straight onto the target device instead of staging through CPU.
    model_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path, device));
    model_->eval();
    // Note: this is a full dtype conversion of the whole module (all weights and
    // buffers), not autocast-style mixed precision.
    model_->to(dtype);
    path_ = model_path;
}

std::vector<torch::Tensor> TorchScriptBackend::forward(const std::vector<torch::jit::IValue>& inputs) {
    if (!model_) {
        throw std::runtime_error("TorchScriptBackend: not loaded");
    }
    torch::InferenceMode guard;
    return iValueToTensorOutputs(model_->forward(inputs));
}

bool TorchScriptBackend::isLoaded() const {
    return model_ != nullptr;
}

const std::string& TorchScriptBackend::loadedPath() const {
    return path_;
}

}  // namespace nuketorch::torch_worker
