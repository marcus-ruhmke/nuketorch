#include <nuketorch/torch_worker/TorchScriptBackend.h>

#include <ATen/autocast_mode.h>

#include <stdexcept>

namespace nuketorch::torch_worker {

namespace {

/// Scoped CUDA autocast region. TorchScript dispatches ops through the
/// dispatcher, so the autocast key intercepts them the same way it does in
/// eager mode; the cast cache must be cleared when the region ends.
class CudaAutocastGuard {
public:
    explicit CudaAutocastGuard(bool enable) : enabled_(enable) {
        if (enabled_) {
            at::autocast::set_autocast_enabled(at::kCUDA, true);
        }
    }
    ~CudaAutocastGuard() {
        if (enabled_) {
            at::autocast::clear_cache();
            at::autocast::set_autocast_enabled(at::kCUDA, false);
        }
    }

    CudaAutocastGuard(const CudaAutocastGuard&) = delete;
    CudaAutocastGuard& operator=(const CudaAutocastGuard&) = delete;

private:
    bool enabled_;
};

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
    // buffers), not autocast-style mixed precision (see setAutocast for that).
    model_->to(dtype);
    path_ = model_path;
    device_ = device;
}

void TorchScriptBackend::setAutocast(bool enabled) {
    autocast_ = enabled;
}

std::vector<torch::Tensor> TorchScriptBackend::forward(const std::vector<torch::jit::IValue>& inputs) {
    if (!model_) {
        throw std::runtime_error("TorchScriptBackend: not loaded");
    }
    torch::InferenceMode guard;
    CudaAutocastGuard autocast(autocast_ && device_.is_cuda());
    return iValueToTensorOutputs(model_->forward(inputs));
}

bool TorchScriptBackend::isLoaded() const {
    return model_ != nullptr;
}

const std::string& TorchScriptBackend::loadedPath() const {
    return path_;
}

}  // namespace nuketorch::torch_worker
