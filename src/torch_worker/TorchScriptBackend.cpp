#include <nuketorch/torch_worker/TorchScriptBackend.h>

#include <nuketorch/torch_worker/TorchWorkerUtils.h>

#include <ATen/autocast_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <stdexcept>

namespace nuketorch::torch_worker {

namespace {

/// Scoped CUDA autocast region, mirroring torch.autocast's enter/exit: saves
/// enabled/dtype/cache flag, bumps the nesting count, and clears the cast
/// cache only when the outermost region ends. TorchScript dispatches ops
/// through the dispatcher, so the autocast key intercepts them as in eager
/// mode. With @p on false the region disables autocast, but only when some
/// enclosing region enabled it; otherwise it is a no-op.
class CudaAutocastGuard {
public:
    CudaAutocastGuard(bool on, at::ScalarType dtype)
        : active_(on || at::autocast::is_autocast_enabled(at::kCUDA)) {
        if (!active_) {
            return;
        }
        prev_enabled_ = at::autocast::is_autocast_enabled(at::kCUDA);
        prev_dtype_ = at::autocast::get_autocast_dtype(at::kCUDA);
        prev_cache_ = at::autocast::is_autocast_cache_enabled();
        at::autocast::increment_nesting();
        at::autocast::set_autocast_enabled(at::kCUDA, on);
        if (on) {
            at::autocast::set_autocast_dtype(at::kCUDA, dtype);
            at::autocast::set_autocast_cache_enabled(true);  // torch.autocast default
        }
    }
    ~CudaAutocastGuard() {
        if (!active_) {
            return;
        }
        if (at::autocast::decrement_nesting() == 0) {
            at::autocast::clear_cache();
        }
        at::autocast::set_autocast_enabled(at::kCUDA, prev_enabled_);
        at::autocast::set_autocast_dtype(at::kCUDA, prev_dtype_);
        at::autocast::set_autocast_cache_enabled(prev_cache_);
    }

    CudaAutocastGuard(const CudaAutocastGuard&) = delete;
    CudaAutocastGuard& operator=(const CudaAutocastGuard&) = delete;

private:
    bool active_;
    bool prev_enabled_ = false;
    at::ScalarType prev_dtype_ = at::kHalf;
    bool prev_cache_ = true;
};

at::ScalarType autocastScalarType(AutocastDtype d) {
    return d == AutocastDtype::bfloat16 ? at::kBFloat16 : at::kHalf;
}

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
    // The profiling executor fixes a method's plan on its first run, so the
    // flag must already hold for anything load runs (__setstate__).
    const std::optional<bool> jit_optimize = precision_.jit_optimize;
    std::optional<torch::jit::GraphOptimizerEnabledGuard> executor;
    if (jit_optimize) {
        executor.emplace(*jit_optimize);
    }
    // Load straight onto the target device instead of staging through CPU.
    auto model = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path, device));
    model->eval();
    // Note: this is a full dtype conversion of the whole module (all weights and
    // buffers), not autocast-style mixed precision (see setPrecision for that).
    model->to(dtype);
    model_ = std::move(model);
    path_ = model_path;
    device_ = device;
    jit_optimize_ = jit_optimize;
}

std::vector<torch::Tensor> TorchScriptBackend::forward(const std::vector<torch::jit::IValue>& inputs) {
    return iValueToTensorOutputs(runMethod("forward", inputs));
}

torch::jit::IValue TorchScriptBackend::runMethod(const std::string& name,
                                                 std::vector<torch::jit::IValue> inputs) {
    return runMethod(name, std::move(inputs), precision_);
}

torch::jit::IValue TorchScriptBackend::runMethod(const std::string& name,
                                                 std::vector<torch::jit::IValue> inputs,
                                                 const PrecisionSpec& spec) {
    if (!model_) {
        throw std::runtime_error("TorchScriptBackend: not loaded");
    }
    std::optional<torch::jit::Method> method = model_->find_method(name);
    if (!method) {
        throw std::invalid_argument("TorchScriptBackend: model has no method \"" + name + "\"");
    }
    torch::InferenceMode inference;
    std::optional<torch::jit::GraphOptimizerEnabledGuard> executor;
    if (jit_optimize_) {
        executor.emplace(*jit_optimize_);
    }
    ScopedTf32 tf32(spec.tf32);
    CudaAutocastGuard autocast(device_.is_cuda() && spec.autocastApplies(name),
                               autocastScalarType(spec.autocast));
    return (*method)(std::move(inputs));
}

bool TorchScriptBackend::hasMethod(const std::string& name) const {
    return model_ && model_->find_method(name).has_value();
}

bool TorchScriptBackend::isLoaded() const {
    return model_ != nullptr;
}

const std::string& TorchScriptBackend::loadedPath() const {
    return path_;
}

}  // namespace nuketorch::torch_worker
