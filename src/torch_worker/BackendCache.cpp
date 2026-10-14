#include <nuketorch/torch_worker/BackendCache.h>

#include <nuketorch/ScopedTimer.h>
#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/TorchWorkerUtils.h>

#include <ATen/Context.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <sstream>
#include <stdexcept>
#include <tuple>

namespace nuketorch::torch_worker {
namespace {

std::string deviceToString(const torch::Device& d) {
    std::ostringstream oss;
    oss << d;
    return oss.str();
}

PrecisionTarget precisionTarget(const torch::Device& device, bool baked) {
    PrecisionTarget t;
    t.baked = baked;
    if (device.is_cuda()) {
        t.device = PrecisionTarget::Device::cuda;
        std::tie(t.cc_major, t.cc_minor) = computeCapability(device);
    } else if (device.is_cpu()) {
        t.device = PrecisionTarget::Device::cpu;
    } else {
        t.device = PrecisionTarget::Device::other;
    }
    return t;
}

}  // namespace

void BackendCache::ensure(const FrameHeader& header,
                          const std::unordered_map<std::string, std::string>& params,
                          nuketorch::InferenceMetrics* metrics) {
    const std::string kind = backendNameFromParams(params);
    torch::Device device = resolveDevice(header.use_gpu);
    const PrecisionSpec requested = precisionSpecFromParams(params, header.mixed_precision);
    const PrecisionTarget target = precisionTarget(device, kind != kBackendTorchScript);
    std::string fallback_note;
    const PrecisionSpec spec = resolvePrecision(requested, target, &fallback_note);
    const torch::ScalarType dtype = spec.half_weights ? torch::kFloat16 : torch::kFloat32;
    // Executor flag of a (re)load: explicit, else the calling thread's own.
    const bool jit_optimize = spec.jit_optimize.value_or(torch::jit::getGraphExecutorOptimize());

    // Process-wide numerics on every request, before any (re)load.
    applyTf32(spec.tf32);
    if (spec.cudnn_benchmark) {
        at::globalContext().setBenchmarkCuDNN(*spec.cudnn_benchmark);
    }

    const bool need_switch_kind = !impl_ || kind != backend_kind_;
    // A method run once keeps the plan of the executor flag it first ran
    // under, so a jit_optimize change needs a fresh module.
    const bool need_reload = need_switch_kind || !impl_->isLoaded() || impl_->loadedPath() != header.model_path ||
                             device_ != device || dtype_ != dtype ||
                             (kind == kBackendTorchScript && jit_optimize_ != jit_optimize);

    if (need_switch_kind) {
        impl_ = createBackend(kind);
        backend_kind_ = kind;
    }
    impl_->setPrecision(spec);
    if (need_reload) {
        double load_ms = 0;
        {
            nuketorch::ScopedTimer t(load_ms);
            impl_->load(header.model_path, device, dtype);
        }
        model_path_ = header.model_path;
        device_ = device;
        dtype_ = dtype;
        jit_optimize_ = jit_optimize;
        if (metrics) {
            metrics->model_load_ms = load_ms;
        }
    } else if (metrics) {
        metrics->model_load_ms = -1;
    }

    const std::string label = precisionLabel(spec, target.baked, fallback_note);
    if (!fallback_note.empty() && (fallback_note != fallback_note_ || need_reload)) {
        workerLog("warning", "precision_policy=fallback_fp32: running " + label + " on " +
                                 deviceToString(device_) + " instead of " + fallback_note);
    }
    if (header.debug && need_reload) {
        workerLog("debug", "loaded " + header.model_path + " (" + backend_kind_ + ", " +
                               deviceToString(device_) + ", " + label + ")");
    }
    spec_ = spec;
    label_ = label;
    fallback_note_ = fallback_note;

    if (metrics) {
        metrics->backend = backend_kind_;
        metrics->device = deviceToString(device_);
        metrics->dtype = label_;
        if (!fallback_note.empty()) {
            metrics->notes += (metrics->notes.empty() ? "" : "; ") + std::string("precision fallback: ") +
                              fallback_note + " -> float32/ieee";
        }
    }
}

InferenceBackend& BackendCache::backend() {
    if (!impl_ || !impl_->isLoaded()) {
        throw std::runtime_error("BackendCache: no loaded backend");
    }
    return *impl_;
}

}  // namespace nuketorch::torch_worker
