#include <nuketorch/torch_worker/BackendCache.h>

#include <nuketorch/ScopedTimer.h>
#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/TorchWorkerUtils.h>

#include <sstream>
#include <stdexcept>

namespace nuketorch::torch_worker {
namespace {

std::string deviceToString(const torch::Device& d) {
    std::ostringstream oss;
    oss << d;
    return oss.str();
}

std::string scalarTypeLabel(torch::ScalarType st) {
    if (st == torch::kFloat16) {
        return "float16";
    }
    if (st == torch::kFloat32) {
        return "float32";
    }
    return "unknown";
}

}  // namespace

void BackendCache::ensure(const FrameHeader& header,
                          const std::unordered_map<std::string, std::string>& params,
                          nuketorch::InferenceMetrics* metrics) {
    const std::string kind = backendNameFromParams(params);
    torch::Device device = resolveDevice(header.use_gpu);
    const bool use_half = header.mixed_precision && device.type() != torch::kCPU;
    const torch::ScalarType dtype = use_half ? torch::kFloat16 : torch::kFloat32;

    const bool need_switch_kind = !impl_ || kind != backend_kind_;
    const bool need_reload = need_switch_kind || !impl_->isLoaded() || impl_->loadedPath() != header.model_path ||
                             device_ != device || dtype_ != dtype;

    if (need_switch_kind) {
        impl_ = createBackend(kind);
        backend_kind_ = kind;
    }
    if (need_reload) {
        double load_ms = 0;
        {
            nuketorch::ScopedTimer t(load_ms);
            impl_->load(header.model_path, device, dtype);
        }
        model_path_ = header.model_path;
        device_ = device;
        dtype_ = dtype;
        if (metrics) {
            metrics->model_load_ms = load_ms;
        }
    } else if (metrics) {
        metrics->model_load_ms = -1;
    }

    if (metrics) {
        metrics->backend = backend_kind_;
        metrics->device = deviceToString(device_);
        metrics->dtype = scalarTypeLabel(dtype_);
    }
}

InferenceBackend& BackendCache::backend() {
    if (!impl_ || !impl_->isLoaded()) {
        throw std::runtime_error("BackendCache: no loaded backend");
    }
    return *impl_;
}

}  // namespace nuketorch::torch_worker
