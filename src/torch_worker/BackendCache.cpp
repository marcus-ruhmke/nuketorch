#include <nuketorch/torch_worker/BackendCache.h>

#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/TorchWorkerUtils.h>

#include <stdexcept>

namespace nuketorch::torch_worker {

void BackendCache::ensure(const FrameHeader& header,
                          const std::unordered_map<std::string, std::string>& params) {
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
        impl_->load(header.model_path, device, dtype);
        model_path_ = header.model_path;
        device_ = device;
        dtype_ = dtype;
    }
}

InferenceBackend& BackendCache::backend() {
    if (!impl_ || !impl_->isLoaded()) {
        throw std::runtime_error("BackendCache: no loaded backend");
    }
    return *impl_;
}

}  // namespace nuketorch::torch_worker
