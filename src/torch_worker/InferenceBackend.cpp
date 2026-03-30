#include <nuketorch/torch_worker/InferenceBackend.h>

#include <nuketorch/torch_worker/AOTInductorBackend.h>
#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/TorchScriptBackend.h>

#ifdef NUKETORCH_TORCH_WORKER_HAS_TENSORRT
#include <nuketorch/torch_worker/TensorRTBackend.h>
#endif

#include <stdexcept>

namespace nuketorch::torch_worker {

std::unique_ptr<InferenceBackend> createBackend(const std::string& canonical_backend_name) {
#ifdef NUKETORCH_TORCH_WORKER_HAS_TENSORRT
    if (canonical_backend_name == kBackendTensorRT) {
        return std::make_unique<TensorRTBackend>();
    }
#else
    if (canonical_backend_name == kBackendTensorRT) {
        throw std::runtime_error(
            "createBackend: TensorRT backend was not built (set NUKETORCH_TORCH_WORKER_HAS_TENSORRT=ON)");
    }
#endif
    if (canonical_backend_name == kBackendAOTInductor) {
        return std::make_unique<AOTInductorBackend>();
    }
    if (canonical_backend_name == kBackendTorchScript) {
        return std::make_unique<TorchScriptBackend>();
    }
    throw std::runtime_error("createBackend: unknown backend \"" + canonical_backend_name + "\"");
}

}  // namespace nuketorch::torch_worker
