#include <nuketorch/torch_worker/InferenceBackend.h>

#include <nuketorch/torch_worker/AOTInductorBackend.h>
#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/TorchScriptBackend.h>

#include <stdexcept>

namespace nuketorch::torch_worker {

std::unique_ptr<InferenceBackend> createBackend(const std::string& canonical_backend_name) {
    if (canonical_backend_name == kBackendAOTInductor) {
        return std::make_unique<AOTInductorBackend>();
    }
    if (canonical_backend_name == kBackendTorchScript) {
        return std::make_unique<TorchScriptBackend>();
    }
    throw std::runtime_error("createBackend: unknown backend \"" + canonical_backend_name + "\"");
}

}  // namespace nuketorch::torch_worker
