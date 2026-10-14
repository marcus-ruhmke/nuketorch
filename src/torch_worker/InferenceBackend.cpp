#include <nuketorch/torch_worker/InferenceBackend.h>

#include <nuketorch/torch_worker/AOTInductorBackend.h>
#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/TorchScriptBackend.h>

#ifdef NUKETORCH_TORCH_WORKER_HAS_TENSORRT
#include <nuketorch/torch_worker/TensorRTBackend.h>
#endif

#include <stdexcept>

namespace nuketorch::torch_worker {

torch::jit::IValue InferenceBackend::runMethod(const std::string& name,
                                               std::vector<torch::jit::IValue> inputs) {
    if (name != "forward") {
        throw std::invalid_argument("backend runs only forward, not \"" + name +
                                    "\" (named methods need the torchscript backend)");
    }
    std::vector<torch::Tensor> outputs = forward(inputs);
    if (outputs.size() == 1) {
        return outputs[0];
    }
    return c10::ivalue::Tuple::create(std::vector<torch::jit::IValue>(outputs.begin(), outputs.end()));
}

torch::jit::IValue InferenceBackend::runMethod(const std::string& name,
                                               std::vector<torch::jit::IValue> inputs,
                                               const PrecisionSpec& spec) {
    (void)spec;  // precision is compiled into the artifact
    return runMethod(name, std::move(inputs));
}

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
