#include <nuketorch/torch_worker/BackendParams.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace nuketorch::torch_worker {

namespace {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

}  // namespace

std::string backendNameFromParams(const std::unordered_map<std::string, std::string>& params) {
    const auto it = params.find("backend");
    if (it == params.end()) {
        return kBackendTorchScript;
    }
    const std::string v = toLower(it->second);
    if (v == "aotinductor" || v == "aoti" || v == "inductor") {
        return kBackendAOTInductor;
    }
    if (v == "tensorrt" || v == "trt") {
        return kBackendTensorRT;
    }
    if (v == "torchscript" || v == "jit" || v == "ts") {
        return kBackendTorchScript;
    }
    // A typo must not silently fall back to TorchScript and then fail with a
    // baffling "invalid TorchScript archive" on a .engine/.pt2 file.
    throw std::invalid_argument("unknown backend \"" + it->second +
                                "\" (expected torchscript, aotinductor, or tensorrt)");
}

}  // namespace nuketorch::torch_worker
