#include <nuketorch/torch_worker/BackendParams.h>

#include <algorithm>
#include <cctype>

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
    if (v == "torchscript" || v == "jit" || v == "ts") {
        return kBackendTorchScript;
    }
    return kBackendTorchScript;
}

}  // namespace nuketorch::torch_worker
