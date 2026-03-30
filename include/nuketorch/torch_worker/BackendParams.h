#pragma once

#include <string>
#include <unordered_map>

namespace nuketorch::torch_worker {

/// Default inference backend when `params["backend"]` is absent.
inline constexpr const char* kBackendTorchScript = "torchscript";
/// AOTInductor packaged artifact (`.pt2` per PyTorch docs).
inline constexpr const char* kBackendAOTInductor = "aotinductor";
/// TensorRT serialized engine (`.engine`).
inline constexpr const char* kBackendTensorRT = "tensorrt";

/// Reads `params["backend"]` (case-insensitive) and returns a canonical backend name.
std::string backendNameFromParams(const std::unordered_map<std::string, std::string>& params);

}  // namespace nuketorch::torch_worker
