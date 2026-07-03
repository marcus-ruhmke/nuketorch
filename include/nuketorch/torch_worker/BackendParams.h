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
/// Throws std::invalid_argument for unrecognized values; absent means TorchScript.
std::string backendNameFromParams(const std::unordered_map<std::string, std::string>& params);

/// Precision strategy for torch_worker backends, from `params["precision"]`.
enum class PrecisionMode {
    /// Convert the whole module to FP16 on GPU (the legacy `mixed_precision=true` behavior).
    half,
    /// Keep FP32 weights and autocast per-op to FP16 at forward time (CUDA only;
    /// falls back to full FP32 on other devices). Real mixed precision: reductions
    /// and other precision-sensitive ops stay FP32.
    autocast,
    /// Full FP32.
    full,
};

/// Reads `params["precision"]` (case-insensitive: "half"/"fp16", "autocast"/"amp",
/// "float32"/"fp32"/"full"). Absent: `half` when @p mixed_precision is set, else
/// `full` — existing consumers keep their numerics unchanged. Throws
/// std::invalid_argument for unrecognized values.
PrecisionMode precisionModeFromParams(const std::unordered_map<std::string, std::string>& params,
                                      bool mixed_precision);

}  // namespace nuketorch::torch_worker
