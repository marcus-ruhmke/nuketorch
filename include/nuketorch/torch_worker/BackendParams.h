#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

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

/// TF32 for cuBLAS matmuls and cuDNN convolutions/RNNs (process-wide flags).
enum class Tf32 {
    /// Leave the process boot state in place (libtorch default: cuDNN TF32 ON,
    /// cuBLAS TF32 OFF — so a plain float32 request runs convolutions in TF32).
    keep,
    /// TF32 for both (`torch.backends.{cuda.matmul,cudnn}.allow_tf32 = True`).
    on,
    /// True IEEE float32 for both.
    off,
};

/// Per-op autocast dtype applied at call time (weights stay float32).
enum class AutocastDtype { none, float16, bfloat16 };

/// What to do when the requested precision cannot run on the device.
enum class PrecisionPolicy {
    /// std::invalid_argument naming the mode and the reason (-> `ERR|bad_request`).
    error,
    /// Run IEEE float32 instead; logged to stderr and labelled in metrics.
    fallback_fp32,
};

/// Full numeric state of a torch_worker request, parsed from `params` by
/// precisionSpecFromParams(). Defaults reproduce the pre-0.3 behaviour.
struct PrecisionSpec {
    /// Convert the whole module to FP16 at load (`precision=half`; GPU only).
    bool half_weights = false;
    /// Per-op CUDA autocast at call time (`autocast`/`amp` = float16, `bf16` = bfloat16).
    AutocastDtype autocast = AutocastDtype::none;
    /// Methods that run under autocast; empty = every method.
    std::vector<std::string> autocast_methods;
    Tf32 tf32 = Tf32::keep;
    /// cuDNN autotuning (process-wide); nullopt leaves it untouched.
    std::optional<bool> cudnn_benchmark;
    /// TorchScript graph executor optimisation for every call into the module,
    /// from its load on (thread-local flag, held by a guard around each call).
    /// nullopt leaves the calling thread's flag untouched (libtorch default:
    /// on). The profiling executor fixes a method's plan on its first call, so
    /// a change means a reload.
    std::optional<bool> jit_optimize;
    PrecisionPolicy policy = PrecisionPolicy::error;

    /// True when @p method runs under autocast.
    bool autocastApplies(const std::string& method) const;

    /// True IEEE float32: float32 weights, no autocast, TF32 off (e.g. to
    /// recompute a non-finite result).
    static PrecisionSpec ieeeFloat32();

    bool operator==(const PrecisionSpec& o) const;
    bool operator!=(const PrecisionSpec& o) const { return !(*this == o); }
};

/// Parses the precision keys of @p params (all values case-insensitive):
/// - `precision`: `half`/`fp16`/`float16` (FP16 weights), `autocast`/`amp`
///   (float16 autocast), `bf16`/`bfloat16` (bfloat16 autocast),
///   `float32`/`fp32`/`float`/`full`, and the sugar `tf32` (= float32 +
///   `tf32=on`) / `ieee` (= float32 + `tf32=off`). Absent: `half` when
///   @p mixed_precision is set, else `float32` (pre-0.3 behaviour).
/// - `tf32`: `on`/`off` (absent: keep).
/// - `autocast_methods`: comma-separated method names (absent: all); needs an
///   autocast precision.
/// - `jit_optimize`: `0`/`1` (absent: leave the thread's flag untouched).
/// - `cudnn_benchmark`: `0`/`1` (absent: keep).
/// - `precision_policy`: `error`/`fallback_fp32` (absent: error).
/// Throws std::invalid_argument for unrecognized or contradictory values.
PrecisionSpec precisionSpecFromParams(const std::unordered_map<std::string, std::string>& params,
                                      bool mixed_precision);

/// Where a spec is about to run, as far as precision support is concerned.
struct PrecisionTarget {
    enum class Device { cpu, cuda, other } device = Device::cpu;
    /// CUDA compute capability (only read for Device::cuda).
    int cc_major = 0;
    int cc_minor = 0;
    /// AOTInductor/TensorRT: precision is compiled into the artifact.
    bool baked = false;
};

/// Validates @p requested against @p target and returns the spec to run.
/// - half weights apply on GPUs only (CPU: float32, as before).
/// - Legacy `autocast` without `autocast_methods` silently runs float32 off
///   CUDA (as before).
/// - bf16, `tf32=on` and `autocast_methods` need CUDA; bf16 and `tf32=on` need
///   compute capability >= 8.0. Otherwise std::invalid_argument naming the
///   mode and the reason, unless the policy is `fallback_fp32`: then the
///   result is IEEE float32 and @p fallback_note (if given) names what was
///   dropped (e.g. "bf16"); it is cleared otherwise.
/// - Baked backends refuse `autocast_methods` (they only run `forward`).
PrecisionSpec resolvePrecision(const PrecisionSpec& requested, const PrecisionTarget& target,
                               std::string* fallback_note = nullptr);

/// Metrics dtype label of a resolved spec (always < 64 characters):
/// legacy requests keep "float32" / "float16" / "autocast(float16)"; an
/// explicit TF32 choice gives "float32/ieee" / "float32/tf32"; autocast shows
/// dtype and method scope ("autocast(bfloat16;features)"); a fallback appends
/// "(fallback:<note>)"; an explicit `jit_optimize=0` appends ",jit0". Baked
/// backends report "baked" for non-legacy requests.
std::string precisionLabel(const PrecisionSpec& resolved, bool baked = false,
                           const std::string& fallback_note = {});

/// Pre-0.3 precision strategy, kept for existing workers. Prefer PrecisionSpec.
enum class PrecisionMode {
    /// Convert the whole module to FP16 on GPU (the legacy `mixed_precision=true` behavior).
    half,
    /// Keep FP32 weights and autocast per-op at call time (CUDA only; falls
    /// back to full FP32 on other devices). Also returned for `bf16`.
    autocast,
    /// Full FP32 (also for the `tf32` / `ieee` sugar).
    full,
};

/// Reads `params["precision"]` only. Absent: `half` when @p mixed_precision
/// is set, else `full`. Throws std::invalid_argument for unrecognized values.
/// The weights dtype it implies (`half` -> FP16 on GPU) matches BackendCache.
PrecisionMode precisionModeFromParams(const std::unordered_map<std::string, std::string>& params,
                                      bool mixed_precision);

}  // namespace nuketorch::torch_worker
