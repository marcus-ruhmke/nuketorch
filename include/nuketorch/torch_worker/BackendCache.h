#pragma once

#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/Protocol.h>
#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/InferenceBackend.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace nuketorch::torch_worker {

/// Owns an `InferenceBackend`, reloads when path/device/dtype/backend or the
/// TorchScript executor flag (`jit_optimize`) change, and applies the request's
/// precision (see precisionSpecFromParams) on every call: TF32 and cuDNN
/// benchmark are process-wide and set before any (re)load.
class BackendCache {
public:
    /// Throws std::invalid_argument for bad or unsupported precision params
    /// (-> `ERR|bad_request`).
    /// @param metrics Optional; `model_load_ms` is set when a load occurs, otherwise `-1`.
    /// Backend/device/dtype (precisionLabel) filled on success; a fallback is added to `notes`.
    void ensure(const FrameHeader& header,
                const std::unordered_map<std::string, std::string>& params,
                nuketorch::InferenceMetrics* metrics = nullptr);

    InferenceBackend& backend();

    /// Precision in effect after the last ensure() (device checks and
    /// fallback applied).
    const PrecisionSpec& spec() const { return spec_; }
    /// Metrics dtype label of the last ensure().
    const std::string& dtypeLabel() const { return label_; }
    /// What precision_policy=fallback_fp32 dropped in the last ensure(); empty if nothing.
    const std::string& fallbackNote() const { return fallback_note_; }

private:
    std::unique_ptr<InferenceBackend> impl_;
    std::string backend_kind_;
    std::string model_path_;
    torch::Device device_ = torch::kCPU;
    torch::ScalarType dtype_ = torch::kFloat32;
    bool jit_optimize_ = true;
    PrecisionSpec spec_;
    std::string label_;
    std::string fallback_note_;
};

}  // namespace nuketorch::torch_worker
