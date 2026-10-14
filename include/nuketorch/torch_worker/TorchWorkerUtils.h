#pragma once

#include <nuketorch/torch_worker/BackendParams.h>

#include <ATen/core/ivalue.h>
#include <torch/torch.h>

#include <functional>
#include <string>
#include <utility>

namespace nuketorch::torch_worker {

/// Maps `use_gpu` to CUDA, MPS, or CPU (same policy as reference nnRetime worker).
torch::Device resolveDevice(bool use_gpu);

/// Pads H/W to the next multiple of `factor` with reflect padding (NCHW).
torch::Tensor padTensorToMultiple(const torch::Tensor& input_tensor, int64_t factor);

/// Sets the process-wide TF32 flags through the legacy pair
/// setAllowTF32CuBLAS / setAllowTF32CuDNN (= Python's
/// `torch.backends.cuda.matmul.allow_tf32` / `torch.backends.cudnn.allow_tf32`;
/// libtorch >= 2.9 maps them onto the CUDA matmul/conv/rnn fp32 precision, so
/// on/off also equals `set_float32_matmul_precision("high"/"highest")`).
/// Tf32::keep restores the flags the process had before the first on/off and
/// is a no-op until then. Process-global: call from one thread only.
void applyTf32(Tf32 mode);

/// Applies @p mode for a scope and restores the previous flags; Tf32::keep
/// leaves them untouched.
class ScopedTf32 {
public:
    explicit ScopedTf32(Tf32 mode);
    ~ScopedTf32();

    ScopedTf32(const ScopedTf32&) = delete;
    ScopedTf32& operator=(const ScopedTf32&) = delete;

private:
    bool active_ = false;
    bool prev_cublas_ = false;
    bool prev_cudnn_ = false;
};

/// CUDA compute capability of @p device as {major, minor}.
std::pair<int, int> computeCapability(const torch::Device& device);

using ComputeCapabilityProbe = std::function<std::pair<int, int>(const torch::Device&)>;

/// Replaces what computeCapability() reports (tests: pretend a 7.5 card);
/// nullptr restores the real query.
void setComputeCapabilityProbe(ComputeCapabilityProbe probe);

/// True when every floating-point tensor in @p value (a tensor, or nested
/// tuples / lists / dict values) is finite; other values are ignored.
/// Synchronizes with the device.
bool allFinite(const c10::IValue& value);

/// Writes "[nuketorch] <level>: <message>" to stderr (the host keeps the
/// worker's stderr tail for crash reports). Gate debug lines on header.debug.
void workerLog(const std::string& level, const std::string& message);

}  // namespace nuketorch::torch_worker
