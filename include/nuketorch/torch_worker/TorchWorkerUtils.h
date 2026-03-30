#pragma once

#include <torch/torch.h>

namespace nuketorch::torch_worker {

/// Maps `use_gpu` to CUDA, MPS, or CPU (same policy as reference nnRetime worker).
torch::Device resolveDevice(bool use_gpu);

/// Pads H/W to the next multiple of `factor` with reflect padding (NCHW).
torch::Tensor padTensorToMultiple(const torch::Tensor& input_tensor, int64_t factor);

}  // namespace nuketorch::torch_worker
