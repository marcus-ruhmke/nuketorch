#pragma once

#include <torch/torch.h>

namespace nuketorch::torch_worker {

/// Apply CustomLogTransferFunction:  y = 1 - (1 + a*x)^(-b).
///
/// Mirrors `dataset_transforms.CustomLogTransferFunction` in InterpAny-Clearer.
/// Defaults match the EMA-d training-time encoding (a = 3.0, b = 1.0); with these
/// defaults the implementation uses the algebraic shortcut y = 1 - 1/(1 + a*x)
/// to avoid a torch::pow call.
///
/// Element-wise; preserves input dtype, device, and shape.
torch::Tensor applyCustomLog(const torch::Tensor& x, double a = 3.0, double b = 1.0);

/// Inverse of applyCustomLog:  x = ((1 - y)^(-1/b) - 1) / a.
///
/// `1 - y` is clamped to >= `eps` (default 1e-9) so the inverse stays finite when
/// the network occasionally predicts y >= 1. Note: in fp16 the smallest positive
/// subnormal is ~6e-8, so a 1e-9 clamp is a no-op there; callers that need full
/// out-of-range protection should cast `y` to fp32 before calling.
///
/// Element-wise; preserves input dtype, device, and shape.
torch::Tensor applyCustomLogInverse(const torch::Tensor& y,
                                    double a = 3.0,
                                    double b = 1.0,
                                    double eps = 1e-9);

}  // namespace nuketorch::torch_worker
