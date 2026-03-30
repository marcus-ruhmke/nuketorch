#include <nuketorch/torch_worker/TorchWorkerUtils.h>

namespace nuketorch::torch_worker {

torch::Device resolveDevice(bool use_gpu) {
    if (!use_gpu) {
        return torch::kCPU;
    }
    if (torch::cuda::is_available()) {
        return torch::kCUDA;
    }
    if (torch::mps::is_available()) {
        return torch::kMPS;
    }
    return torch::kCPU;
}

torch::Tensor padTensorToMultiple(const torch::Tensor& input_tensor, int64_t factor) {
    const auto sizes = input_tensor.sizes();
    const int64_t h = sizes[2];
    const int64_t w = sizes[3];
    const int64_t pad_h = ((h - 1) / factor + 1) * factor;
    const int64_t pad_w = ((w - 1) / factor + 1) * factor;
    const int64_t pad_bottom = pad_h - h;
    const int64_t pad_right = pad_w - w;
    if (pad_bottom == 0 && pad_right == 0) {
        return input_tensor;
    }
    // Reflect padding requires pad < spatial extent; fall back to replicate for tiny inputs.
    const bool reflect_safe = pad_bottom < h && pad_right < w;
    if (reflect_safe) {
        return torch::nn::functional::pad(
            input_tensor,
            torch::nn::functional::PadFuncOptions({0, pad_right, 0, pad_bottom}).mode(torch::kReflect));
    }
    return torch::nn::functional::pad(
        input_tensor,
        torch::nn::functional::PadFuncOptions({0, pad_right, 0, pad_bottom}).mode(torch::kReplicate));
}

}  // namespace nuketorch::torch_worker
