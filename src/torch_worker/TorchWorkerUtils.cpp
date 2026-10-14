#include <nuketorch/torch_worker/TorchWorkerUtils.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <mutex>

namespace nuketorch::torch_worker {

namespace {

void setTf32Flags(bool cublas, bool cudnn) {
    at::globalContext().setAllowTF32CuBLAS(cublas);
    at::globalContext().setAllowTF32CuDNN(cudnn);
}

std::mutex& probeMutex() {
    static std::mutex m;
    return m;
}

ComputeCapabilityProbe& probeOverride() {
    static ComputeCapabilityProbe probe;
    return probe;
}

}  // namespace

torch::Device resolveDevice(bool use_gpu) {
    if (!use_gpu) {
        return torch::kCPU;
    }
    if (torch::cuda::is_available()) {
        // Return a Device with an explicit index. `torch::kCUDA` alone
        // implicitly converts to a Device whose index() is -1 ("unspecified"),
        // which works for tensor.to(device) but fails any API that calls
        // device.index() directly (e.g. CUDACachingAllocator::resetPeakStats /
        // getDeviceStats), with: "Invalid device argument: did you call init?"
        return torch::Device(torch::kCUDA, at::cuda::current_device());
    }
    if (torch::mps::is_available()) {
        return torch::Device(torch::kMPS, 0);
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

void applyTf32(Tf32 mode) {
    // Snapshot before nuketorch first touches the flags: libtorch boots with
    // cuDNN TF32 on and cuBLAS TF32 off (env TORCH_ALLOW_TF32_CUBLAS_OVERRIDE).
    static const bool boot_cublas = at::globalContext().allowTF32CuBLAS();
    static const bool boot_cudnn = at::globalContext().allowTF32CuDNN();
    static bool touched = false;
    if (mode == Tf32::keep) {
        if (touched) {
            setTf32Flags(boot_cublas, boot_cudnn);
            touched = false;
        }
        return;
    }
    const bool on = mode == Tf32::on;
    setTf32Flags(on, on);
    touched = true;
}

ScopedTf32::ScopedTf32(Tf32 mode) : active_(mode != Tf32::keep) {
    if (!active_) {
        return;
    }
    prev_cublas_ = at::globalContext().allowTF32CuBLAS();
    prev_cudnn_ = at::globalContext().allowTF32CuDNN();
    const bool on = mode == Tf32::on;
    setTf32Flags(on, on);
}

ScopedTf32::~ScopedTf32() {
    if (active_) {
        setTf32Flags(prev_cublas_, prev_cudnn_);
    }
}

std::pair<int, int> computeCapability(const torch::Device& device) {
    {
        std::lock_guard<std::mutex> lock(probeMutex());
        if (probeOverride()) {
            return probeOverride()(device);
        }
    }
    const cudaDeviceProp* prop =
        at::cuda::getDeviceProperties(device.has_index() ? device.index() : at::cuda::current_device());
    return {prop->major, prop->minor};
}

void setComputeCapabilityProbe(ComputeCapabilityProbe probe) {
    std::lock_guard<std::mutex> lock(probeMutex());
    probeOverride() = std::move(probe);
}

bool allFinite(const c10::IValue& value) {
    if (value.isTensor()) {
        const at::Tensor& t = value.toTensor();
        if (!t.defined() || !(t.is_floating_point() || t.is_complex())) {
            return true;
        }
        return t.isfinite().all().item<bool>();
    }
    if (value.isTuple()) {
        for (const auto& e : value.toTupleRef().elements()) {
            if (!allFinite(e)) {
                return false;
            }
        }
        return true;
    }
    if (value.isList()) {
        for (const c10::IValue& e : value.toListRef()) {
            if (!allFinite(e)) {
                return false;
            }
        }
        return true;
    }
    if (value.isGenericDict()) {
        for (const auto& kv : value.toGenericDict()) {
            if (!allFinite(kv.value())) {
                return false;
            }
        }
    }
    return true;
}

void workerLog(const std::string& level, const std::string& message) {
    std::cerr << "[nuketorch] " << level << ": " << message << std::endl;
}

}  // namespace nuketorch::torch_worker
