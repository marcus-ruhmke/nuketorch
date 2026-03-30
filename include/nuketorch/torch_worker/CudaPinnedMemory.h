#pragma once

#include <cstddef>

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#define NUKETORCH_TORCH_WORKER_HAS_CUDA_RUNTIME 1
#endif

namespace nuketorch::torch_worker {

/// Registers host memory with CUDA for faster async copies when the worker uses the GPU.
class CudaPinnedMemory {
public:
    CudaPinnedMemory(void* ptr, size_t size, bool use_gpu);
    ~CudaPinnedMemory();

    CudaPinnedMemory(const CudaPinnedMemory&) = delete;
    CudaPinnedMemory& operator=(const CudaPinnedMemory&) = delete;

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
    bool is_pinned_ = false;
};

}  // namespace nuketorch::torch_worker
