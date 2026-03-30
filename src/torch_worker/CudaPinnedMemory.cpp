#include <nuketorch/torch_worker/CudaPinnedMemory.h>

namespace nuketorch::torch_worker {

CudaPinnedMemory::CudaPinnedMemory(void* ptr, size_t size, bool use_gpu)
    : ptr_(ptr), size_(size), is_pinned_(false) {
#ifdef NUKETORCH_TORCH_WORKER_HAS_CUDA_RUNTIME
    if (use_gpu && ptr_ != nullptr && size_ > 0) {
        cudaError_t err = cudaHostRegister(ptr_, size_, cudaHostRegisterDefault);
        if (err == cudaSuccess) {
            is_pinned_ = true;
        }
    }
#else
    (void)use_gpu;
#endif
}

CudaPinnedMemory::~CudaPinnedMemory() {
#ifdef NUKETORCH_TORCH_WORKER_HAS_CUDA_RUNTIME
    if (is_pinned_ && ptr_ != nullptr) {
        cudaHostUnregister(ptr_);
    }
#endif
}

}  // namespace nuketorch::torch_worker
