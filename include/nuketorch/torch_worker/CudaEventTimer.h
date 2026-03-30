#pragma once

#include <cuda_runtime_api.h>

namespace nuketorch::torch_worker {

/// CUDA event pair for timing GPU work on a stream. If event creation fails, `elapsed_ms()` returns `-1`.
class CudaEventTimer {
public:
    explicit CudaEventTimer(cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start();
    void stop();
    /// Milliseconds between the two recorded events, or `-1` if invalid / not stopped.
    float elapsed_ms() const;

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
    bool valid_ = false;
    bool stopped_ = false;
};

}  // namespace nuketorch::torch_worker
