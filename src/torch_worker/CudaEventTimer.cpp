#include <nuketorch/torch_worker/CudaEventTimer.h>

namespace nuketorch::torch_worker {

CudaEventTimer::CudaEventTimer(cudaStream_t stream) : stream_(stream) {
    if (cudaEventCreate(&start_event_) != cudaSuccess) {
        return;
    }
    if (cudaEventCreate(&stop_event_) != cudaSuccess) {
        cudaEventDestroy(start_event_);
        start_event_ = nullptr;
        return;
    }
    valid_ = true;
}

CudaEventTimer::~CudaEventTimer() {
    if (start_event_) {
        cudaEventDestroy(start_event_);
    }
    if (stop_event_) {
        cudaEventDestroy(stop_event_);
    }
}

void CudaEventTimer::start() {
    if (!valid_) {
        return;
    }
    stopped_ = false;
    (void)cudaEventRecord(start_event_, stream_);
}

void CudaEventTimer::stop() {
    if (!valid_) {
        return;
    }
    (void)cudaEventRecord(stop_event_, stream_);
    (void)cudaEventSynchronize(stop_event_);
    stopped_ = true;
}

float CudaEventTimer::elapsed_ms() const {
    if (!valid_ || !stopped_) {
        return -1.f;
    }
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, start_event_, stop_event_) != cudaSuccess) {
        return -1.f;
    }
    return ms;
}

}  // namespace nuketorch::torch_worker
