#pragma once

#include <chrono>

namespace nuketorch {

/// RAII wall-clock timer in milliseconds using `std::chrono::steady_clock`.
class ScopedTimer {
public:
    explicit ScopedTimer(double& target_ms)
        : target_(target_ms), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        const auto end = std::chrono::steady_clock::now();
        target_ = std::chrono::duration<double, std::milli>(end - start_).count();
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    double& target_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace nuketorch
