#include <nuketorch/torch_worker/CustomLogTransfer.h>

namespace nuketorch::torch_worker {

namespace {
constexpr double kDefaultB = 1.0;
}  // namespace

torch::Tensor applyCustomLog(const torch::Tensor& x, double a, double b) {
    if (b == kDefaultB) {
        return 1.0 - 1.0 / (1.0 + a * x);
    }
    return 1.0 - torch::pow(1.0 + a * x, -b);
}

torch::Tensor applyCustomLogInverse(const torch::Tensor& y, double a, double b, double eps) {
    auto one_minus_y = (1.0 - y).clamp(eps, 1.0);
    if (b == kDefaultB) {
        return (1.0 / one_minus_y - 1.0) / a;
    }
    return (torch::pow(one_minus_y, -1.0 / b) - 1.0) / a;
}

}  // namespace nuketorch::torch_worker
