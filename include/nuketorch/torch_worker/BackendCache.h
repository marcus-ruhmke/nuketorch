#pragma once

#include <nuketorch/Protocol.h>
#include <nuketorch/torch_worker/InferenceBackend.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace nuketorch::torch_worker {

/// Owns an `InferenceBackend`, reloads when path/device/dtype/backend change.
class BackendCache {
public:
    void ensure(const FrameHeader& header, const std::unordered_map<std::string, std::string>& params);

    InferenceBackend& backend();

private:
    std::unique_ptr<InferenceBackend> impl_;
    std::string backend_kind_;
    std::string model_path_;
    torch::Device device_ = torch::kCPU;
    torch::ScalarType dtype_ = torch::kFloat32;
};

}  // namespace nuketorch::torch_worker
