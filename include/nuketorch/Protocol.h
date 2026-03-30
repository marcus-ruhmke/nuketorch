#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nuketorch {

/// Image dimensions and run flags carried with every inference request (host and worker agree on these fields).
struct FrameHeader {
    /// Pixel width.
    int width = 0;
    /// Pixel height.
    int height = 0;
    /// Number of channels (e.g. 3 for RGB).
    int channels = 0;
    /// Prefer CUDA/MPS when available in the worker.
    bool use_gpu = true;
    /// Use half precision on GPU when supported.
    bool mixed_precision = true;
    /// If true, workers may enable verbose logging.
    bool debug = false;
    /// Path to the model artifact on the worker machine (e.g. TorchScript `.pt`, AOTInductor `.pt2`).
    std::string model_path;
};

/// One inference job: where frame data lives in shared memory plus model-specific parameters as strings.
struct InferenceRequest {
    FrameHeader header;
    /// One POSIX shm name per input plane (length `num_inputs` on the wire).
    std::vector<std::string> shm_inputs;
    /// POSIX shm name for the output buffer (single plane, same layout as inputs).
    std::string shm_output;
    /// Model-specific options (e.g. `"timestep"`, `"max_depth"`); values are opaque strings parsed by the worker.
    std::unordered_map<std::string, std::string> params;
};

/// Serialize @p req to a binary blob (magic `kInferenceWireMagic`, versioned, big-endian length-prefixed strings).
std::string serialize(const InferenceRequest& req);
/// Parse @p payload into @p out. On failure returns false and sets @p error to a short diagnostic.
bool deserialize(const std::string& payload, InferenceRequest& out, std::string& error);

/// First four bytes of `serialize()` output; used to detect the wire format.
inline constexpr char kInferenceWireMagic[4] = {'N', 'T', 'W', '2'};

}  // namespace nuketorch
