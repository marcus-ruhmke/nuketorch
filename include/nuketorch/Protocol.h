#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace nuketorch {

/// Wire protocol version spoken by this build. The worker announces it in the
/// `READY|<version>` handshake and the client refuses to talk to a mismatch.
inline constexpr uint32_t kProtocolVersion = 2;

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
    /// Convert the model to half precision on GPU (full FP16, not autocast) when supported.
    bool mixed_precision = true;
    /// If true, workers may enable verbose logging.
    bool debug = false;
    /// Path to the model artifact on the worker machine (TorchScript `.pt`, AOTInductor `.pt2`, TensorRT `.engine`).
    std::string model_path;
};

/// One inference job. Frame buffers are anonymous memfd segments whose file
/// descriptors travel as SCM_RIGHTS ancillary data on the same socket message,
/// ordered `[input 0 .. input N-1, output, cancel-flag]`. The payload itself
/// only carries the expected input count so the worker can validate the fd set.
struct InferenceRequest {
    FrameHeader header;
    /// Monotonically increasing per-connection id; the worker echoes it in the `R|<id>|...` reply.
    uint64_t request_id = 0;
    /// Number of input planes; must match the number of input fds sent with the message.
    uint32_t num_inputs = 0;
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
