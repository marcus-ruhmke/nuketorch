#pragma once

#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nuketorch {

class IPCServer;

/// Host-side planar float buffers: N inputs + one output, all same width/height/channels (channel-major, Nuke scanline order).
struct FrameBuffers {
    /// Pointers to each input plane; size must match `InferenceClient` construction `num_inputs`.
    std::vector<const float*> inputs;
    /// Writable output plane (same dimensions as inputs).
    float* output = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
};

/// Options sent to the worker together with shared-memory names (header fields + string `params`).
struct InferenceConfig {
    std::string model_path;
    bool use_gpu = true;
    bool mixed_precision = true;
    bool debug = false;
    /// Model-specific key/value pairs (serialized into `InferenceRequest::params`).
    std::unordered_map<std::string, std::string> params;
};

/// Runs a worker executable in a child process, exchanges control messages over a Unix socket, and copies frames through POSIX shm.
/// Keeps the Nuke/plugin process free of libtorch.
class InferenceClient {
public:
    /// @param worker_binary Path to the worker executable (passed as argv[0] to `execl`).
    /// @param socket_path Unix socket path the server binds before forking (must be writable).
    /// @param num_inputs Number of input `FrameBuffers::inputs` entries (and shm segments) per frame.
    InferenceClient(const std::string& worker_binary, const std::string& socket_path, int num_inputs = 2);
    ~InferenceClient();

    InferenceClient(const InferenceClient&) = delete;
    InferenceClient& operator=(const InferenceClient&) = delete;

    /// Create socket, fork worker, wait for `"READY"`.
    void start();
    /// Send `QUIT`, wait for `"BYE"` or terminate the child.
    void stop();
    /// `SIGKILL` the worker and drop the server socket.
    void abort();
    /// Send `PING`, expect `PONG` (returns false if the child died or times out).
    bool ping();
    /// Send `GPUINFO`; on success returns text after the `OK|` prefix (empty or default string on failure).
    std::string getGpuInfo();

    /// Copy inputs to shm (with vertical flip), send serialized `InferenceRequest`, wait for `OK` + metrics blob, copy output back.
    /// If @p is_aborted becomes true while waiting, still completes the round-trip then throws `std::runtime_error("Cancelled by user")`.
    /// When @p metrics is non-null, fills host timings and merges worker metrics from the reply.
    void processFrame(const FrameBuffers& buffers,
                      const InferenceConfig& config,
                      std::function<bool()> is_aborted = nullptr,
                      InferenceMetrics* metrics = nullptr);

private:
    std::string worker_binary_;
    std::string socket_path_;
    int num_inputs_;
    std::unique_ptr<IPCServer> server_;
    int worker_pid_;
    unsigned long long request_counter_;

    std::vector<std::unique_ptr<SharedMemoryBuffer>> shm_inputs_;
    std::unique_ptr<SharedMemoryBuffer> shm_out_;
    size_t current_shm_size_ = 0;

    std::string makeSharedMemoryName(const char* role);
};

}  // namespace nuketorch
