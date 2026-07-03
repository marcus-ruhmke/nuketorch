#pragma once

#include <nuketorch/Errors.h>
#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <sys/types.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
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

/// Direct views into the shared-memory frame buffers (zero-copy path).
/// Pointers stay valid until the next mapFrame()/processFrame() call with a
/// larger frame, or until stop()/abort().
struct MappedFrame {
    std::vector<float*> inputs;
    float* output = nullptr;
};

/// Options sent to the worker together with the frame (header fields + string `params`).
struct InferenceConfig {
    std::string model_path;
    bool use_gpu = true;
    /// Convert the model to half precision on GPU (full FP16 conversion, not autocast).
    bool mixed_precision = true;
    bool debug = false;
    /// Watchdog for one frame in milliseconds. When > 0 and the worker has not
    /// replied in time, the worker is killed and TimeoutError is thrown.
    /// 0 (default) waits as long as the worker process stays alive.
    int frame_timeout_ms = 0;
    /// Model-specific key/value pairs (serialized into `InferenceRequest::params`).
    std::unordered_map<std::string, std::string> params;
};

/// Runs a worker executable in a child process (posix_spawn), exchanges control
/// messages over a Unix socket, and shares frames through anonymous memfd
/// segments passed as file descriptors. Keeps the Nuke/plugin process free of libtorch.
///
/// Thread safety: NOT thread-safe. Nuke may call render entry points from
/// several threads; callers must serialize all access to one InferenceClient
/// (or use one client per thread with distinct socket paths).
///
/// Failure model: methods throw subclasses of nuketorch::Error (see Errors.h).
/// After WorkerDiedError / TimeoutError the worker is gone; call start() to
/// respawn and continue.
class InferenceClient {
public:
    /// @param worker_binary Path to the worker executable (argv[0] of the spawned process).
    /// @param socket_path Unix socket path the server binds before spawning (must be writable,
    ///                    and unique per client instance).
    /// @param num_inputs Number of input `FrameBuffers::inputs` entries (and shm segments) per frame.
    InferenceClient(const std::string& worker_binary, const std::string& socket_path, int num_inputs = 2);
    ~InferenceClient();

    InferenceClient(const InferenceClient&) = delete;
    InferenceClient& operator=(const InferenceClient&) = delete;

    /// Create socket, spawn worker, wait for the versioned `READY` handshake.
    /// Safe to call again after a worker death to respawn.
    void start();
    /// Graceful shutdown: `QUIT`/`BYE`, then reap with SIGTERM -> SIGKILL escalation.
    void stop();
    /// `SIGKILL` the worker and drop the server socket.
    void abort();
    /// Round-trip a `PING`; returns false if the child died, is unresponsive, or times out.
    bool ping();
    /// Query the worker's `GPUINFO` line; returns a human-readable fallback on failure.
    std::string getGpuInfo();

    /// Copy inputs to shared memory (with vertical flip), run one frame, copy the output back.
    /// If @p is_aborted becomes true while waiting, the cancel flag shared with the worker is
    /// raised so a cooperative worker can return early; CancelledError is thrown either way.
    /// When @p metrics is non-null, fills host timings and merges worker metrics from the reply.
    void processFrame(const FrameBuffers& buffers,
                      const InferenceConfig& config,
                      std::function<bool()> is_aborted = nullptr,
                      InferenceMetrics* metrics = nullptr);

    /// Zero-copy alternative: map (or grow) the shared frame buffers for the given
    /// dimensions and return direct pointers. Fill the inputs in place, call
    /// processMappedFrame(), then read the output through the returned pointer.
    /// Layout and scanline order are a contract between caller and worker; no
    /// vertical flip is applied on this path.
    MappedFrame mapFrame(int width, int height, int channels);

    /// Run one frame against the buffers most recently returned by mapFrame().
    void processMappedFrame(const InferenceConfig& config,
                            std::function<bool()> is_aborted = nullptr,
                            InferenceMetrics* metrics = nullptr);

private:
    std::string worker_binary_;
    std::string socket_path_;
    int num_inputs_;
    std::unique_ptr<IPCServer> server_;
    pid_t worker_pid_;
    unsigned long long request_counter_;

    std::vector<std::unique_ptr<SharedMemoryBuffer>> shm_inputs_;
    std::unique_ptr<SharedMemoryBuffer> shm_out_;
    std::unique_ptr<SharedMemoryBuffer> shm_cancel_;
    size_t current_shm_size_ = 0;
    int mapped_width_ = 0;
    int mapped_height_ = 0;
    int mapped_channels_ = 0;

    int stderr_fd_ = -1;
    std::string stderr_tail_;

    void ensureCapacity(int width, int height, int channels);
    void setCancelFlag(bool cancelled);

    void spawnWorker();
    void killWorker();
    bool reapWithTimeout(int timeout_ms, int* status);
    void closeStderrPipe();
    void drainStderr();
    std::string stderrTailSuffix() const;

    enum class WorkerWait { alive, exited, reaped_elsewhere };
    /// Non-blocking death check; `reaped_elsewhere` covers a host SIGCHLD handler
    /// stealing the wait status (waitpid == ECHILD).
    WorkerWait pollWorkerExit(int* status);
    /// Tears down (pid, server, stderr pipe) and throws WorkerDiedError with the
    /// exit description and captured stderr tail.
    [[noreturn]] void throwWorkerDied(const std::string& context, WorkerWait how, int status);

    /// Parses `R|<id>|<body>`; kills the worker and throws ProtocolError on malformed input.
    std::pair<unsigned long long, std::string> parseReply(const std::string& msg);
    /// Waits (bounded) for the reply to @p id, discarding stale replies to older requests.
    std::string expectReply(unsigned long long id, int timeout_ms);

    /// Sends the request, supervises the wait (abort flag, worker death, frame
    /// deadline), and parses the reply into @p metrics. Returns true when the
    /// abort predicate fired during the wait.
    bool runFrame(int width,
                  int height,
                  int channels,
                  const InferenceConfig& config,
                  const std::function<bool()>& is_aborted,
                  InferenceMetrics* metrics);
};

}  // namespace nuketorch
