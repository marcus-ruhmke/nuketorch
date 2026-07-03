#include <nuketorch/InferenceClient.h>

#include <nuketorch/FreezeDebug.h>
#include <nuketorch/ImageUtils.h>
#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/IPC.h>
#include <nuketorch/Protocol.h>
#include <nuketorch/ScopedTimer.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <errno.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <stdexcept>

namespace nuketorch {

InferenceClient::InferenceClient(const std::string& worker_binary,
                                 const std::string& socket_path,
                                 int num_inputs)
    : worker_binary_(worker_binary),
      socket_path_(socket_path),
      num_inputs_(num_inputs),
      worker_pid_(-1),
      request_counter_(0) {
    if (num_inputs_ < 1) {
        throw std::runtime_error("num_inputs must be >= 1");
    }
}

InferenceClient::~InferenceClient() {
    try {
        stop();
    } catch (...) {
    }
}

void InferenceClient::start() {
    FREEZE_LOG("PLUGIN", "InferenceClient::start() entry, worker_binary=%s socket=%s",
               worker_binary_.c_str(), socket_path_.c_str());
    if (worker_pid_ > 0) {
        FREEZE_LOG("PLUGIN", "InferenceClient::start(): worker_pid_=%d already set, returning",
                   static_cast<int>(worker_pid_));
        return;
    }

    FREEZE_LOG("PLUGIN", "InferenceClient::start(): creating IPCServer");
    server_ = std::make_unique<IPCServer>(socket_path_);
    FREEZE_LOG("PLUGIN", "InferenceClient::start(): IPCServer ready, ABOUT TO fork()");
    const pid_t pid = fork();
    if (pid < 0) {
        const int err = errno;
        FREEZE_LOG("PLUGIN", "InferenceClient::start(): fork() FAILED errno=%d (%s)",
                   err, std::strerror(err));
        server_.reset();
        throw std::runtime_error("fork failed");
    }

    if (pid == 0) {
        FREEZE_LOG("CHILD", "post-fork in child, ABOUT TO execl(%s)",
                   worker_binary_.c_str());
        execl(worker_binary_.c_str(), worker_binary_.c_str(), socket_path_.c_str(), nullptr);
        const int err = errno;
        FREEZE_LOG("CHILD", "execl() FAILED errno=%d (%s)", err, std::strerror(err));
        _exit(127);
    }

    worker_pid_ = pid;

    // Wait for READY with a finite timeout AND poll for early child death, so
    // we can never wedge the caller's (typically GUI) thread if the worker
    // exits during dynamic linking / static init (e.g. missing shared lib).
    constexpr int kReadyTimeoutMs = 30000;     // total budget for worker bring-up
    constexpr int kPollIntervalMs = 200;       // how often we waitpid()
    int waited_ms = 0;
    FREEZE_LOG("PLUGIN", "InferenceClient::start(): post-fork parent, child pid=%d, "
               "polling for READY (timeout %d ms)",
               static_cast<int>(pid), kReadyTimeoutMs);

    std::string ready;
    while (true) {
        // 1. Has the child already died? execve() failure or missing shared
        //    library show up here before any IPC ever happens.
        int status = 0;
        const pid_t r = ::waitpid(worker_pid_, &status, WNOHANG);
        if (r == worker_pid_) {
            const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            const int term_sig = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
            FREEZE_LOG("PLUGIN",
                       "InferenceClient::start(): worker died early "
                       "exit_code=%d term_sig=%d (likely missing shared lib; "
                       "run worker standalone to see ld-linux error)",
                       exit_code, term_sig);
            worker_pid_ = -1;
            server_.reset();
            throw std::runtime_error(
                "worker process exited before sending READY (exit code " +
                std::to_string(exit_code) +
                "; run the worker binary directly to see the error)");
        }

        // 2. Is the READY message available?
        if (server_->hasData(kPollIntervalMs)) {
            ready = server_->receive(1000);
            FREEZE_LOG("PLUGIN",
                       "InferenceClient::start(): receive() returned '%s'",
                       ready.c_str());
            break;
        }

        waited_ms += kPollIntervalMs;
        if (waited_ms >= kReadyTimeoutMs) {
            FREEZE_LOG("PLUGIN",
                       "InferenceClient::start(): timed out after %d ms waiting for READY",
                       waited_ms);
            // Best-effort: kill the child and clean up.
            ::kill(worker_pid_, SIGKILL);
            int dummy = 0;
            (void)::waitpid(worker_pid_, &dummy, 0);
            worker_pid_ = -1;
            server_.reset();
            throw std::runtime_error(
                "worker did not report READY within " +
                std::to_string(kReadyTimeoutMs) + " ms");
        }
    }

    if (ready != "READY") {
        FREEZE_LOG("PLUGIN", "InferenceClient::start(): handshake mismatch, calling stop()");
        stop();
        throw std::runtime_error("worker did not report READY");
    }
    FREEZE_LOG("PLUGIN", "InferenceClient::start(): handshake OK, returning");
}

void InferenceClient::stop() {
    if (worker_pid_ <= 0) {
        return;
    }

    bool graceful = false;
    try {
        server_->send("QUIT");
        if (server_->hasData(3000)) {
            const std::string bye = server_->receive(-1);
            graceful = (bye == "BYE");
        }
    } catch (...) {
    }

    int status = 0;
    if (graceful) {
        (void)waitpid(worker_pid_, &status, 0);
    } else {
        kill(worker_pid_, SIGTERM);
        (void)waitpid(worker_pid_, &status, 0);
    }

    worker_pid_ = -1;
    server_.reset();
}

void InferenceClient::abort() {
    if (worker_pid_ > 0) {
        kill(worker_pid_, SIGKILL);
        int status = 0;
        (void)waitpid(worker_pid_, &status, 0);
        worker_pid_ = -1;
    }
    server_.reset();
}

bool InferenceClient::ping() {
    if (!server_ || worker_pid_ <= 0) {
        return false;
    }
    int status = 0;
    if (waitpid(worker_pid_, &status, WNOHANG) > 0) {
        worker_pid_ = -1;
        return false;
    }
    try {
        server_->send("PING");
        if (server_->hasData(1000)) {
            const std::string reply = server_->receive(5000);
            return reply == "PONG";
        }
        return false;
    } catch (...) {
        return false;
    }
}

std::string InferenceClient::getGpuInfo() {
    if (!server_ || worker_pid_ <= 0) {
        return "Unknown (Worker not started)";
    }

    try {
        server_->send("GPUINFO");
        if (server_->hasData(1000)) {
            const std::string reply = server_->receive(5000);
            if (reply.rfind("OK|", 0) == 0) {
                return reply.substr(3);
            }
        }
    } catch (...) {
    }
    return "Unknown (No CUDA device)";
}

std::string InferenceClient::makeSharedMemoryName(const char* role) {
    ++request_counter_;
    return "/nuketorch_" + std::string(role) + "_" + std::to_string(getpid()) + "_" +
           std::to_string(request_counter_);
}

void InferenceClient::processFrame(const FrameBuffers& buffers,
                                   const InferenceConfig& config,
                                   std::function<bool()> is_aborted,
                                   InferenceMetrics* metrics) {
    const auto t_total_start = std::chrono::steady_clock::now();

    if (!server_ || worker_pid_ <= 0) {
        throw std::runtime_error("worker not started");
    }
    if (static_cast<int>(buffers.inputs.size()) != num_inputs_) {
        throw std::runtime_error("invalid frame buffers: input count mismatch");
    }
    for (const float* p : buffers.inputs) {
        if (!p) {
            throw std::runtime_error("invalid frame buffers");
        }
    }
    if (!buffers.output) {
        throw std::runtime_error("invalid frame buffers");
    }
    if (buffers.width <= 0 || buffers.height <= 0 || buffers.channels <= 0) {
        throw std::runtime_error("invalid frame dimensions");
    }

    const size_t pixel_count = static_cast<size_t>(buffers.width) * static_cast<size_t>(buffers.height) *
                               static_cast<size_t>(buffers.channels);
    const size_t bytes = pixel_count * sizeof(float);

    if (current_shm_size_ < bytes || shm_inputs_.size() != static_cast<size_t>(num_inputs_) || !shm_out_) {
        shm_inputs_.clear();
        shm_out_.reset();
        for (int i = 0; i < num_inputs_; ++i) {
            const std::string role = "in" + std::to_string(i);
            const std::string name = makeSharedMemoryName(role.c_str());
            shm_inputs_.push_back(std::make_unique<SharedMemoryBuffer>(name, bytes, true));
        }
        shm_out_ = std::make_unique<SharedMemoryBuffer>(makeSharedMemoryName("out"), bytes, true);
        current_shm_size_ = bytes;
    }

    double shm_write_ms = 0;
    {
        nuketorch::ScopedTimer t(shm_write_ms);
        for (int i = 0; i < num_inputs_; ++i) {
            copyPlanarWithVerticalFlip(buffers.inputs[static_cast<size_t>(i)],
                                       static_cast<float*>(shm_inputs_[static_cast<size_t>(i)]->data()),
                                       buffers.width,
                                       buffers.height,
                                       buffers.channels);
        }
    }

    InferenceRequest req;
    req.header.model_path = config.model_path;
    req.header.width = buffers.width;
    req.header.height = buffers.height;
    req.header.channels = buffers.channels;
    req.header.use_gpu = config.use_gpu;
    req.header.mixed_precision = config.mixed_precision;
    req.header.debug = config.debug;
    req.params = config.params;
    req.shm_output = shm_out_->name();
    req.shm_inputs.reserve(static_cast<size_t>(num_inputs_));
    for (const auto& in : shm_inputs_) {
        req.shm_inputs.push_back(in->name());
    }

    double round_trip_ms = 0;
    std::string response;
    bool was_aborted = false;
    {
        nuketorch::ScopedTimer t(round_trip_ms);
        server_->send(serialize(req));

        int poll_interval_ms = 100;

        while (true) {
            if (!was_aborted && is_aborted && is_aborted()) {
                was_aborted = true;
            }

            if (server_->hasData(poll_interval_ms)) {
                break;
            }

            int st = 0;
            if (waitpid(worker_pid_, &st, WNOHANG) > 0) {
                worker_pid_ = -1;
                throw std::runtime_error("Worker process died unexpectedly");
            }
        }

        response = server_->receive(-1);
    }

    InferenceMetrics parsed_worker;
    std::string parse_error;
    if (!parseInferenceOkResponse(response, parsed_worker, parse_error)) {
        throw std::runtime_error("worker process failed: " + response);
    }

    double shm_read_ms = 0;
    {
        nuketorch::ScopedTimer t(shm_read_ms);
        copyPlanarWithVerticalFlip(static_cast<const float*>(shm_out_->data()),
                                   buffers.output,
                                   buffers.width,
                                   buffers.height,
                                   buffers.channels);
    }

    const auto t_total_end = std::chrono::steady_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

    if (metrics) {
        *metrics = parsed_worker;
        metrics->shm_write_ms = shm_write_ms;
        metrics->round_trip_ms = round_trip_ms;
        metrics->shm_read_ms = shm_read_ms;
        metrics->total_ms = total_ms;
    }

    if (was_aborted) {
        throw std::runtime_error("Cancelled by user");
    }
}

}  // namespace nuketorch
