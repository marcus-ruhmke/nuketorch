#include <nuketorch/InferenceClient.h>

#include <nuketorch/ImageUtils.h>
#include <nuketorch/IPC.h>
#include <nuketorch/Protocol.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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
    if (worker_pid_ > 0) {
        return;
    }

    server_ = std::make_unique<IPCServer>(socket_path_);
    const pid_t pid = fork();
    if (pid < 0) {
        server_.reset();
        throw std::runtime_error("fork failed");
    }

    if (pid == 0) {
        execl(worker_binary_.c_str(), worker_binary_.c_str(), socket_path_.c_str(), nullptr);
        _exit(127);
    }

    worker_pid_ = pid;
    const std::string ready = server_->receive(-1);
    if (ready != "READY") {
        stop();
        throw std::runtime_error("worker did not report READY");
    }
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
                                   std::function<bool()> is_aborted) {
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

    for (int i = 0; i < num_inputs_; ++i) {
        copyPlanarWithVerticalFlip(buffers.inputs[static_cast<size_t>(i)],
                                   static_cast<float*>(shm_inputs_[static_cast<size_t>(i)]->data()),
                                   buffers.width,
                                   buffers.height,
                                   buffers.channels);
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

    server_->send(serialize(req));

    int poll_interval_ms = 100;
    bool was_aborted = false;

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

    const std::string response = server_->receive(-1);
    if (response != "OK") {
        throw std::runtime_error("worker process failed: " + response);
    }

    copyPlanarWithVerticalFlip(static_cast<const float*>(shm_out_->data()),
                               buffers.output,
                               buffers.width,
                               buffers.height,
                               buffers.channels);

    if (was_aborted) {
        throw std::runtime_error("Cancelled by user");
    }
}

}  // namespace nuketorch
