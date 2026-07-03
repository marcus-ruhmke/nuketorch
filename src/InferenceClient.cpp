#include <nuketorch/InferenceClient.h>

#include <nuketorch/FreezeDebug.h>
#include <nuketorch/ImageUtils.h>
#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/IPC.h>
#include <nuketorch/Protocol.h>
#include <nuketorch/ScopedTimer.h>

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

extern char** environ;

namespace nuketorch {
namespace {

using Clock = std::chrono::steady_clock;

/// Total budget for worker bring-up (dynamic linking + libtorch static init can be slow).
constexpr int kReadyTimeoutMs = 30000;
/// How often the supervision loops poll for death / abort / stderr.
constexpr int kPollIntervalMs = 100;
/// Budget for control round-trips (PING, GPUINFO) and for finishing a reply
/// whose first bytes have already arrived.
constexpr int kControlTimeoutMs = 5000;
/// Keep the last N bytes of worker stderr for error messages.
constexpr size_t kStderrTailBytes = 8192;

std::string describeExit(int status) {
    if (WIFEXITED(status)) {
        return "exit code " + std::to_string(WEXITSTATUS(status));
    }
    if (WIFSIGNALED(status)) {
        const int sig = WTERMSIG(status);
        const char* name = strsignal(sig);
        return "signal " + std::to_string(sig) + (name ? std::string(" (") + name + ")" : "");
    }
    return "unknown wait status";
}

}  // namespace

InferenceClient::InferenceClient(const std::string& worker_binary,
                                 const std::string& socket_path,
                                 int num_inputs)
    : worker_binary_(worker_binary),
      socket_path_(socket_path),
      num_inputs_(num_inputs),
      worker_pid_(-1),
      request_counter_(0) {
    if (num_inputs_ < 1) {
        throw Error(ErrorCode::invalid_argument, "num_inputs must be >= 1");
    }
    // Each request carries [inputs..., output, cancel] as SCM_RIGHTS descriptors.
    if (static_cast<size_t>(num_inputs_) + 2 > kMaxFdsPerMessage) {
        throw Error(ErrorCode::invalid_argument,
                    "num_inputs must be <= " + std::to_string(kMaxFdsPerMessage - 2) +
                        " (SCM_RIGHTS fd limit per message)");
    }
}

InferenceClient::~InferenceClient() {
    try {
        stop();
    } catch (...) {
    }
}

void InferenceClient::spawnWorker() {
    int pipefd[2] = {-1, -1};
    if (pipe2(pipefd, O_CLOEXEC) != 0) {
        server_.reset();
        throw Error(ErrorCode::internal, std::string("pipe2 failed: ") + std::strerror(errno));
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDERR_FILENO);

    char* argv[] = {
        const_cast<char*>(worker_binary_.c_str()),
        const_cast<char*>(socket_path_.c_str()),
        nullptr,
    };

    pid_t pid = -1;
    const int rc = posix_spawn(&pid, worker_binary_.c_str(), &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    ::close(pipefd[1]);
    if (rc != 0) {
        ::close(pipefd[0]);
        server_.reset();
        throw SpawnError("failed to spawn worker \"" + worker_binary_ + "\": " + std::strerror(rc));
    }

    worker_pid_ = pid;
    stderr_fd_ = pipefd[0];
    (void)fcntl(stderr_fd_, F_SETFL, O_NONBLOCK);
    FREEZE_LOG("PLUGIN", "spawned worker pid=%d binary=%s", static_cast<int>(pid),
               worker_binary_.c_str());
}

void InferenceClient::start() {
    FREEZE_LOG("PLUGIN", "InferenceClient::start() worker=%s socket=%s",
               worker_binary_.c_str(), socket_path_.c_str());
    if (worker_pid_ > 0) {
        return;
    }

    // Destroy any server left over from a dead worker *before* binding anew:
    // IPCServer's destructor unlinks its socket path, which would otherwise
    // delete the freshly bound socket and break every restart.
    server_.reset();
    closeStderrPipe();
    stderr_tail_.clear();

    server_ = std::make_unique<IPCServer>(socket_path_);
    spawnWorker();

    int waited_ms = 0;
    std::string ready;
    while (true) {
        drainStderr();

        // Death first: exec failures and missing shared libraries show up here
        // before any IPC ever happens.
        int status = 0;
        const WorkerWait wait = pollWorkerExit(&status);
        if (wait != WorkerWait::alive) {
            FREEZE_LOG("PLUGIN", "start(): worker died before READY");
            throwWorkerDied("worker exited before sending READY", wait, status);
        }

        if (server_->hasData(kPollIntervalMs)) {
            try {
                server_->acceptClient(1000, worker_pid_);
                ready = server_->receive(kControlTimeoutMs);
            } catch (const Error&) {
                killWorker();
                throw;
            }
            break;
        }

        waited_ms += kPollIntervalMs;
        if (waited_ms >= kReadyTimeoutMs) {
            drainStderr();
            killWorker();
            throw TimeoutError("worker did not report READY within " +
                               std::to_string(kReadyTimeoutMs) + " ms" + stderrTailSuffix());
        }
    }

    const std::string expected = "READY|" + std::to_string(kProtocolVersion);
    if (ready != expected) {
        killWorker();
        throw ProtocolError("worker handshake mismatch: expected \"" + expected + "\", got \"" +
                            ready.substr(0, 64) +
                            "\" (worker was built against a different nuketorch protocol; rebuild it)");
    }

    if (!shm_cancel_) {
        shm_cancel_ = std::make_unique<SharedMemoryBuffer>(
            SharedMemoryBuffer::create(sizeof(uint32_t)));
    }
    setCancelFlag(false);
    FREEZE_LOG("PLUGIN", "start(): handshake OK, worker pid=%d", static_cast<int>(worker_pid_));
}

bool InferenceClient::reapWithTimeout(int timeout_ms, int* status) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        const pid_t r = ::waitpid(worker_pid_, status, WNOHANG);
        if (r == worker_pid_) {
            return true;
        }
        if (r < 0) {
            // ECHILD: something else (a host SIGCHLD handler) reaped it; it is gone.
            return true;
        }
        if (Clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void InferenceClient::killWorker() {
    if (worker_pid_ > 0) {
        ::kill(worker_pid_, SIGKILL);
        // Bounded reap only: a worker wedged in an uninterruptible driver call
        // (D state) cannot be reaped until the kernel releases it, and blocking
        // here would hang the caller's (GUI) thread — the exact failure the
        // watchdog exists to escape. An unreaped zombie is reclaimed later.
        int status = 0;
        (void)reapWithTimeout(2000, &status);
        worker_pid_ = -1;
    }
    server_.reset();
    drainStderr();
    closeStderrPipe();
}

InferenceClient::WorkerWait InferenceClient::pollWorkerExit(int* status) {
    const pid_t r = ::waitpid(worker_pid_, status, WNOHANG);
    if (r == worker_pid_) {
        return WorkerWait::exited;
    }
    if (r < 0 && errno == ECHILD) {
        return WorkerWait::reaped_elsewhere;
    }
    return WorkerWait::alive;
}

void InferenceClient::throwWorkerDied(const std::string& context, WorkerWait how, int status) {
    const std::string exit_desc = how == WorkerWait::exited
                                      ? describeExit(status)
                                      : "reaped by the host process's SIGCHLD handler";
    worker_pid_ = -1;
    server_.reset();
    drainStderr();
    const std::string message = context + " (" + exit_desc + ")" + stderrTailSuffix();
    closeStderrPipe();
    throw WorkerDiedError(message);
}

void InferenceClient::stop() {
    if (worker_pid_ <= 0) {
        server_.reset();
        closeStderrPipe();
        return;
    }

    bool graceful = false;
    if (server_) {
        try {
            const unsigned long long id = ++request_counter_;
            server_->send("QUIT|" + std::to_string(id));
            graceful = (expectReply(id, 3000) == "BYE");
        } catch (...) {
        }
    }

    // expectReply kills the worker itself on protocol garbage.
    if (worker_pid_ > 0) {
        int status = 0;
        bool reaped = graceful && reapWithTimeout(5000, &status);
        if (!reaped) {
            ::kill(worker_pid_, SIGTERM);
            reaped = reapWithTimeout(3000, &status);
        }
        if (!reaped) {
            ::kill(worker_pid_, SIGKILL);
            (void)reapWithTimeout(2000, &status);
        }
        worker_pid_ = -1;
    }
    server_.reset();
    drainStderr();
    closeStderrPipe();
}

void InferenceClient::abort() {
    killWorker();
}

bool InferenceClient::ping() {
    if (!server_ || worker_pid_ <= 0) {
        return false;
    }
    int status = 0;
    if (pollWorkerExit(&status) != WorkerWait::alive) {
        worker_pid_ = -1;
        return false;
    }
    try {
        const unsigned long long id = ++request_counter_;
        server_->send("PING|" + std::to_string(id));
        return expectReply(id, 2000) == "PONG";
    } catch (const Error&) {
        // Timeout: a late PONG is discarded by id on the next exchange.
        // Protocol garbage: expectReply already killed the worker.
        return false;
    }
}

std::string InferenceClient::getGpuInfo() {
    if (!server_ || worker_pid_ <= 0) {
        return "Unknown (Worker not started)";
    }
    try {
        const unsigned long long id = ++request_counter_;
        server_->send("GPUINFO|" + std::to_string(id));
        const std::string body = expectReply(id, kControlTimeoutMs);
        if (body.rfind("OK|", 0) == 0) {
            return body.substr(3);
        }
    } catch (const Error&) {
    }
    return "Unknown (No CUDA device)";
}

std::pair<unsigned long long, std::string> InferenceClient::parseReply(const std::string& msg) {
    // Replies are framed as R|<decimal id>|<body>. Pure parser: lifecycle
    // decisions (killing the worker on garbage) belong to the callers.
    bool ok = msg.size() >= 4 && msg[0] == 'R' && msg[1] == '|';
    size_t pos = 2;
    unsigned long long id = 0;
    bool any_digit = false;
    while (ok && pos < msg.size() && msg[pos] >= '0' && msg[pos] <= '9') {
        id = id * 10 + static_cast<unsigned long long>(msg[pos] - '0');
        ++pos;
        any_digit = true;
    }
    ok = ok && any_digit && pos < msg.size() && msg[pos] == '|';
    if (!ok) {
        throw ProtocolError("malformed reply from worker: \"" + msg.substr(0, 64) + "\"");
    }
    return {id, msg.substr(pos + 1)};
}

std::string InferenceClient::expectReply(unsigned long long id, int timeout_ms) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                              deadline - Clock::now()).count();
        const int remaining = left > 0 ? static_cast<int>(left) : 0;
        const std::string msg = server_->receive(remaining);
        std::pair<unsigned long long, std::string> parsed;
        try {
            parsed = parseReply(msg);
        } catch (const ProtocolError&) {
            killWorker();  // garbage framing: the connection is unusable
            throw;
        }
        if (parsed.first == id) {
            return std::move(parsed.second);
        }
        if (parsed.first < id) {
            continue;  // stale reply to a request that timed out earlier; drop it
        }
        killWorker();
        throw ProtocolError("reply id " + std::to_string(parsed.first) +
                            " is ahead of request id " + std::to_string(id));
    }
}

void InferenceClient::ensureCapacity(int width, int height, int channels) {
    size_t bytes = 0;
    if (!computeFrameBytes(width, height, channels, bytes)) {
        throw Error(ErrorCode::invalid_argument,
                    "frame dimensions overflow the addressable buffer size");
    }
    if (current_shm_size_ < bytes || shm_inputs_.size() != static_cast<size_t>(num_inputs_) ||
        !shm_out_) {
        shm_inputs_.clear();
        shm_out_.reset();
        for (int i = 0; i < num_inputs_; ++i) {
            shm_inputs_.push_back(
                std::make_unique<SharedMemoryBuffer>(SharedMemoryBuffer::create(bytes)));
        }
        shm_out_ = std::make_unique<SharedMemoryBuffer>(SharedMemoryBuffer::create(bytes));
        current_shm_size_ = bytes;
    }
}

void InferenceClient::setCancelFlag(bool cancelled) {
    if (!shm_cancel_) {
        return;
    }
    __atomic_store_n(static_cast<uint32_t*>(shm_cancel_->data()), cancelled ? 1u : 0u,
                     __ATOMIC_SEQ_CST);
}

MappedFrame InferenceClient::mapFrame(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw Error(ErrorCode::invalid_argument, "invalid frame dimensions");
    }
    ensureCapacity(width, height, channels);
    mapped_width_ = width;
    mapped_height_ = height;
    mapped_channels_ = channels;
    MappedFrame frame;
    frame.inputs.reserve(shm_inputs_.size());
    for (const auto& buf : shm_inputs_) {
        frame.inputs.push_back(static_cast<float*>(buf->data()));
    }
    frame.output = static_cast<float*>(shm_out_->data());
    return frame;
}

bool InferenceClient::runFrame(int width,
                               int height,
                               int channels,
                               const InferenceConfig& config,
                               const std::function<bool()>& is_aborted,
                               InferenceMetrics* metrics) {
    if (!server_ || worker_pid_ <= 0) {
        throw Error(ErrorCode::not_started, "worker not started");
    }

    const unsigned long long id = ++request_counter_;
    InferenceRequest req;
    req.request_id = id;
    req.num_inputs = static_cast<uint32_t>(num_inputs_);
    req.header.model_path = config.model_path;
    req.header.width = width;
    req.header.height = height;
    req.header.channels = channels;
    req.header.use_gpu = config.use_gpu;
    req.header.mixed_precision = config.mixed_precision;
    req.header.debug = config.debug;
    req.params = config.params;

    setCancelFlag(false);

    std::vector<int> fds;
    fds.reserve(shm_inputs_.size() + 2);
    for (const auto& buf : shm_inputs_) {
        fds.push_back(buf->fd());
    }
    fds.push_back(shm_out_->fd());
    fds.push_back(shm_cancel_->fd());

    double round_trip_ms = 0;
    std::string body;
    bool was_aborted = false;
    {
        ScopedTimer t(round_trip_ms);
        const bool has_deadline = config.frame_timeout_ms > 0;
        const auto frame_deadline =
            Clock::now() + std::chrono::milliseconds(config.frame_timeout_ms);

        try {
            server_->sendWithFds(serialize(req), fds);

            while (true) {
                drainStderr();
                if (!was_aborted && is_aborted && is_aborted()) {
                    was_aborted = true;
                    setCancelFlag(true);
                }

                if (server_->hasData(kPollIntervalMs)) {
                    // Once bytes started arriving, bound the rest of the message
                    // by the frame deadline too — a worker that stalls mid-reply
                    // must not extend the watchdog by kControlTimeoutMs.
                    int recv_budget = kControlTimeoutMs;
                    if (has_deadline) {
                        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                                              frame_deadline - Clock::now()).count();
                        recv_budget = static_cast<int>(
                            std::min<long long>(kControlTimeoutMs, left > 1 ? left : 1));
                    }
                    std::pair<unsigned long long, std::string> parsed;
                    try {
                        parsed = parseReply(server_->receive(recv_budget));
                    } catch (const TimeoutError&) {
                        // Mid-message timeout leaves the stream desynchronized.
                        killWorker();
                        throw TimeoutError("worker stalled mid-reply; worker killed" +
                                           stderrTailSuffix());
                    } catch (const ProtocolError&) {
                        killWorker();
                        throw;
                    }
                    if (parsed.first == id) {
                        body = std::move(parsed.second);
                        break;
                    }
                    if (parsed.first < id) {
                        continue;  // stale reply from an earlier, timed-out exchange
                    }
                    killWorker();
                    throw ProtocolError("reply id " + std::to_string(parsed.first) +
                                        " is ahead of request id " + std::to_string(id));
                }

                int status = 0;
                const WorkerWait wait = pollWorkerExit(&status);
                if (wait != WorkerWait::alive) {
                    throwWorkerDied("worker died during frame", wait, status);
                }

                if (has_deadline && Clock::now() >= frame_deadline) {
                    drainStderr();
                    killWorker();
                    throw TimeoutError("frame timed out after " +
                                       std::to_string(config.frame_timeout_ms) +
                                       " ms; worker killed" + stderrTailSuffix());
                }
            }
        } catch (const IPCClosedError&) {
            killWorker();  // bounded reap whether it exited or merely dropped the socket
            throw WorkerDiedError("worker closed the connection mid-frame" + stderrTailSuffix());
        }
    }

    if (body.size() >= 2 && body[0] == 'O' && body[1] == 'K') {
        InferenceMetrics worker_metrics;
        std::string err;
        if (!parseInferenceOkResponse(body, worker_metrics, err)) {
            killWorker();
            throw ProtocolError("bad metrics payload from worker: " + err);
        }
        if (metrics) {
            *metrics = worker_metrics;
            metrics->round_trip_ms = round_trip_ms;
        }
        return was_aborted;
    }

    if (body.rfind("ERR|", 0) == 0) {
        const std::string rest = body.substr(4);
        const size_t bar = rest.find('|');
        const std::string code = bar == std::string::npos ? rest : rest.substr(0, bar);
        const std::string message = bar == std::string::npos ? "" : rest.substr(bar + 1);
        if (code == "cancelled") {
            throw CancelledError(message.empty() ? "Cancelled by user" : message);
        }
        if (code == "bad_request") {
            throw BadRequestError("worker rejected request: " + message);
        }
        throw WorkerReportedError("worker error: " + message);
    }

    killWorker();
    throw ProtocolError("unexpected inference reply from worker");
}

void InferenceClient::processFrame(const FrameBuffers& buffers,
                                   const InferenceConfig& config,
                                   std::function<bool()> is_aborted,
                                   InferenceMetrics* metrics) {
    const auto t_total_start = Clock::now();

    // Fail fast before allocating segments or copying planes.
    if (!server_ || worker_pid_ <= 0) {
        throw Error(ErrorCode::not_started, "worker not started");
    }
    if (static_cast<int>(buffers.inputs.size()) != num_inputs_) {
        throw Error(ErrorCode::invalid_argument, "invalid frame buffers: input count mismatch");
    }
    for (const float* p : buffers.inputs) {
        if (!p) {
            throw Error(ErrorCode::invalid_argument, "invalid frame buffers: null input");
        }
    }
    if (!buffers.output) {
        throw Error(ErrorCode::invalid_argument, "invalid frame buffers: null output");
    }
    if (buffers.width <= 0 || buffers.height <= 0 || buffers.channels <= 0) {
        throw Error(ErrorCode::invalid_argument, "invalid frame dimensions");
    }

    ensureCapacity(buffers.width, buffers.height, buffers.channels);
    // The copy path may have reallocated the segments; pointers handed out by
    // an earlier mapFrame() are no longer trustworthy.
    mapped_width_ = 0;
    mapped_height_ = 0;
    mapped_channels_ = 0;

    double shm_write_ms = 0;
    {
        ScopedTimer t(shm_write_ms);
        for (int i = 0; i < num_inputs_; ++i) {
            copyPlanarWithVerticalFlip(buffers.inputs[static_cast<size_t>(i)],
                                       static_cast<float*>(shm_inputs_[static_cast<size_t>(i)]->data()),
                                       buffers.width,
                                       buffers.height,
                                       buffers.channels);
        }
    }

    const bool was_aborted =
        runFrame(buffers.width, buffers.height, buffers.channels, config, is_aborted, metrics);

    double shm_read_ms = 0;
    {
        ScopedTimer t(shm_read_ms);
        copyPlanarWithVerticalFlip(static_cast<const float*>(shm_out_->data()),
                                   buffers.output,
                                   buffers.width,
                                   buffers.height,
                                   buffers.channels);
    }

    if (metrics) {
        metrics->shm_write_ms = shm_write_ms;
        metrics->shm_read_ms = shm_read_ms;
        metrics->total_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t_total_start).count();
    }

    if (was_aborted) {
        throw CancelledError();
    }
}

void InferenceClient::processMappedFrame(const InferenceConfig& config,
                                         std::function<bool()> is_aborted,
                                         InferenceMetrics* metrics) {
    if (mapped_width_ <= 0 || mapped_height_ <= 0 || mapped_channels_ <= 0) {
        throw Error(ErrorCode::invalid_argument, "no mapped frame; call mapFrame() first");
    }
    const auto t_total_start = Clock::now();

    const bool was_aborted =
        runFrame(mapped_width_, mapped_height_, mapped_channels_, config, is_aborted, metrics);

    if (metrics) {
        metrics->total_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t_total_start).count();
    }

    if (was_aborted) {
        throw CancelledError();
    }
}

void InferenceClient::closeStderrPipe() {
    if (stderr_fd_ >= 0) {
        ::close(stderr_fd_);
        stderr_fd_ = -1;
    }
}

void InferenceClient::drainStderr() {
    if (stderr_fd_ < 0) {
        return;
    }
    char buf[1024];
    while (true) {
        const ssize_t n = ::read(stderr_fd_, buf, sizeof(buf));
        if (n <= 0) {
            break;  // EOF, EAGAIN, or EINTR: the next drain catches up
        }
        stderr_tail_.append(buf, static_cast<size_t>(n));
        if (stderr_tail_.size() > kStderrTailBytes) {
            stderr_tail_.erase(0, stderr_tail_.size() - kStderrTailBytes);
        }
    }
}

std::string InferenceClient::stderrTailSuffix() const {
    if (stderr_tail_.empty()) {
        return "";
    }
    return "\n--- worker stderr (tail) ---\n" + stderr_tail_;
}

}  // namespace nuketorch
