#include <nuketorch/WorkerHarness.h>

#include <nuketorch/Errors.h>
#include <nuketorch/FreezeDebug.h>
#include <nuketorch/ImageUtils.h>
#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/IPC.h>
#include <nuketorch/Protocol.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <signal.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace nuketorch {
namespace {

/// Owns file descriptors received with a message until they are adopted into
/// SharedMemoryBuffers; closes whatever is left on scope exit.
class FdGuard {
public:
    explicit FdGuard(std::vector<int> fds) : fds_(std::move(fds)) {}
    ~FdGuard() {
        for (int fd : fds_) {
            if (fd >= 0) {
                ::close(fd);
            }
        }
    }

    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;

    size_t count() const { return fds_.size(); }

    /// Transfer ownership of slot @p i to the caller.
    int take(size_t i) {
        const int fd = fds_[i];
        fds_[i] = -1;
        return fd;
    }

private:
    std::vector<int> fds_;
};

/// If @p msg is "<command>|<decimal id>", returns true and fills @p id_str.
bool parseControl(const std::string& msg, const char* command, std::string& id_str) {
    const std::string prefix = std::string(command) + "|";
    if (msg.rfind(prefix, 0) != 0 || msg.size() <= prefix.size()) {
        return false;
    }
    for (size_t i = prefix.size(); i < msg.size(); ++i) {
        if (msg[i] < '0' || msg[i] > '9') {
            return false;
        }
    }
    id_str = msg.substr(prefix.size());
    return true;
}

std::string replyPrefix(const std::string& id_str) {
    return "R|" + id_str + "|";
}

std::string replyPrefix(uint64_t id) {
    return "R|" + std::to_string(id) + "|";
}

}  // namespace

int workerMain(int argc, char** argv, InferenceCallback inference, GpuInfoCallback gpu_info) {
    FREEZE_LOG("WORKER", "workerMain() entry argc=%d argv[1]=%s",
               argc, argc > 1 ? argv[1] : "?");

    if (argc < 2) {
        std::cerr << "Usage: worker <socket_path>\n";
        return 2;
    }
    if (!inference) {
        std::cerr << "workerMain: inference callback is null\n";
        return 2;
    }

    // The host may vanish while we are writing a reply; that must surface as
    // IPCClosedError (EPIPE), not terminate us with SIGPIPE.
    ::signal(SIGPIPE, SIG_IGN);

    const std::string socket_path = argv[1];

    try {
        IPCClient client(socket_path);
        client.send("READY|" + std::to_string(kProtocolVersion));
        FREEZE_LOG("WORKER", "READY|%u sent, entering message loop", kProtocolVersion);

        while (true) {
            std::vector<int> raw_fds;
            const std::string msg = client.receive(-1, &raw_fds);
            FdGuard fds(std::move(raw_fds));

            std::string id_str;
            if (parseControl(msg, "QUIT", id_str)) {
                client.send(replyPrefix(id_str) + "BYE");
                break;
            }
            if (parseControl(msg, "PING", id_str)) {
                client.send(replyPrefix(id_str) + "PONG");
                continue;
            }
            if (parseControl(msg, "GPUINFO", id_str)) {
                client.send(replyPrefix(id_str) +
                            (gpu_info ? gpu_info() : std::string("ERROR|GPU info not available")));
                continue;
            }

            InferenceRequest req;
            std::string parse_error;
            if (!deserialize(msg, req, parse_error)) {
                // Garbage on the wire means the two sides no longer agree on the
                // protocol; the connection is unusable. Exit so the host gets a
                // WorkerDiedError with this explanation in the stderr tail.
                std::cerr << "worker: undecodable request (" << parse_error << "), exiting\n";
                return 1;
            }
            const std::string prefix = replyPrefix(req.request_id);

            const int w = req.header.width;
            const int h = req.header.height;
            const int c = req.header.channels;
            if (w <= 0 || h <= 0 || c <= 0) {
                client.send(prefix + "ERR|bad_request|invalid dimensions");
                continue;
            }
            if (req.num_inputs == 0) {
                client.send(prefix + "ERR|bad_request|no inputs");
                continue;
            }
            // Expected fd layout: [input 0 .. input N-1, output, cancel-flag].
            if (fds.count() != static_cast<size_t>(req.num_inputs) + 2) {
                client.send(prefix + "ERR|bad_request|fd count mismatch (got " +
                            std::to_string(fds.count()) + ", expected " +
                            std::to_string(req.num_inputs + 2) + ")");
                continue;
            }

            size_t bytes = 0;
            if (!computeFrameBytes(w, h, c, bytes)) {
                client.send(prefix + "ERR|bad_request|frame dimensions overflow");
                continue;
            }

            try {
                std::vector<SharedMemoryBuffer> input_bufs;
                input_bufs.reserve(req.num_inputs);
                std::vector<void*> input_ptrs;
                input_ptrs.reserve(req.num_inputs);
                for (uint32_t i = 0; i < req.num_inputs; ++i) {
                    input_bufs.push_back(SharedMemoryBuffer::adopt(fds.take(i), bytes));
                    input_ptrs.push_back(input_bufs.back().data());
                }
                SharedMemoryBuffer out_buf =
                    SharedMemoryBuffer::adopt(fds.take(req.num_inputs), bytes);
                SharedMemoryBuffer cancel_buf =
                    SharedMemoryBuffer::adopt(fds.take(req.num_inputs + 1), sizeof(uint32_t));

                const uint32_t* cancel_ptr = static_cast<const uint32_t*>(cancel_buf.data());
                WorkerContext ctx{req,
                                  std::move(input_ptrs),
                                  out_buf.data(),
                                  bytes,
                                  [cancel_ptr]() {
                                      return __atomic_load_n(cancel_ptr, __ATOMIC_SEQ_CST) != 0;
                                  },
                                  {}};

                try {
                    inference(ctx);
                    client.send(prefix + "OK" + serializeMetrics(ctx.metrics));
                } catch (const CancelledError& e) {
                    client.send(prefix + "ERR|cancelled|" + e.what());
                } catch (const std::invalid_argument& e) {
                    // Bad caller-supplied params (e.g. an unknown backend name)
                    // are the requester's fault, not a worker malfunction.
                    client.send(prefix + "ERR|bad_request|" + e.what());
                } catch (const std::exception& e) {
                    client.send(prefix + "ERR|exception|" + e.what());
                } catch (...) {
                    client.send(prefix + "ERR|unknown|");
                }
            } catch (const BadRequestError& e) {
                client.send(prefix + "ERR|bad_request|" + e.what());
            } catch (const std::exception& e) {
                // Mapping failures (fstat/mmap) must fail the frame, not the worker.
                client.send(prefix + "ERR|exception|" + e.what());
            }
        }
    } catch (const IPCClosedError&) {
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Worker fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

}  // namespace nuketorch
