#include <nuketorch/WorkerHarness.h>

#include <nuketorch/Errors.h>
#include <nuketorch/FreezeDebug.h>
#include <nuketorch/ImageUtils.h>
#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/IPC.h>
#include <nuketorch/Protocol.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>
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

/// Caches adopted memfd mappings across requests. The client reuses its
/// segments frame-to-frame (reallocating only on growth), so re-mmapping a
/// ~100 MB frame every request is pure overhead; a cache hit costs one fstat.
/// Keyed by (st_dev, st_ino, st_ctim): memfd inode numbers come from a 32-bit
/// counter and can in principle be recycled, so the creation timestamp is
/// included to make a stale hit require an exact ns-level ctime collision too.
class MappingCache {
public:
    /// Takes ownership of @p fd. Returns a mapping covering at least @p bytes.
    /// The pointer stays valid until endRequest() decides to evict the entry
    /// (never within the request that used it).
    void* acquire(int fd, size_t bytes) {
        struct stat st{};
        if (fstat(fd, &st) == -1) {
            const int err = errno;
            ::close(fd);
            throw Error(ErrorCode::internal,
                        std::string("fstat failed on frame fd: ") + std::strerror(err));
        }
        if (st.st_size < 0 || static_cast<size_t>(st.st_size) < bytes) {
            ::close(fd);
            throw BadRequestError("shared memory segment smaller than requested mapping");
        }

        const Key key{st.st_dev, st.st_ino, st.st_ctim.tv_sec, st.st_ctim.tv_nsec};
        auto it = map_.find(key);
        if (it != map_.end() && it->second.buf.size() >= bytes) {
            ::close(fd);
            it->second.last_used = tick_;
            return it->second.buf.data();
        }
        if (it != map_.end()) {
            map_.erase(it);  // same segment, previously mapped smaller
        }

        // Map the whole segment so a later, larger frame in the same segment
        // still hits the cache.
        SharedMemoryBuffer buf =
            SharedMemoryBuffer::adopt(fd, static_cast<size_t>(st.st_size));
        void* ptr = buf.data();
        map_.emplace(key, Entry{std::move(buf), tick_});
        return ptr;
    }

    /// Advances the clock and evicts least-recently-used entries beyond the cap.
    /// Entries used by the current request are never evicted here (cap exceeds
    /// any single request's fd count).
    void endRequest() {
        ++tick_;
        while (map_.size() > kMaxEntries) {
            auto oldest = map_.begin();
            for (auto it = map_.begin(); it != map_.end(); ++it) {
                if (it->second.last_used < oldest->second.last_used) {
                    oldest = it;
                }
            }
            map_.erase(oldest);
        }
    }

private:
    static constexpr size_t kMaxEntries = 2 * kMaxFdsPerMessage;

    struct Key {
        dev_t dev;
        ino_t ino;
        time_t ctime_sec;
        long ctime_nsec;
        bool operator==(const Key& o) const {
            return dev == o.dev && ino == o.ino && ctime_sec == o.ctime_sec &&
                   ctime_nsec == o.ctime_nsec;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            size_t h = std::hash<unsigned long long>()(static_cast<unsigned long long>(k.ino));
            h ^= std::hash<unsigned long long>()(static_cast<unsigned long long>(k.dev)) * 31u;
            h ^= std::hash<long long>()(static_cast<long long>(k.ctime_sec)) * 131u;
            h ^= std::hash<long long>()(static_cast<long long>(k.ctime_nsec)) * 1313u;
            return h;
        }
    };
    struct Entry {
        SharedMemoryBuffer buf;
        uint64_t last_used;
    };

    std::unordered_map<Key, Entry, KeyHash> map_;
    uint64_t tick_ = 0;
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

        MappingCache mappings;

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
                std::vector<void*> input_ptrs;
                input_ptrs.reserve(req.num_inputs);
                for (uint32_t i = 0; i < req.num_inputs; ++i) {
                    input_ptrs.push_back(mappings.acquire(fds.take(i), bytes));
                }
                void* output_ptr = mappings.acquire(fds.take(req.num_inputs), bytes);
                const uint32_t* cancel_ptr = static_cast<const uint32_t*>(
                    mappings.acquire(fds.take(req.num_inputs + 1), sizeof(uint32_t)));

                WorkerContext ctx{req,
                                  std::move(input_ptrs),
                                  output_ptr,
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
            mappings.endRequest();
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
