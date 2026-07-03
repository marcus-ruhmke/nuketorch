#pragma once

#include <nuketorch/InferenceClient.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace nuketorch {

namespace detail {
struct PoolEntry {
    PoolEntry(const std::string& worker_binary, const std::string& socket_path, int num_inputs)
        : client(worker_binary, socket_path, num_inputs) {}
    std::mutex mu;
    InferenceClient client;
};
}  // namespace detail

/// Refcounted handle to a pooled worker. Copyable and cheap to pass around;
/// the worker process stops when the last handle is destroyed.
class SharedWorker {
public:
    SharedWorker() = default;

    /// Runs @p fn with exclusive access to the shared InferenceClient,
    /// blocking while another node holds it. Call start() inside on first use
    /// (it is idempotent and also respawns after a worker death). Zero-copy
    /// users must keep mapFrame() and processMappedFrame() within one
    /// withClient call — the mapping is not theirs once the lock is released.
    template <typename Fn>
    auto withClient(Fn&& fn) {
        std::lock_guard<std::mutex> lock(entry_->mu);
        return fn(entry_->client);
    }

    explicit operator bool() const { return entry_ != nullptr; }

private:
    friend class WorkerPool;
    explicit SharedWorker(std::shared_ptr<detail::PoolEntry> entry) : entry_(std::move(entry)) {}

    std::shared_ptr<detail::PoolEntry> entry_;
};

/// Process-wide pool sharing one worker process — and therefore one loaded
/// model on the GPU — across all plugin/node instances that ask for the same
/// key, instead of one worker per node.
///
/// Sharing is within this process only (one Nuke session); cross-process
/// pooling would need a broker daemon and a multi-client protocol, which this
/// library deliberately does not implement. Include a discriminator in
/// @p share_key when different nodes run different models on the same worker
/// binary — otherwise alternating frames make the worker reload the model
/// every frame.
class WorkerPool {
public:
    static WorkerPool& instance();

    /// Returns a handle to the entry for (worker_binary, num_inputs, share_key),
    /// creating it — with a unique socket path under $XDG_RUNTIME_DIR or /tmp —
    /// when no live handle exists.
    SharedWorker acquire(const std::string& worker_binary,
                         int num_inputs,
                         const std::string& share_key = "");

private:
    std::mutex mu_;
    std::unordered_map<std::string, std::weak_ptr<detail::PoolEntry>> entries_;
    unsigned long long seq_ = 0;
};

}  // namespace nuketorch
