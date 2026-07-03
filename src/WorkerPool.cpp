#include <nuketorch/WorkerPool.h>

#include <unistd.h>

#include <cstdlib>

namespace nuketorch {
namespace {

std::string socketDirectory() {
    const char* dir = std::getenv("XDG_RUNTIME_DIR");
    return (dir && *dir) ? dir : "/tmp";
}

}  // namespace

WorkerPool& WorkerPool::instance() {
    static WorkerPool pool;
    return pool;
}

SharedWorker WorkerPool::acquire(const std::string& worker_binary,
                                 int num_inputs,
                                 const std::string& share_key) {
    const std::string key =
        worker_binary + '\0' + std::to_string(num_inputs) + '\0' + share_key;

    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        if (auto held = it->second.lock()) {
            return SharedWorker(std::move(held));
        }
    }

    const std::string socket_path = socketDirectory() + "/nuketorch_pool_" +
                                    std::to_string(getpid()) + "_" + std::to_string(++seq_) +
                                    ".sock";
    auto entry = std::make_shared<detail::PoolEntry>(worker_binary, socket_path, num_inputs);
    entries_[key] = entry;
    return SharedWorker(std::move(entry));
}

}  // namespace nuketorch
