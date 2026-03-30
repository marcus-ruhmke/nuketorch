#pragma once

#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/Protocol.h>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace nuketorch {

/// Arguments passed to the worker's inference callback after shared memory is mapped.
struct WorkerContext {
    /// Deserialized request (dimensions, flags, shm names, `params`).
    const InferenceRequest& request;
    /// Mapped pointers for each `request.shm_inputs` segment, same order.
    std::vector<void*> input_ptrs;
    /// Mapped pointer for `request.shm_output`.
    void* output_ptr = nullptr;
    /// Byte size of each input/output shm region (`width * height * channels * sizeof(float)`).
    size_t buffer_bytes = 0;
    /// Worker fills profiling data; serialized after `"OK"` on success.
    mutable InferenceMetrics metrics;
};

/// User code that runs Torch (or any) inference; may throw; errors are turned into `ERROR|...` replies.
using InferenceCallback = std::function<void(const WorkerContext& ctx)>;
/// Optional handler for the `GPUINFO` control message; return the full wire line (e.g. `"OK|NVIDIA ..."` or `"ERROR|..."`).
using GpuInfoCallback = std::function<std::string()>;

/// Worker executable entry: connect to argv[1] socket, send `READY`, dispatch `PING`/`GPUINFO`/binary inference payloads, `QUIT`/`BYE`.
/// @return process exit code (0 on normal shutdown).
int workerMain(int argc,
               char** argv,
               InferenceCallback inference,
               GpuInfoCallback gpu_info = nullptr);

}  // namespace nuketorch
