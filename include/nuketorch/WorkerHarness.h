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
    /// Deserialized request (dimensions, flags, `params`).
    const InferenceRequest& request;
    /// Mapped pointers for each input plane, in request order.
    std::vector<void*> input_ptrs;
    /// Mapped pointer for the output plane.
    void* output_ptr = nullptr;
    /// Byte size of each input/output shm region (`width * height * channels * sizeof(float)`).
    size_t buffer_bytes = 0;
    /// Returns true once the host has raised the cancel flag for this frame.
    /// Long-running callbacks should poll this between work chunks and throw
    /// nuketorch::CancelledError to stop early; the host maps that to a
    /// CancelledError on its side.
    std::function<bool()> cancelled;
    /// Worker fills profiling data; serialized after `"OK"` on success.
    mutable InferenceMetrics metrics;
};

/// User code that runs Torch (or any) inference; may throw; errors are turned into `ERR|...` replies.
using InferenceCallback = std::function<void(const WorkerContext& ctx)>;
/// Optional handler for the `GPUINFO` control message; return the full wire line (e.g. `"OK|NVIDIA ..."` or `"ERROR|..."`).
using GpuInfoCallback = std::function<std::string()>;

/// Worker executable entry: connect to the argv[1] socket, send `READY|<protocol version>`,
/// then dispatch `PING`/`GPUINFO`/binary inference payloads until `QUIT`.
/// Frame buffers arrive as memfd file descriptors with each request.
/// Installs SIG_IGN for SIGPIPE (a vanished host must not kill the worker mid-write).
/// @return process exit code (0 on normal shutdown).
int workerMain(int argc,
               char** argv,
               InferenceCallback inference,
               GpuInfoCallback gpu_info = nullptr);

}  // namespace nuketorch
