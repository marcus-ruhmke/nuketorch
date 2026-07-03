# Writing a Nuke ML plugin with nuketorch

This guide assumes you clone **nuketorch** next to your plugin repo (or install nuketorch and use `find_package`). The reference implementation is **nnRetime** (`../nnRetime` when both live under the same parent directory).

## Prerequisites

- **nuketorch** built or installed (headers + static library `nuketorch::nuketorch`).
- **libtorch** for the worker only (CMake `find_package(Torch REQUIRED)`).
- **Nuke NDK** for the plugin only (`find_package(Nuke REQUIRED)` / `Nuke::NDK`).
- Linux: POSIX shared memory and Unix domain sockets.

## Recommended layout

```text
myPlugin/
  CMakeLists.txt
  src/
    myPlugin.cpp          # PlanarIop (or other Iop)
    myPlugin.h
    myPluginWorker.cpp    # torch + workerMain
  resources/
    init.py menu.py *.pt
  tests/
    CMakeLists.txt
    FakeWorker.cpp
    InferenceClientTest.cpp
    WorkerLifecycleTest.cpp
```

Ship the worker executable next to the plugin `.so` (same directory as Nuke’s `NUKE_PATH` entry), mirroring nnRetime.

If you launch the worker through a wrapper script (e.g. to set `LD_LIBRARY_PATH`), end the script with `exec ./realWorker "$@"`: the client verifies via `SO_PEERCRED` that the connecting process is the pid it spawned, so a wrapper that forks instead of exec-ing is rejected.

---

## Step 1: Worker executable

Implement `main()` with `nuketorch::workerMain`. You supply:

1. **`InferenceCallback`** — map shared memory to tensors, run `torch::jit` (or custom ops), write the output shm.
2. **`GpuInfoCallback`** (optional) — return a full line for the host, e.g. `"OK|NVIDIA RTX ..."` or `"ERROR|No CUDA"`.

Minimal shape:

```cpp
#include <nuketorch/Errors.h>
#include <nuketorch/WorkerHarness.h>
#include <torch/torch.h>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  MyCache cache;  // model, device, pinned buffers, etc.

  auto gpu_info = []() -> std::string {
    if (torch::cuda::is_available()) { /* snprintf OK|... */ return "OK|..."; }
    return "ERROR|No CUDA GPU found";
  };

  auto inference = [&cache](const nuketorch::WorkerContext& ctx) {
    const auto& p = ctx.request.params;
    float alpha = std::stof(p.at("alpha"));  // example model-specific key
    (void)alpha;
    // Use ctx.input_ptrs[i], ctx.output_ptr, ctx.buffer_bytes
    // ctx.request.header.{width,height,channels,model_path,...}

    // Long-running models: poll ctx.cancelled() between work chunks so the
    // artist's cancel takes effect immediately instead of after the frame.
    if (ctx.cancelled()) {
      throw nuketorch::CancelledError("cancelled between passes");
    }
  };

  return nuketorch::workerMain(argc, argv, inference, gpu_info);
}
```

**Full example:** see `../nnRetime/src/nnRetimeWorker.cpp` — caching, `CudaPinnedMemory`, `torch::jit::load`, and parsing multiple params from `ctx.request.params`.

---

## Step 2: Nuke plugin (host)

- Include `<nuketorch/InferenceClient.h>`.
- Resolve paths: plugin directory for `.so`, same dir for the worker binary name (e.g. `getPluginPath() + "/myPluginWorker"`).
- Construct `nuketorch::InferenceClient worker(workerPath, socketPath, numInputs)`; call `start()` when safe (constructor or `_validate`).
- In `renderStripe()` (or equivalent), fill `nuketorch::FrameBuffers` with pointers to Nuke planar float data and `nuketorch::InferenceConfig` with `model_path`, GPU flags, and string `params` your worker expects.

Example fragment:

```cpp
nuketorch::InferenceConfig cfg;
cfg.model_path = modelPathOnDisk;
cfg.use_gpu = useGpu;
cfg.mixed_precision = useMp;   // note: full FP16 conversion, not autocast
cfg.debug = debug;
cfg.frame_timeout_ms = 120000; // watchdog: kill + TimeoutError instead of a hung GUI thread
cfg.params["alpha"] = std::to_string(alphaKnob);

nuketorch::FrameBuffers fb;
fb.inputs = { planeA, planeB };
fb.output = outPlane;
fb.width = w;
fb.height = h;
fb.channels = c;

try {
  worker->processFrame(fb, cfg, [this]() { return aborted() || cancelled(); });
} catch (const nuketorch::CancelledError&) {
  // user cancelled; not an error
} catch (const nuketorch::WorkerDiedError& e) {
  // e.what() includes the worker's exit status and a stderr tail;
  // worker->start() respawns and recovers
}
```

`InferenceClient` is **not** thread-safe: serialize access (Nuke calls render
entry points from several threads). For large frames, the zero-copy path skips
two full-frame copies: `mapFrame(w, h, c)` returns direct pointers into shared
memory, fill them in place, then `processMappedFrame(cfg, ...)` (no vertical
flip is applied on this path — flip in the worker with `tensor.flip(-2)` on
GPU, where it is effectively free).

### Sharing one worker across node instances

Ten instances of your node should not mean ten libtorch processes each holding
the model on the GPU. `WorkerPool` shares one worker per key within the Nuke
process and serializes access for you:

```cpp
#include <nuketorch/WorkerPool.h>

// Use the model path as the share key so nodes with different models get
// different workers instead of thrashing reloads.
auto worker = nuketorch::WorkerPool::instance().acquire(workerPath, /*num_inputs=*/2,
                                                        modelPathOnDisk);
worker.withClient([&](nuketorch::InferenceClient& c) {
  c.start();  // idempotent; also respawns after a worker death
  c.processFrame(fb, cfg, [this]() { return aborted(); }, &metrics);
});
```

The worker stops when the last node holding a handle is destroyed. Sharing is
per-process; cross-process pooling (multiple Nuke sessions sharing one worker)
would need a broker daemon and is deliberately out of scope.

### Precision (torch_worker backends)

`cfg.mixed_precision = true` keeps its historical meaning: the whole module is
converted to FP16. For real mixed precision set
`cfg.params["precision"] = "autocast"` — weights stay FP32 and eligible ops run
FP16 at forward time (TorchScript backend, CUDA only; AOTInductor/TensorRT
artifacts have precision baked in). `"float32"` forces full precision.

**Canonical reference:** `../nnRetime/src/nnRetime.cpp` (`renderStripe`, worker restart/`ping`, mutex if you serialize GPU access across stripes).

---

## Step 3: CMake

Two targets:

| Target | Sources | Link |
|--------|---------|------|
| Plugin module | Nuke-facing `.cpp`, utils | `Nuke::NDK`, `nuketorch::nuketorch`, logging |
| Worker exe | `*Worker.cpp` | `${TORCH_LIBRARIES}`, `nuketorch::nuketorch`, `${TORCH_CXX_FLAGS}` |

Match **CXX ABI** flags to Nuke’s toolchain for the plugin (nnRetime uses `_GLIBCXX_USE_CXX11_ABI=0` on the plugin and ABI 1 on the worker).

**libtorch:** After making nuketorch available (`FetchContent` or `find_package`), add its module path and include `FetchLibtorch` **before** `find_package(Torch REQUIRED)` so workers can auto-download a pinned libtorch (`-DTORCH_VERSION=…`, `-DCUDA_VARIANT=…`) or you can pass `-DLIBTORCH_ROOT=…` to use an existing install.

**Reference:** `../nnRetime/CMakeLists.txt` — `FetchContent`/`SOURCE_DIR` for nuketorch, `include(FetchLibtorch)`, `add_nuke_plugin`, `nnRetimeWorker` executable, `BUILD_RPATH` for libtorch.

---

## Step 4: Tests

### Fake worker (no Torch)

Build a test executable that links only nuketorch and implements a deterministic callback (e.g. blend inputs + a param). See [`tests/FakeWorker.cpp`](../tests/FakeWorker.cpp) in this repository — same pattern: `workerMain` + trivial math on `float*`.

### InferenceClient integration test

`nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocket, numInputs);`  
`start();` `ping();` `processFrame(...);` `stop();`  
Use GoogleTest; link `gtest_main` and `rt` if needed.

### Lifecycle test against the real worker

Parent creates `nuketorch::IPCServer`, spawns your real worker binary with the socket path, expects `READY|2` (the protocol version), then `PING|1` → `R|1|PONG` and `QUIT|2` → `R|2|BYE`. See [`tests/WorkerHarnessTest.cpp`](../tests/WorkerHarnessTest.cpp) in this repository for the v2 message shapes.

---

## Protocol reminder (v2)

- The worker announces `READY|<protocol version>`; the client refuses a mismatch with a clear error, so a stale worker binary fails fast instead of mysteriously.
- Control messages carry request ids (`PING|7` → `R|7|PONG`); a timed-out reply is discarded by id and can never desynchronize the stream.
- Frame jobs are binary blobs from `nuketorch::serialize` / `deserialize` (`InferenceRequest`); the frame buffers travel as `memfd` file descriptors attached to the same message, ordered `[inputs..., output, cancel-flag]`.
- Keep worker-only keys documented for your team (string map in `params`).
- Debugging a startup freeze? Set `NUKETORCH_FREEZE_LOG=/path/to/log` in both processes for an fsync-per-line trace that survives a wedged machine.

---

## See also

- [README.md](../README.md) — build, install, header index.
- [InferenceClient.h](../include/nuketorch/InferenceClient.h) — host API.
- [WorkerHarness.h](../include/nuketorch/WorkerHarness.h) — worker API.
