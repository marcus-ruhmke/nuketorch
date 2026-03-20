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

---

## Step 1: Worker executable

Implement `main()` with `nuketorch::workerMain`. You supply:

1. **`InferenceCallback`** — map shared memory to tensors, run `torch::jit` (or custom ops), write the output shm.
2. **`GpuInfoCallback`** (optional) — return a full line for the host, e.g. `"OK|NVIDIA RTX ..."` or `"ERROR|No CUDA"`.

Minimal shape:

```cpp
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
cfg.mixed_precision = useMp;
cfg.debug = debug;
cfg.params["alpha"] = std::to_string(alphaKnob);

nuketorch::FrameBuffers fb;
fb.inputs = { planeA, planeB };
fb.output = outPlane;
fb.width = w;
fb.height = h;
fb.channels = c;

worker->processFrame(fb, cfg, [this]() { return aborted() || cancelled(); });
```

**Canonical reference:** `../nnRetime/src/nnRetime.cpp` (`renderStripe`, worker restart/`ping`, mutex if you serialize GPU access across stripes).

---

## Step 3: CMake

Two targets:

| Target | Sources | Link |
|--------|---------|------|
| Plugin module | Nuke-facing `.cpp`, utils | `Nuke::NDK`, `nuketorch::nuketorch`, logging |
| Worker exe | `*Worker.cpp` | `${TORCH_LIBRARIES}`, `nuketorch::nuketorch`, `${TORCH_CXX_FLAGS}` |

Match **CXX ABI** flags to Nuke’s toolchain for the plugin (nnRetime uses `_GLIBCXX_USE_CXX11_ABI=0` on the plugin and ABI 1 on the worker).

**Reference:** `../nnRetime/CMakeLists.txt` — `FetchContent`/`SOURCE_DIR` for nuketorch, `add_nuke_plugin`, `nnRetimeWorker` executable, `BUILD_RPATH` for libtorch.

---

## Step 4: Tests

### Fake worker (no Torch)

Build a test executable that links only nuketorch and implements a deterministic callback (e.g. blend inputs + a param). See [`tests/FakeWorker.cpp`](../tests/FakeWorker.cpp) in this repository — same pattern: `workerMain` + trivial math on `float*`.

### InferenceClient integration test

`nuketorch::InferenceClient client(FAKE_WORKER_BIN, uniqueSocket, numInputs);`  
`start();` `ping();` `processFrame(...);` `stop();`  
Use GoogleTest; link `gtest_main` and `rt` if needed.

### Lifecycle test against the real worker

Parent creates `nuketorch::IPCServer`, `fork` + `execl` your real worker binary with the socket path, expect `READY`, `PING`/`PONG`, `QUIT`/`BYE`. See `../nnRetime/tests/WorkerLifecycleTest.cpp`.

---

## Protocol reminder

- Control messages are plain ASCII (`READY`, `PING`, `GPUINFO`, …).
- Frame jobs are binary blobs from `nuketorch::serialize` / `deserialize` (`InferenceRequest`).
- Keep worker-only keys documented for your team (string map in `params`).

---

## See also

- [README.md](../README.md) — build, install, header index.
- [InferenceClient.h](../include/nuketorch/InferenceClient.h) — host API.
- [WorkerHarness.h](../include/nuketorch/WorkerHarness.h) — worker API.
