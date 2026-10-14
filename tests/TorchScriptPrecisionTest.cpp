#include <gtest/gtest.h>

#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/Protocol.h>
#include <nuketorch/torch_worker/BackendCache.h>
#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/TorchScriptBackend.h>
#include <nuketorch/torch_worker/TorchWorkerUtils.h>

#include <ATen/Context.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/cuda.h>
#include <torch/script.h>

#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tw = nuketorch::torch_worker;
using Params = std::unordered_map<std::string, std::string>;

namespace {

/// A TorchScript module saved to a temp file for the lifetime of the object.
class SavedModule {
public:
    explicit SavedModule(const torch::jit::Module& m) {
        static int counter = 0;
        path_ = "/tmp/nuketorch_precision_test_" + std::to_string(getpid()) + "_" +
                std::to_string(counter++) + ".pt";
        m.save(path_);
    }
    ~SavedModule() { std::remove(path_.c_str()); }
    SavedModule(const SavedModule&) = delete;
    SavedModule& operator=(const SavedModule&) = delete;
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

/// forward / feat (list[Tensor]) / head / pair (tuple) / sizes (list[int]) /
/// chain (an elementwise chain the CUDA fuser takes once optimised) / conv.
torch::jit::Module multiMethodModule() {
    torch::jit::Module m("MultiMethod");
    torch::manual_seed(0);
    m.register_parameter("w", torch::randn({8, 8}).set_requires_grad(true), /*is_buffer=*/false);
    m.define(R"JIT(
        def forward(self, x):
            return torch.mm(x, self.w)

        def feat(self, x):
            return [torch.mm(x, self.w), torch.mm(x, self.w) + 1.0]

        def head(self, feats: List[Tensor]):
            return torch.mm(feats[0].float(), self.w)

        def pair(self, x):
            return torch.mm(x, self.w), x + 1.0

        def sizes(self, h: int, w: int):
            return [h // 2, w // 2]

        def chain(self, x):
            return torch.sigmoid(x * 1.7 + 0.3) * torch.tanh(x) - torch.exp(x) * 0.01

        def matmul(self, a, b):
            return torch.mm(a, b)

        def conv(self, x, k):
            return torch.conv2d(x, k, None, [1, 1], [1, 1])
    )JIT");
    return m;
}

nuketorch::FrameHeader header(const std::string& model_path, bool use_gpu) {
    nuketorch::FrameHeader h;
    h.width = 8;
    h.height = 8;
    h.channels = 3;
    h.use_gpu = use_gpu;
    h.mixed_precision = false;
    h.model_path = model_path;
    return h;
}

bool cudaAmpere() {
    return torch::cuda::is_available() && tw::computeCapability(torch::Device(torch::kCUDA, 0)).first >= 8;
}

torch::Device testDevice() {
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
}

int autocastNesting() {
    const int n = at::autocast::increment_nesting();
    at::autocast::decrement_nesting();
    return n - 1;
}

struct AutocastState {
    bool enabled;
    at::ScalarType dtype;
    bool cache;
    int nesting;
    bool operator==(const AutocastState& o) const {
        return enabled == o.enabled && dtype == o.dtype && cache == o.cache && nesting == o.nesting;
    }
};

std::ostream& operator<<(std::ostream& os, const AutocastState& s) {
    return os << "{enabled=" << s.enabled << " dtype=" << s.dtype << " cache=" << s.cache
              << " nesting=" << s.nesting << "}";
}

AutocastState autocastState() {
    return {at::autocast::is_autocast_enabled(at::kCUDA), at::autocast::get_autocast_dtype(at::kCUDA),
            at::autocast::is_autocast_cache_enabled(), autocastNesting()};
}

/// What a caller (NNDepth's worker) might do: one region around several calls.
class OuterAutocastRegion {
public:
    explicit OuterAutocastRegion(at::ScalarType dtype) {
        at::autocast::increment_nesting();
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        at::autocast::set_autocast_dtype(at::kCUDA, dtype);
    }
    ~OuterAutocastRegion() {
        if (at::autocast::decrement_nesting() == 0) {
            at::autocast::clear_cache();
        }
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        at::autocast::set_autocast_dtype(at::kCUDA, at::kHalf);
    }
};

tw::PrecisionSpec autocastSpec(tw::AutocastDtype dtype, std::vector<std::string> methods = {}) {
    tw::PrecisionSpec s;
    s.autocast = dtype;
    s.autocast_methods = std::move(methods);
    return s;
}

bool lastGraphContains(const char* needle) {
    const std::shared_ptr<torch::jit::Graph> g = torch::jit::lastExecutedOptimizedGraph();
    return g && g->toString().find(needle) != std::string::npos;
}

/// Restores the real compute-capability query when a test ends.
struct ProbeReset {
    ~ProbeReset() { tw::setComputeCapabilityProbe(nullptr); }
};

/// Restores the calling thread's executor flag when a test ends.
struct ExecutorFlagReset {
    bool saved = torch::jit::getGraphExecutorOptimize();
    ~ExecutorFlagReset() { torch::jit::setGraphExecutorOptimize(saved); }
};

double relError(const torch::Tensor& got, const torch::Tensor& ref) {
    return ((got.to(torch::kDouble) - ref).abs().max() / ref.abs().max()).item<double>();
}

}  // namespace

// ---------------------------------------------------------------- 2. autocast dtype + state

TEST(TorchScriptPrecisionTest, AutocastUsesTheRequestedDtypeAndRestoresState) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    SavedModule mod(multiMethodModule());
    tw::TorchScriptBackend backend;
    backend.load(mod.path(), torch::Device(torch::kCUDA, 0), torch::kFloat32);
    const auto x = torch::randn({8, 8}, torch::TensorOptions().device(torch::kCUDA));

    at::autocast::set_autocast_cache_enabled(false);  // a caller's non-default flag must come back
    const AutocastState before = autocastState();

    backend.setPrecision(autocastSpec(tw::AutocastDtype::bfloat16));
    EXPECT_EQ(backend.runMethod("forward", {x}).toTensor().scalar_type(), torch::kBFloat16);
    EXPECT_EQ(autocastState(), before);

    backend.setPrecision(autocastSpec(tw::AutocastDtype::float16));
    EXPECT_EQ(backend.forward({x})[0].scalar_type(), torch::kFloat16);
    EXPECT_EQ(autocastState(), before);

    backend.setPrecision(tw::PrecisionSpec{});
    EXPECT_EQ(backend.forward({x})[0].scalar_type(), torch::kFloat32);
    EXPECT_EQ(autocastState(), before);
    at::autocast::set_autocast_cache_enabled(true);
}

// ---------------------------------------------------------------- 3. inside an outer region

TEST(TorchScriptPrecisionTest, RegionsNestInsideAnOuterFp16Region) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    SavedModule mod(multiMethodModule());
    tw::TorchScriptBackend backend;
    backend.load(mod.path(), torch::Device(torch::kCUDA, 0), torch::kFloat32);
    const auto x = torch::randn({8, 8}, torch::TensorOptions().device(torch::kCUDA));
    const AutocastState outside = autocastState();
    {
        OuterAutocastRegion outer(at::kHalf);
        const AutocastState inside = autocastState();
        ASSERT_EQ(inside.nesting, outside.nesting + 1);
        const auto dtypeOf = [&](const tw::PrecisionSpec& spec) {
            return backend.runMethod("forward", {x}, spec).toTensor().scalar_type();
        };

        EXPECT_EQ(dtypeOf(autocastSpec(tw::AutocastDtype::float16)), torch::kFloat16);
        EXPECT_EQ(autocastState(), inside);

        // Another dtype inside the region, restored afterwards. (No cross-dtype
        // cast-cache hit: autocast does not cache under InferenceMode.)
        EXPECT_EQ(dtypeOf(autocastSpec(tw::AutocastDtype::bfloat16)), torch::kBFloat16);
        EXPECT_EQ(autocastState(), inside);

        // A spec without autocast runs float32 even inside the region.
        EXPECT_EQ(dtypeOf(tw::PrecisionSpec::ieeeFloat32()), torch::kFloat32);
        EXPECT_EQ(autocastState(), inside);

        // The region itself still autocasts to fp16 afterwards.
        EXPECT_EQ(dtypeOf(autocastSpec(tw::AutocastDtype::float16)), torch::kFloat16);
    }
    EXPECT_EQ(autocastState(), outside);
}

// ---------------------------------------------------------------- 4. named methods

TEST(TorchScriptPrecisionTest, NamedMethodsReturnRawValues) {
    SavedModule mod(multiMethodModule());
    tw::TorchScriptBackend backend;
    backend.load(mod.path(), torch::kCPU, torch::kFloat32);
    const auto x = torch::randn({8, 8});

    const c10::IValue fwd = backend.runMethod("forward", {x});
    ASSERT_TRUE(fwd.isTensor());
    EXPECT_TRUE(torch::equal(fwd.toTensor(), backend.forward({x})[0]));

    const c10::IValue pair = backend.runMethod("pair", {x});
    ASSERT_TRUE(pair.isTuple());
    EXPECT_EQ(pair.toTupleRef().elements().size(), 2u);

    const c10::IValue feat = backend.runMethod("feat", {x});
    ASSERT_TRUE(feat.isTensorList());
    const std::vector<torch::Tensor> feats = feat.toTensorVector();
    ASSERT_EQ(feats.size(), 2u);
    EXPECT_TRUE(torch::allclose(feats[1], feats[0] + 1.0));

    const c10::IValue head = backend.runMethod("head", {feat});
    ASSERT_TRUE(head.isTensor());

    const c10::IValue sizes = backend.runMethod("sizes", {int64_t{252}, int64_t{518}});
    ASSERT_TRUE(sizes.isIntList());
    EXPECT_EQ(sizes.toIntVector(), (std::vector<int64_t>{126, 259}));

    EXPECT_TRUE(backend.hasMethod("feat"));
    EXPECT_FALSE(backend.hasMethod("encode_keyframes"));
    EXPECT_THROW(backend.runMethod("encode_keyframes", {x}), std::invalid_argument);
}

TEST(TorchScriptPrecisionTest, AutocastMethodsScopeTheRegion) {
    if (!cudaAmpere()) {
        GTEST_SKIP() << "needs CUDA compute capability >= 8.0 (bf16)";
    }
    SavedModule mod(multiMethodModule());
    tw::BackendCache cache;
    cache.ensure(header(mod.path(), true),
                 {{"precision", "bf16"}, {"autocast_methods", "feat"}, {"jit_optimize", "0"}});
    tw::InferenceBackend& backend = cache.backend();
    const auto x = torch::randn({8, 8}, torch::TensorOptions().device(torch::kCUDA));

    const c10::IValue feat = backend.runMethod("feat", {x});
    for (const torch::Tensor& t : feat.toTensorVector()) {
        EXPECT_EQ(t.scalar_type(), torch::kBFloat16);
    }
    EXPECT_EQ(backend.runMethod("head", {feat}).toTensor().scalar_type(), torch::kFloat32);
    EXPECT_EQ(backend.runMethod("forward", {x}).toTensor().scalar_type(), torch::kFloat32);
    EXPECT_FALSE(at::autocast::is_autocast_enabled(at::kCUDA));
}

namespace {

/// A backend without named methods (as AOTInductor / TensorRT).
class TwoOutputBackend : public tw::InferenceBackend {
public:
    void load(const std::string& path, torch::Device, torch::ScalarType) override { path_ = path; }
    std::vector<torch::Tensor> forward(const std::vector<torch::jit::IValue>& inputs) override {
        const torch::Tensor x = inputs.at(0).toTensor();
        return {x * 2, x * 3};
    }
    bool isLoaded() const override { return true; }
    const std::string& loadedPath() const override { return path_; }

private:
    std::string path_;
};

}  // namespace

TEST(TorchScriptPrecisionTest, BakedBackendsRunOnlyForward) {
    TwoOutputBackend backend;
    const auto x = torch::ones({2});
    const c10::IValue out = backend.runMethod("forward", {x}, tw::PrecisionSpec::ieeeFloat32());
    ASSERT_TRUE(out.isTuple());
    EXPECT_EQ(out.toTupleRef().elements().size(), 2u);
    EXPECT_TRUE(backend.hasMethod("forward"));
    EXPECT_FALSE(backend.hasMethod("features"));
    EXPECT_THROW(backend.runMethod("features", {x}), std::invalid_argument);
}

// ---------------------------------------------------------------- 5. TF32

TEST(TorchScriptPrecisionTest, Tf32FlagsFollowEveryRequestAndKeepRestoresBoot) {
    SavedModule mod(multiMethodModule());
    tw::applyTf32(tw::Tf32::keep);
    auto& ctx = at::globalContext();
    const bool boot_cublas = ctx.allowTF32CuBLAS();
    const bool boot_cudnn = ctx.allowTF32CuDNN();
    if (!std::getenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")) {
        // libtorch's default: the trap that makes plain float32 convolutions TF32.
        EXPECT_FALSE(boot_cublas);
        EXPECT_TRUE(boot_cudnn);
    }

    tw::BackendCache cache;
    const auto h = header(mod.path(), false);
    cache.ensure(h, {{"precision", "float32"}, {"tf32", "off"}});
    EXPECT_FALSE(ctx.allowTF32CuBLAS());
    EXPECT_FALSE(ctx.allowTF32CuDNN());
    // Same state validate's set_float32_matmul_precision("highest") produces.
    EXPECT_EQ(ctx.float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL),
              at::Float32Precision::IEEE);

    cache.ensure(h, {{"precision", "float32"}});
    EXPECT_EQ(ctx.allowTF32CuBLAS(), boot_cublas);
    EXPECT_EQ(ctx.allowTF32CuDNN(), boot_cudnn);

    // tf32=on is refused off CUDA; the flags themselves are process state, so
    // check them through the low-level setter.
    tw::applyTf32(tw::Tf32::on);
    EXPECT_TRUE(ctx.allowTF32CuBLAS());
    EXPECT_TRUE(ctx.allowTF32CuDNN());
    EXPECT_EQ(ctx.float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL),
              at::Float32Precision::TF32);
    cache.ensure(h, {});
    EXPECT_EQ(ctx.allowTF32CuBLAS(), boot_cublas);
    EXPECT_EQ(ctx.allowTF32CuDNN(), boot_cudnn);

    {
        tw::ScopedTf32 scoped(tw::Tf32::off);
        EXPECT_FALSE(ctx.allowTF32CuDNN());
    }
    EXPECT_EQ(ctx.allowTF32CuDNN(), boot_cudnn);
}


TEST(TorchScriptPrecisionTest, Tf32ChangesCudaMatmulAndConvNumerics) {
    if (!cudaAmpere()) {
        GTEST_SKIP() << "needs CUDA compute capability >= 8.0 (TF32)";
    }
    if (std::getenv("NVIDIA_TF32_OVERRIDE")) {
        GTEST_SKIP() << "NVIDIA_TF32_OVERRIDE pins TF32 below libtorch";
    }
    SavedModule mod(multiMethodModule());
    tw::applyTf32(tw::Tf32::keep);
    const bool boot_cudnn = at::globalContext().allowTF32CuDNN();
    tw::BackendCache cache;
    const auto h = header(mod.path(), true);
    const auto opts = torch::TensorOptions().device(torch::kCUDA);
    torch::manual_seed(1);
    const auto a = torch::randn({2048, 2048}, opts);
    const auto b = torch::randn({2048, 2048}, opts);
    // cuDNN picks a TF32 engine for this shape (not for every shape, e.g. 1x128x64x64).
    const auto img = torch::randn({4, 64, 128, 128}, opts);
    const auto k = torch::randn({64, 64, 3, 3}, opts);
    const auto mm_ref = torch::mm(a.to(torch::kDouble), b.to(torch::kDouble));
    const auto conv_ref = torch::conv2d(img.to(torch::kDouble), k.to(torch::kDouble), {}, 1, 1);

    struct Rel {
        double mm, conv;
    };
    const auto run = [&](const Params& p) {
        cache.ensure(h, p);
        tw::InferenceBackend& backend = cache.backend();
        return Rel{relError(backend.runMethod("matmul", {a, b}).toTensor(), mm_ref),
                   relError(backend.runMethod("conv", {img, k}).toTensor(), conv_ref)};
    };
    const Rel off = run({{"precision", "float32"}, {"tf32", "off"}});
    const Rel on = run({{"precision", "float32"}, {"tf32", "on"}});
    const Rel keep = run({{"precision", "float32"}});  // the boot state
    const Rel ieee_alias = run({{"precision", "ieee"}});

    std::cout << "[tf32] rel error vs float64 — matmul: off=" << off.mm << " on=" << on.mm
              << " keep=" << keep.mm << " | conv: off=" << off.conv << " on=" << on.conv
              << " keep=" << keep.conv << "\n";
    EXPECT_LT(off.mm, 1e-5);
    EXPECT_LT(off.conv, 2e-5);
    EXPECT_GT(on.mm, 5e-5);
    EXPECT_GT(on.mm, 20 * off.mm);
    EXPECT_GT(on.conv, 5e-5);
    EXPECT_GT(on.conv, 5 * off.conv);
    EXPECT_EQ(ieee_alias.mm, off.mm);
    EXPECT_EQ(ieee_alias.conv, off.conv);
    if (boot_cudnn) {
        // The trap: precision=float32 alone keeps cuDNN TF32 on.
        EXPECT_GT(keep.conv, 5 * off.conv);
    }
}

// ---------------------------------------------------------------- 6. executor guard

namespace {

/// Runs `chain` on @p frames in @p order through a freshly loaded module.
std::vector<torch::Tensor> runChain(const std::string& path, bool use_gpu, const Params& params,
                                    const std::vector<torch::Tensor>& frames,
                                    const std::vector<size_t>& order, bool* saw_profiling = nullptr) {
    tw::BackendCache cache;
    cache.ensure(header(path, use_gpu), params);
    std::vector<torch::Tensor> out(frames.size());
    for (size_t i : order) {
        out[i] = cache.backend().runMethod("chain", {frames[i]}).toTensor().cpu();
        if (saw_profiling) {
            *saw_profiling = *saw_profiling || lastGraphContains("prim::profile");
        }
    }
    return out;
}

}  // namespace

TEST(TorchScriptPrecisionTest, JitOptimizeOffIsFrameOrderIndependent) {
    SavedModule mod(multiMethodModule());
    const bool gpu = torch::cuda::is_available();
    const torch::Device device = testDevice();
    torch::manual_seed(2);
    std::vector<torch::Tensor> frames;
    for (int i = 0; i < 6; ++i) {
        frames.push_back(torch::randn({256, 256}, torch::TensorOptions().device(device)));
    }
    const std::vector<size_t> forward_order = {0, 1, 2, 3, 4, 5};
    const std::vector<size_t> reverse_order = {5, 4, 3, 2, 1, 0};
    ASSERT_TRUE(torch::jit::getGraphExecutorOptimize());

    bool profiled = false;
    const auto a = runChain(mod.path(), gpu, {{"jit_optimize", "0"}}, frames, forward_order, &profiled);
    const auto b = runChain(mod.path(), gpu, {{"jit_optimize", "0"}}, frames, reverse_order, &profiled);
    EXPECT_FALSE(profiled) << "a call ran the profiling executor under jit_optimize=0";
    for (size_t i = 0; i < frames.size(); ++i) {
        EXPECT_TRUE(torch::equal(a[i], b[i])) << "frame " << i;
    }
    EXPECT_TRUE(torch::jit::getGraphExecutorOptimize()) << "thread flag not restored";

    // Control: the optimising executor profiles its first call(s) — the
    // probe above is able to see it.
    bool profiled_on = false;
    const auto c = runChain(mod.path(), gpu, {{"jit_optimize", "1"}}, frames, forward_order, &profiled_on);
    const auto d = runChain(mod.path(), gpu, {{"jit_optimize", "1"}}, frames, reverse_order);
    EXPECT_TRUE(profiled_on);
    int differ = 0;
    for (size_t i = 0; i < frames.size(); ++i) {
        differ += torch::equal(c[i], d[i]) ? 0 : 1;
    }
    std::cout << "[executor] jit_optimize=1 on " << device << ": " << differ << " of " << frames.size()
              << " frames depend on the call order (diagnostic)\n";
}

// ---------------------------------------------------------------- 7. jit_optimize reloads

TEST(TorchScriptPrecisionTest, JitOptimizeChangeForcesReload) {
    SavedModule mod(multiMethodModule());
    tw::BackendCache cache;
    const auto h = header(mod.path(), false);
    const auto x = torch::randn({8, 8});
    nuketorch::InferenceMetrics m;

    cache.ensure(h, {}, &m);
    EXPECT_GE(m.model_load_ms, 0);
    cache.backend().runMethod("forward", {x});  // fixes the optimised plan
    cache.ensure(h, {{"jit_optimize", "1"}}, &m);
    EXPECT_EQ(m.model_load_ms, -1) << "absent == the thread's flag (on): no reload";

    cache.ensure(h, {{"jit_optimize", "0"}}, &m);
    EXPECT_GE(m.model_load_ms, 0) << "jit_optimize 1 -> 0 must reload";
    cache.backend().runMethod("forward", {x});
    EXPECT_FALSE(lastGraphContains("prim::profile"));
    cache.ensure(h, {{"jit_optimize", "0"}}, &m);
    EXPECT_EQ(m.model_load_ms, -1);

    cache.ensure(h, {{"jit_optimize", "1"}}, &m);
    EXPECT_GE(m.model_load_ms, 0) << "jit_optimize 0 -> 1 must reload";
}

TEST(TorchScriptPrecisionTest, AbsentJitOptimizeLeavesTheThreadFlagAlone) {
    // NNVignette's worker turns the optimiser off on its inference thread and
    // sends no jit_optimize: nuketorch must not switch it back on.
    ExecutorFlagReset reset;
    torch::jit::setGraphExecutorOptimize(false);
    SavedModule mod(multiMethodModule());
    tw::BackendCache cache;
    const auto h = header(mod.path(), false);
    nuketorch::InferenceMetrics m;
    cache.ensure(h, {{"precision", "float32"}}, &m);
    EXPECT_EQ(m.dtype, "float32");
    cache.backend().runMethod("chain", {torch::randn({4, 4})});
    EXPECT_FALSE(lastGraphContains("prim::profile"));
    EXPECT_FALSE(torch::jit::getGraphExecutorOptimize());

    cache.ensure(h, {{"jit_optimize", "0"}}, &m);
    EXPECT_EQ(m.model_load_ms, -1) << "loaded under the thread's 0 already";
}

// ---------------------------------------------------------------- 8. compute capability (mocked)

TEST(TorchScriptPrecisionTest, MockedTuringCardRefusesBf16AndTf32) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    ProbeReset reset;
    tw::setComputeCapabilityProbe([](const torch::Device&) { return std::make_pair(7, 5); });
    SavedModule mod(multiMethodModule());
    tw::BackendCache cache;
    const auto h = header(mod.path(), true);
    try {
        cache.ensure(h, {{"precision", "bf16"}, {"tf32", "off"}, {"jit_optimize", "0"}});
        FAIL() << "bf16 on a mocked 7.5 card must be refused";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("7.5"), std::string::npos) << e.what();
    }
    EXPECT_THROW(cache.ensure(h, {{"precision", "float32"}, {"tf32", "on"}}), std::invalid_argument);

    nuketorch::InferenceMetrics m;
    cache.ensure(h, {{"precision", "autocast"}, {"autocast_methods", "feat"}, {"jit_optimize", "0"}}, &m);
    EXPECT_EQ(m.dtype, "autocast(float16;feat),jit0");
    const auto x = torch::randn({8, 8}, torch::TensorOptions().device(torch::kCUDA));
    EXPECT_EQ(cache.backend().runMethod("feat", {x}).toTensorVector()[0].scalar_type(), torch::kFloat16);

    m = {};
    cache.ensure(h, {{"precision", "bf16"}, {"jit_optimize", "0"}, {"precision_policy", "fallback_fp32"}},
                 &m);
    EXPECT_EQ(m.dtype, "float32/ieee(fallback:bf16),jit0");
    EXPECT_NE(m.notes.find("fallback"), std::string::npos);
    EXPECT_EQ(cache.backend().runMethod("forward", {x}).toTensor().scalar_type(), torch::kFloat32);
}

// ---------------------------------------------------------------- 9. CPU

TEST(TorchScriptPrecisionTest, CpuRefusesBf16AndFallsBackOnRequest) {
    SavedModule mod(multiMethodModule());
    tw::BackendCache cache;
    const auto h = header(mod.path(), false);
    EXPECT_THROW(cache.ensure(h, {{"precision", "bf16"}}), std::invalid_argument);
    EXPECT_THROW(cache.ensure(h, {{"precision", "autocast"}, {"autocast_methods", "feat"}}),
                 std::invalid_argument);

    nuketorch::InferenceMetrics m;
    cache.ensure(h, {{"precision", "bf16"}, {"precision_policy", "fallback_fp32"}}, &m);
    EXPECT_EQ(m.dtype, "float32/ieee(fallback:bf16)");
    EXPECT_EQ(m.notes, "precision fallback: bf16 -> float32/ieee");
    EXPECT_EQ(cache.fallbackNote(), "bf16");
    EXPECT_EQ(cache.spec().autocast, tw::AutocastDtype::none);
    EXPECT_EQ(cache.backend().runMethod("forward", {torch::randn({8, 8})}).toTensor().scalar_type(),
              torch::kFloat32);

    // Legacy autocast still runs float32 off CUDA without complaint.
    m = {};
    cache.ensure(h, {{"precision", "autocast"}}, &m);
    EXPECT_EQ(m.dtype, "float32");
    EXPECT_TRUE(m.notes.empty());
    EXPECT_TRUE(cache.fallbackNote().empty());
}

// ---------------------------------------------------------------- 11. cuDNN benchmark

TEST(TorchScriptPrecisionTest, CudnnBenchmarkIsPinnedPerRequest) {
    SavedModule mod(multiMethodModule());
    tw::BackendCache cache;
    const auto h = header(mod.path(), false);
    auto& ctx = at::globalContext();
    const bool before = ctx.benchmarkCuDNN();

    cache.ensure(h, {{"cudnn_benchmark", "1"}});
    EXPECT_TRUE(ctx.benchmarkCuDNN());
    cache.ensure(h, {});
    EXPECT_TRUE(ctx.benchmarkCuDNN()) << "absent keeps";
    cache.ensure(h, {{"cudnn_benchmark", "0"}});
    EXPECT_FALSE(ctx.benchmarkCuDNN());
    ctx.setBenchmarkCuDNN(before);
}

// ---------------------------------------------------------------- utilities

TEST(TorchScriptPrecisionTest, AllFiniteWalksNestedValues) {
    const auto ok = torch::ones({3});
    auto bad = torch::ones({3});
    bad[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(tw::allFinite(ok));
    EXPECT_FALSE(tw::allFinite(bad));
    EXPECT_FALSE(tw::allFinite(c10::ivalue::Tuple::create({ok, bad})));
    EXPECT_FALSE(tw::allFinite(c10::List<torch::Tensor>({ok, bad.to(torch::kBFloat16)})));
    EXPECT_TRUE(tw::allFinite(c10::List<int64_t>({1, 2})));
    EXPECT_TRUE(tw::allFinite(c10::IValue()));
    auto inf = torch::full({2}, std::numeric_limits<float>::infinity());
    EXPECT_FALSE(tw::allFinite(c10::ivalue::Tuple::create({ok, c10::ivalue::Tuple::create({inf})})));
    EXPECT_TRUE(tw::allFinite(torch::arange(4)));  // integer tensors are always finite
}
