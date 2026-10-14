#include <gtest/gtest.h>

#include <nuketorch/torch_worker/BackendParams.h>
#include <nuketorch/torch_worker/TorchScriptBackend.h>

#include <torch/cuda.h>
#include <torch/script.h>

#include <unistd.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>

using nuketorch::torch_worker::AutocastDtype;
using nuketorch::torch_worker::PrecisionMode;
using nuketorch::torch_worker::PrecisionPolicy;
using nuketorch::torch_worker::PrecisionSpec;
using nuketorch::torch_worker::PrecisionTarget;
using nuketorch::torch_worker::Tf32;
using nuketorch::torch_worker::precisionLabel;
using nuketorch::torch_worker::precisionModeFromParams;
using nuketorch::torch_worker::precisionSpecFromParams;
using nuketorch::torch_worker::resolvePrecision;

using Params = std::unordered_map<std::string, std::string>;

namespace {

PrecisionTarget cudaTarget(int major, int minor) {
    PrecisionTarget t;
    t.device = PrecisionTarget::Device::cuda;
    t.cc_major = major;
    t.cc_minor = minor;
    return t;
}

PrecisionTarget cpuTarget() {
    return PrecisionTarget{};
}

/// Parse -> resolve -> label, as BackendCache::ensure does.
std::string labelFor(const Params& params, const PrecisionTarget& target, bool mixed_precision = false) {
    std::string note;
    const PrecisionSpec s = resolvePrecision(precisionSpecFromParams(params, mixed_precision), target, &note);
    return precisionLabel(s, target.baked, note);
}

}  // namespace

// ---------------------------------------------------------------- legacy API

TEST(BackendPrecisionTest, DefaultFollowsMixedPrecisionFlag) {
    EXPECT_EQ(precisionModeFromParams({}, true), PrecisionMode::half);
    EXPECT_EQ(precisionModeFromParams({}, false), PrecisionMode::full);
}

TEST(BackendPrecisionTest, ParsesAliasesCaseInsensitively) {
    EXPECT_EQ(precisionModeFromParams({{"precision", "AutoCast"}}, true), PrecisionMode::autocast);
    EXPECT_EQ(precisionModeFromParams({{"precision", "amp"}}, false), PrecisionMode::autocast);
    EXPECT_EQ(precisionModeFromParams({{"precision", "fp16"}}, false), PrecisionMode::half);
    EXPECT_EQ(precisionModeFromParams({{"precision", "float32"}}, true), PrecisionMode::full);
}

TEST(BackendPrecisionTest, RejectsUnknownValue) {
    EXPECT_THROW(precisionModeFromParams({{"precision", "quantum"}}, true), std::invalid_argument);
}

TEST(BackendPrecisionTest, LegacyModeMapsNewValuesOntoTheWeightsDtype) {
    // mlHumanMatte derives its input dtype from this: only `half` means FP16 weights.
    EXPECT_EQ(precisionModeFromParams({{"precision", "bf16"}}, true), PrecisionMode::autocast);
    EXPECT_EQ(precisionModeFromParams({{"precision", "tf32"}}, true), PrecisionMode::full);
    EXPECT_EQ(precisionModeFromParams({{"precision", "ieee"}}, true), PrecisionMode::full);
    EXPECT_EQ(precisionModeFromParams({{"precision", "float"}}, true), PrecisionMode::full);
    EXPECT_EQ(precisionModeFromParams({{"precision", "float16"}}, false), PrecisionMode::half);
}

// ---------------------------------------------------------------- 1. parsing

TEST(PrecisionSpecTest, AbsentParamsKeepThePre03Behaviour) {
    for (const bool mixed : {true, false}) {
        const PrecisionSpec s = precisionSpecFromParams({}, mixed);
        EXPECT_EQ(s.half_weights, mixed);
        EXPECT_EQ(s.autocast, AutocastDtype::none);
        EXPECT_TRUE(s.autocast_methods.empty());
        EXPECT_EQ(s.tf32, Tf32::keep);
        EXPECT_FALSE(s.cudnn_benchmark.has_value());
        EXPECT_FALSE(s.jit_optimize.has_value());
        EXPECT_EQ(s.policy, PrecisionPolicy::error);
    }
}

TEST(PrecisionSpecTest, LegacyValuesParseAsBefore) {
    const PrecisionSpec half = precisionSpecFromParams({{"precision", "HALF"}}, false);
    EXPECT_TRUE(half.half_weights);
    EXPECT_EQ(half.autocast, AutocastDtype::none);

    const PrecisionSpec amp = precisionSpecFromParams({{"precision", "amp"}}, true);
    EXPECT_FALSE(amp.half_weights);
    EXPECT_EQ(amp.autocast, AutocastDtype::float16);

    for (const char* v : {"float32", "fp32", "float", "full"}) {
        const PrecisionSpec s = precisionSpecFromParams({{"precision", v}}, true);
        EXPECT_FALSE(s.half_weights) << v;
        EXPECT_EQ(s.autocast, AutocastDtype::none) << v;
        EXPECT_EQ(s.tf32, Tf32::keep) << v << ": float32 alone must not touch TF32";
    }
}

TEST(PrecisionSpecTest, ParsesNewValues) {
    for (const char* v : {"bf16", "BFloat16"}) {
        const PrecisionSpec s = precisionSpecFromParams({{"precision", v}}, true);
        EXPECT_FALSE(s.half_weights) << v;
        EXPECT_EQ(s.autocast, AutocastDtype::bfloat16) << v;
    }
    EXPECT_EQ(precisionSpecFromParams({{"precision", "tf32"}}, true).tf32, Tf32::on);
    EXPECT_EQ(precisionSpecFromParams({{"precision", "ieee"}}, true).tf32, Tf32::off);
    EXPECT_EQ(precisionSpecFromParams({{"precision", "tf32"}, {"tf32", "on"}}, true).tf32, Tf32::on);

    const PrecisionSpec s = precisionSpecFromParams({{"precision", "bf16"},
                                                     {"tf32", "OFF"},
                                                     {"autocast_methods", " features , fuse "},
                                                     {"jit_optimize", "0"},
                                                     {"cudnn_benchmark", "1"},
                                                     {"precision_policy", "fallback_fp32"}},
                                                    true);
    EXPECT_EQ(s.tf32, Tf32::off);
    EXPECT_EQ(s.autocast_methods, (std::vector<std::string>{"features", "fuse"}));
    EXPECT_EQ(s.jit_optimize, std::optional<bool>(false));
    EXPECT_EQ(s.cudnn_benchmark, std::optional<bool>(true));
    EXPECT_EQ(s.policy, PrecisionPolicy::fallback_fp32);
    EXPECT_TRUE(s.autocastApplies("features"));
    EXPECT_TRUE(s.autocastApplies("fuse"));
    EXPECT_FALSE(s.autocastApplies("forward"));

    EXPECT_EQ(precisionSpecFromParams({{"tf32", "on"}}, false).tf32, Tf32::on);
    EXPECT_EQ(precisionSpecFromParams({{"jit_optimize", "1"}}, false).jit_optimize,
              std::optional<bool>(true));
}

TEST(PrecisionSpecTest, RejectsBadAndContradictoryValues) {
    const Params bad[] = {
        {{"precision", "int8"}},
        {{"tf32", "maybe"}},
        {{"precision", "tf32"}, {"tf32", "off"}},
        {{"precision", "ieee"}, {"tf32", "on"}},
        {{"jit_optimize", "2"}},
        {{"cudnn_benchmark", "yes please"}},
        {{"precision_policy", "fallback"}},
        {{"precision", "bf16"}, {"autocast_methods", ""}},
        {{"precision", "bf16"}, {"autocast_methods", "features,,fuse"}},
        {{"precision", "bf16"}, {"autocast_methods", "fea-tures"}},
        // autocast_methods without an autocast precision is a mapping bug, not a no-op.
        {{"precision", "float32"}, {"autocast_methods", "features"}},
        {{"autocast_methods", "features"}},
    };
    for (const Params& p : bad) {
        EXPECT_THROW(precisionSpecFromParams(p, true), std::invalid_argument)
            << p.begin()->first << "=" << p.begin()->second;
    }
}

TEST(PrecisionSpecTest, AutocastAppliesToEveryMethodWithoutAList) {
    PrecisionSpec s;
    EXPECT_FALSE(s.autocastApplies("forward"));
    s.autocast = AutocastDtype::bfloat16;
    EXPECT_TRUE(s.autocastApplies("forward"));
    EXPECT_TRUE(s.autocastApplies("encode_keyframes"));
}

TEST(PrecisionSpecTest, IeeeFloat32IsPlainFloat32WithTf32Off) {
    const PrecisionSpec s = PrecisionSpec::ieeeFloat32();
    EXPECT_FALSE(s.half_weights);
    EXPECT_EQ(s.autocast, AutocastDtype::none);
    EXPECT_EQ(s.tf32, Tf32::off);
}

// ---------------------------------------------------------------- 8. compute capability

TEST(ResolvePrecisionTest, Bf16AndTf32NeedAmpere) {
    const PrecisionSpec bf16 = precisionSpecFromParams({{"precision", "bf16"}}, false);
    try {
        resolvePrecision(bf16, cudaTarget(7, 5));
        FAIL() << "bf16 on 7.5 must be refused";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("bf16"), std::string::npos) << msg;
        EXPECT_NE(msg.find("7.5"), std::string::npos) << msg;
        EXPECT_NE(msg.find("8.0"), std::string::npos) << msg;
    }
    EXPECT_EQ(resolvePrecision(bf16, cudaTarget(8, 0)).autocast, AutocastDtype::bfloat16);

    const PrecisionSpec tf32 = precisionSpecFromParams({{"tf32", "on"}}, false);
    try {
        resolvePrecision(tf32, cudaTarget(7, 5));
        FAIL() << "tf32=on on 7.5 must be refused";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("tf32=on"), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("7.5"), std::string::npos) << e.what();
    }
    EXPECT_EQ(resolvePrecision(tf32, cudaTarget(8, 6)).tf32, Tf32::on);
    // TF32 off is IEEE float32 on every card.
    const PrecisionSpec off = precisionSpecFromParams({{"tf32", "off"}}, false);
    EXPECT_EQ(resolvePrecision(off, cudaTarget(6, 1)).tf32, Tf32::off);
}

TEST(ResolvePrecisionTest, Fp16AutocastRunsOnAnyCudaCard) {
    const PrecisionSpec amp =
        precisionSpecFromParams({{"precision", "autocast"}, {"autocast_methods", "features"}}, false);
    const PrecisionSpec s = resolvePrecision(amp, cudaTarget(7, 5));
    EXPECT_EQ(s.autocast, AutocastDtype::float16);
    EXPECT_EQ(s.autocast_methods, (std::vector<std::string>{"features"}));
}

TEST(ResolvePrecisionTest, FallbackOnOldCardIsIeeeFloat32WithANote) {
    const PrecisionSpec req = precisionSpecFromParams(
        {{"precision", "bf16"}, {"jit_optimize", "0"}, {"precision_policy", "fallback_fp32"}}, false);
    std::string note;
    const PrecisionSpec s = resolvePrecision(req, cudaTarget(7, 5), &note);
    EXPECT_EQ(s.autocast, AutocastDtype::none);
    EXPECT_EQ(s.tf32, Tf32::off);
    EXPECT_EQ(s.jit_optimize, std::optional<bool>(false));
    EXPECT_EQ(note, "bf16");
    EXPECT_EQ(precisionLabel(s, false, note), "float32/ieee(fallback:bf16),jit0");
}

// ---------------------------------------------------------------- 9. non-CUDA devices

TEST(ResolvePrecisionTest, CpuRefusesNewModesUnlessFallback) {
    const Params refused[] = {
        {{"precision", "bf16"}},
        {{"precision", "tf32"}},
        {{"tf32", "on"}},
        {{"precision", "autocast"}, {"autocast_methods", "features"}},
        {{"precision", "bf16"}, {"autocast_methods", "features"}},
    };
    for (Params p : refused) {
        EXPECT_THROW(resolvePrecision(precisionSpecFromParams(p, false), cpuTarget()), std::invalid_argument);
        p["precision_policy"] = "fallback_fp32";
        std::string note;
        const PrecisionSpec s = resolvePrecision(precisionSpecFromParams(p, false), cpuTarget(), &note);
        EXPECT_EQ(s.autocast, AutocastDtype::none);
        EXPECT_TRUE(s.autocast_methods.empty());
        EXPECT_EQ(s.tf32, Tf32::off);
        EXPECT_FALSE(note.empty());
    }
    std::string note;
    resolvePrecision(precisionSpecFromParams(
                         {{"precision", "bf16"}, {"autocast_methods", "features"}, {"tf32", "on"},
                          {"precision_policy", "fallback_fp32"}},
                         false),
                     cpuTarget(), &note);
    EXPECT_EQ(note, "bf16;features+tf32");
}

TEST(ResolvePrecisionTest, CpuKeepsTheLegacySilentFallbacks) {
    // Plain autocast and half weights quietly run float32 off the GPU, as before 0.3.
    std::string note = "stale";
    const PrecisionSpec amp =
        resolvePrecision(precisionSpecFromParams({{"precision", "amp"}}, false), cpuTarget(), &note);
    EXPECT_EQ(amp.autocast, AutocastDtype::none);
    EXPECT_TRUE(note.empty());
    EXPECT_FALSE(resolvePrecision(precisionSpecFromParams({}, true), cpuTarget()).half_weights);
    // tf32=off is IEEE float32 on the CPU anyway: accepted, a no-op.
    const PrecisionSpec off = precisionSpecFromParams({{"tf32", "off"}}, false);
    EXPECT_EQ(resolvePrecision(off, cpuTarget()).tf32, Tf32::off);

    PrecisionTarget mps;
    mps.device = PrecisionTarget::Device::other;
    EXPECT_TRUE(resolvePrecision(precisionSpecFromParams({}, true), mps).half_weights);
    EXPECT_THROW(resolvePrecision(precisionSpecFromParams({{"precision", "bf16"}}, false), mps),
                 std::invalid_argument);
}

TEST(ResolvePrecisionTest, BakedBackendsRefuseAutocastMethodsAndReportBaked) {
    PrecisionTarget baked = cudaTarget(8, 6);
    baked.baked = true;
    const PrecisionSpec scoped =
        precisionSpecFromParams({{"precision", "bf16"}, {"autocast_methods", "features"}}, false);
    EXPECT_THROW(resolvePrecision(scoped, baked), std::invalid_argument);
    EXPECT_EQ(labelFor({{"precision", "bf16"}}, baked), "baked");
    EXPECT_EQ(labelFor({{"precision", "float32"}, {"tf32", "off"}, {"jit_optimize", "0"}}, baked), "baked");
    // Legacy requests keep their pre-0.3 labels.
    EXPECT_EQ(labelFor({}, baked, true), "float16");
    EXPECT_EQ(labelFor({{"precision", "float32"}}, baked), "float32");
    baked.device = PrecisionTarget::Device::cpu;
    EXPECT_EQ(labelFor({{"precision", "bf16"}}, baked), "baked");  // no device check: the artifact decides
}

// ---------------------------------------------------------------- 10. labels

TEST(PrecisionLabelTest, LegacyLabelsAreUnchanged) {
    EXPECT_EQ(labelFor({}, cudaTarget(8, 6), false), "float32");
    EXPECT_EQ(labelFor({}, cudaTarget(8, 6), true), "float16");
    EXPECT_EQ(labelFor({}, cpuTarget(), true), "float32");
    EXPECT_EQ(labelFor({{"precision", "autocast"}}, cudaTarget(8, 6)), "autocast(float16)");
    EXPECT_EQ(labelFor({{"precision", "autocast"}}, cpuTarget()), "float32");
    EXPECT_EQ(labelFor({{"precision", "float32"}}, cudaTarget(8, 6)), "float32");
}

TEST(PrecisionLabelTest, NnDepthModesMapToTheAgreedParamsAndLabels) {
    // stable-dpt sidecar `nuketorch_params` per runtime mode.
    const Params common = {{"tf32", "off"}, {"jit_optimize", "0"}, {"cudnn_benchmark", "0"}};
    const auto with = [&](Params extra) {
        Params p = common;
        for (auto& kv : extra) {
            p[kv.first] = kv.second;
        }
        return p;
    };
    const PrecisionTarget rtx = cudaTarget(12, 0);
    EXPECT_EQ(labelFor(with({{"precision", "float32"}}), rtx), "float32/ieee,jit0");
    EXPECT_EQ(labelFor(with({{"precision", "float32"}, {"tf32", "on"}}), rtx), "float32/tf32,jit0");
    EXPECT_EQ(labelFor(with({{"precision", "bf16"}}), rtx), "autocast(bfloat16),jit0");
    EXPECT_EQ(labelFor(with({{"precision", "bf16"}, {"autocast_methods", "features"}}), rtx),
              "autocast(bfloat16;features),jit0");
    EXPECT_EQ(labelFor(with({{"precision", "autocast"}, {"autocast_methods", "features"}}), rtx),
              "autocast(float16;features),jit0");

    const PrecisionSpec bf16_enc = resolvePrecision(
        precisionSpecFromParams(with({{"precision", "bf16"}, {"autocast_methods", "features"}}), false), rtx);
    EXPECT_EQ(bf16_enc.cudnn_benchmark, std::optional<bool>(false));
    EXPECT_TRUE(bf16_enc.autocastApplies("features"));
    EXPECT_FALSE(bf16_enc.autocastApplies("keyframe_memory"));
    EXPECT_FALSE(bf16_enc.autocastApplies("fuse"));
}

TEST(PrecisionLabelTest, OtherExplicitStates) {
    EXPECT_EQ(labelFor({{"precision", "tf32"}}, cudaTarget(8, 6)), "float32/tf32");
    EXPECT_EQ(labelFor({{"precision", "ieee"}}, cudaTarget(8, 6)), "float32/ieee");
    EXPECT_EQ(labelFor({{"precision", "bf16"}, {"tf32", "on"}}, cudaTarget(8, 6)), "autocast(bfloat16)/tf32");
    EXPECT_EQ(labelFor({{"precision", "half"}, {"jit_optimize", "0"}}, cudaTarget(8, 6)), "float16,jit0");
    // jit_optimize=1 is today's executor; only an explicit 0 is labelled.
    EXPECT_EQ(labelFor({{"precision", "float32"}, {"jit_optimize", "1"}}, cudaTarget(8, 6)), "float32");
}

TEST(PrecisionLabelTest, LabelsFitTheCApiMetricsStrings) {
    const Params long_lists[] = {
        {{"precision", "bf16"},
         {"autocast_methods", "encode_keyframes,features,keyframe_memory,fuse,forward"},
         {"tf32", "on"},
         {"jit_optimize", "0"}},
        {{"precision", "bf16"},
         {"autocast_methods", "encode_keyframes,features,keyframe_memory,fuse,forward"},
         {"tf32", "on"},
         {"jit_optimize", "0"},
         {"precision_policy", "fallback_fp32"}},
    };
    for (const Params& p : long_lists) {
        for (const PrecisionTarget& t : {cudaTarget(8, 6), cudaTarget(7, 5), cpuTarget()}) {
            try {
                const std::string label = labelFor(p, t);
                EXPECT_LT(label.size(), 64u) << label;
            } catch (const std::invalid_argument&) {
                // refused without fallback: no label
            }
        }
    }
    EXPECT_EQ(labelFor(long_lists[0], cudaTarget(8, 6)), "autocast(bfloat16;5 methods)/tf32,jit0");
    EXPECT_EQ(labelFor(long_lists[1], cpuTarget()), "float32/ieee(fallback:bf16;5 methods+tf32),jit0");
}

// ---------------------------------------------------------------- legacy setAutocast

namespace {

/// Builds and saves a minimal TorchScript module whose forward is a matmul —
/// an op on autocast's FP16 list.
std::string saveMatmulModule() {
    torch::jit::Module m("MatmulModule");
    m.define(R"JIT(
        def forward(self, x):
            return torch.mm(x, x)
    )JIT");
    const std::string path =
        "/tmp/nuketorch_autocast_test_" + std::to_string(getpid()) + ".pt";
    m.save(path);
    return path;
}

}  // namespace

TEST(BackendPrecisionTest, AutocastRunsMatmulInHalfOnCuda) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    const std::string path = saveMatmulModule();

    nuketorch::torch_worker::TorchScriptBackend backend;
    backend.load(path, torch::Device(torch::kCUDA, 0), torch::kFloat32);

    const auto x = torch::randn(
        {8, 8}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    backend.setAutocast(false);
    const auto full = backend.forward({x});
    ASSERT_EQ(full.size(), 1u);
    EXPECT_EQ(full[0].scalar_type(), torch::kFloat32);

    backend.setAutocast(true);
    const auto amp = backend.forward({x});
    ASSERT_EQ(amp.size(), 1u);
    EXPECT_EQ(amp[0].scalar_type(), torch::kFloat16);
    EXPECT_TRUE(torch::allclose(amp[0].to(torch::kFloat32), full[0], /*rtol=*/1e-2, /*atol=*/1e-2));

    // The region must not leak: disabling restores FP32 execution.
    backend.setAutocast(false);
    const auto full_again = backend.forward({x});
    EXPECT_EQ(full_again[0].scalar_type(), torch::kFloat32);

    std::remove(path.c_str());
}
