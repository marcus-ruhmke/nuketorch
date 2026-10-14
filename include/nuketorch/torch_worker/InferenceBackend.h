#pragma once

#include <nuketorch/torch_worker/BackendParams.h>

#include <memory>
#include <string>
#include <vector>

#include <torch/script.h>

namespace nuketorch::torch_worker {

/// Pluggable inference runtime (TorchScript, AOTInductor `.pt2`, optional TensorRT `.engine`).
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    /// Loads @p model_path onto @p device. TorchScript converts weights to
    /// @p dtype and loads under the executor flag of the current precision
    /// (call setPrecision() first).
    virtual void load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) = 0;

    /// Runs the compiled model. One entry per output tensor (TorchScript may return a tuple).
    virtual std::vector<torch::Tensor> forward(const std::vector<torch::jit::IValue>& inputs) = 0;

    /// Runs method @p name under the current precision (autocast only when
    /// PrecisionSpec::autocastApplies(name)) and returns its raw result
    /// (tensor, tuple, list[int], list[Tensor], ...). An unknown method throws
    /// std::invalid_argument. AOTInductor/TensorRT run only "forward" and
    /// return a tensor (one output) or a tuple.
    virtual torch::jit::IValue runMethod(const std::string& name, std::vector<torch::jit::IValue> inputs);

    /// Same, with @p spec's autocast and TF32 for this call only; weights and
    /// jit_optimize stay as loaded. E.g. PrecisionSpec::ieeeFloat32() to
    /// recompute a non-finite result.
    virtual torch::jit::IValue runMethod(const std::string& name,
                                         std::vector<torch::jit::IValue> inputs,
                                         const PrecisionSpec& spec);

    /// True when runMethod(@p name, ...) can run.
    virtual bool hasMethod(const std::string& name) const { return name == "forward"; }

    virtual bool isLoaded() const = 0;
    virtual const std::string& loadedPath() const = 0;

    /// Precision for subsequent calls (and the executor flag of the next
    /// TorchScript load()). Only meaningful for backends that execute eager
    /// kernels (TorchScript); AOTInductor/TensorRT artifacts have their
    /// precision baked in.
    virtual void setPrecision(const PrecisionSpec& spec) { precision_ = spec; }
    const PrecisionSpec& precision() const { return precision_; }

    /// Deprecated: use setPrecision(). true = float16 autocast for every
    /// method, false = no autocast.
    void setAutocast(bool enabled) {
        PrecisionSpec spec = precision_;
        spec.autocast = enabled ? AutocastDtype::float16 : AutocastDtype::none;
        spec.autocast_methods.clear();
        setPrecision(spec);
    }

protected:
    PrecisionSpec precision_;
};

std::unique_ptr<InferenceBackend> createBackend(const std::string& canonical_backend_name);

}  // namespace nuketorch::torch_worker
