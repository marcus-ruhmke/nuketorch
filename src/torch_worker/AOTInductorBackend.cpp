#include <nuketorch/torch_worker/AOTInductorBackend.h>

#include <stdexcept>

namespace nuketorch::torch_worker {

namespace {

std::vector<torch::Tensor> iValuesToTensors(const std::vector<torch::jit::IValue>& inputs,
                                            torch::Device device) {
    torch::Device ref_device = device;
    for (const auto& iv : inputs) {
        if (iv.isTensor()) {
            ref_device = iv.toTensor().device();
            break;
        }
    }
    std::vector<torch::Tensor> tensors;
    tensors.reserve(inputs.size());
    for (const auto& iv : inputs) {
        if (iv.isTensor()) {
            tensors.push_back(iv.toTensor());
        } else if (iv.isDouble()) {
            tensors.push_back(torch::tensor({iv.toDouble()}, torch::TensorOptions().device(ref_device).dtype(torch::kFloat64)));
        } else if (iv.isInt()) {
            tensors.push_back(torch::tensor({iv.toInt()}, torch::TensorOptions().device(ref_device).dtype(torch::kInt64)));
        } else if (iv.isBool()) {
            tensors.push_back(torch::tensor({iv.toBool()}, torch::TensorOptions().device(ref_device).dtype(torch::kBool)));
        } else {
            throw std::runtime_error("AOTInductorBackend: unsupported IValue input for conversion to tensor");
        }
    }
    return tensors;
}

}  // namespace

AOTInductorBackend::~AOTInductorBackend() = default;

void AOTInductorBackend::load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) {
    (void)dtype;
    c10::DeviceIndex idx = -1;
    if (device.is_cuda()) {
        idx = device.has_index() ? device.index() : 0;
    }
    loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        model_path, /*model_name=*/"model", /*run_single_threaded=*/false,
        /*num_runners=*/1, /*device_index=*/idx);
    path_ = model_path;
    device_ = device;
    dtype_ = dtype;
}

std::vector<torch::Tensor> AOTInductorBackend::forward(const std::vector<torch::jit::IValue>& inputs) {
    if (!loader_) {
        throw std::runtime_error("AOTInductorBackend: not loaded");
    }
    torch::InferenceMode guard;
    std::vector<torch::Tensor> tensors = iValuesToTensors(inputs, device_);
    std::vector<torch::Tensor> outputs = loader_->run(tensors);
    if (outputs.empty()) {
        throw std::runtime_error("AOTInductorBackend: empty output from package");
    }
    return outputs;
}

bool AOTInductorBackend::isLoaded() const {
    return loader_ != nullptr;
}

const std::string& AOTInductorBackend::loadedPath() const {
    return path_;
}

}  // namespace nuketorch::torch_worker
