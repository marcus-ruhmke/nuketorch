#include <nuketorch/torch_worker/TensorRTBackend.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace nuketorch::torch_worker {

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING && msg != nullptr) {
            std::cerr << "[TensorRTBackend] " << msg << std::endl;
        }
    }
};

static TrtLogger g_trt_logger;

torch::ScalarType trtDataTypeToTorch(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT:
            return torch::kFloat32;
        case nvinfer1::DataType::kHALF:
            return torch::kFloat16;
        case nvinfer1::DataType::kINT32:
            return torch::kInt32;
        case nvinfer1::DataType::kINT8:
            return torch::kInt8;
        case nvinfer1::DataType::kBOOL:
            return torch::kBool;
        case nvinfer1::DataType::kINT64:
            return torch::kInt64;
        default:
            throw std::runtime_error("TensorRTBackend: unsupported engine tensor DataType");
    }
}

std::vector<int64_t> dimsToSizes(const nvinfer1::Dims& d) {
    std::vector<int64_t> out;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) {
            throw std::runtime_error("TensorRTBackend: dynamic tensor shapes are not supported yet");
        }
        out.push_back(static_cast<int64_t>(d.d[i]));
    }
    return out;
}

bool sizesMatch(const torch::Tensor& t, const std::vector<int64_t>& expected) {
    if (static_cast<size_t>(t.dim()) != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (t.size(static_cast<int64_t>(i)) != expected[i]) {
            return false;
        }
    }
    return true;
}

std::vector<torch::Tensor> iValuesToTensors(const std::vector<torch::jit::IValue>& inputs, torch::Device device) {
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
            tensors.push_back(torch::tensor(
                {iv.toDouble()},
                torch::TensorOptions().device(ref_device).dtype(torch::kFloat64)));
        } else if (iv.isInt()) {
            tensors.push_back(torch::tensor(
                {iv.toInt()},
                torch::TensorOptions().device(ref_device).dtype(torch::kInt64)));
        } else if (iv.isBool()) {
            tensors.push_back(torch::tensor(
                {iv.toBool()},
                torch::TensorOptions().device(ref_device).dtype(torch::kBool)));
        } else {
            throw std::runtime_error("TensorRTBackend: unsupported IValue input for conversion to tensor");
        }
    }
    return tensors;
}

}  // namespace

void TensorRTBackend::releaseTensorRtResources() noexcept {
    if (stream_) {
        if (device_.is_cuda()) {
            cudaSetDevice(device_.has_index() ? static_cast<int>(device_.index()) : 0);
        }
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    if (context_) {
        delete context_;
        context_ = nullptr;
    }
    if (engine_) {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_) {
        delete runtime_;
        runtime_ = nullptr;
    }
    input_names_.clear();
    output_names_.clear();
    path_.clear();
    device_ = torch::kCPU;
}

TensorRTBackend::~TensorRTBackend() {
    releaseTensorRtResources();
}

void TensorRTBackend::load(const std::string& model_path, torch::Device device, torch::ScalarType dtype) {
    (void)dtype;
    if (!device.is_cuda()) {
        throw std::runtime_error("TensorRTBackend: CUDA device required");
    }

    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("TensorRTBackend: cannot open engine file: " + model_path);
    }
    const std::streamsize file_size = file.tellg();
    if (file_size <= 0) {
        throw std::runtime_error("TensorRTBackend: engine file is empty: " + model_path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(file_size));
    if (!file.read(buffer.data(), file_size)) {
        throw std::runtime_error("TensorRTBackend: failed to read engine file: " + model_path);
    }
    file.close();

    releaseTensorRtResources();

    const int dev_index = device.has_index() ? static_cast<int>(device.index()) : 0;
    cudaSetDevice(dev_index);

    runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runtime_) {
        throw std::runtime_error("TensorRTBackend: createInferRuntime failed");
    }
    engine_ = runtime_->deserializeCudaEngine(buffer.data(), buffer.size());
    if (!engine_) {
        throw std::runtime_error("TensorRTBackend: deserializeCudaEngine failed (invalid or corrupt engine)");
    }
    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("TensorRTBackend: createExecutionContext failed");
    }

    input_names_.clear();
    output_names_.clear();
    const int32_t nb = engine_->getNbIOTensors();
    for (int32_t i = 0; i < nb; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (name == nullptr) {
            continue;
        }
        const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_names_.emplace_back(name);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_names_.emplace_back(name);
        }
    }

    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        throw std::runtime_error("TensorRTBackend: cudaStreamCreate failed");
    }

    path_ = model_path;
    device_ = device;
}

std::vector<torch::Tensor> TensorRTBackend::forward(const std::vector<torch::jit::IValue>& inputs) {
    if (!engine_ || !context_ || !stream_) {
        throw std::runtime_error("TensorRTBackend: not loaded");
    }
    if (!device_.is_cuda()) {
        throw std::runtime_error("TensorRTBackend: CUDA device required");
    }
    const int dev_index = device_.has_index() ? static_cast<int>(device_.index()) : 0;
    cudaSetDevice(dev_index);

    if (inputs.size() != input_names_.size()) {
        std::ostringstream oss;
        oss << "TensorRTBackend: expected " << input_names_.size() << " inputs, got " << inputs.size();
        throw std::runtime_error(oss.str());
    }

    std::vector<torch::Tensor> in_tensors = iValuesToTensors(inputs, device_);
    for (size_t i = 0; i < in_tensors.size(); ++i) {
        in_tensors[i] = in_tensors[i].to(device_, /*non_blocking=*/true).contiguous();
        const char* name = input_names_[i].c_str();
        const nvinfer1::Dims d = engine_->getTensorShape(name);
        const std::vector<int64_t> expected = dimsToSizes(d);
        if (!sizesMatch(in_tensors[i], expected)) {
            throw std::runtime_error("TensorRTBackend: input tensor shape mismatch for " + input_names_[i]);
        }
        const nvinfer1::DataType trt_dt = engine_->getTensorDataType(name);
        const torch::ScalarType want = trtDataTypeToTorch(trt_dt);
        if (in_tensors[i].scalar_type() != want) {
            in_tensors[i] = in_tensors[i].to(want, /*non_blocking=*/true);
        }
        if (!context_->setTensorAddress(name, in_tensors[i].data_ptr())) {
            throw std::runtime_error("TensorRTBackend: setTensorAddress failed for input " + input_names_[i]);
        }
    }

    std::vector<torch::Tensor> outputs;
    outputs.reserve(output_names_.size());
    for (const std::string& out_name : output_names_) {
        const nvinfer1::Dims d = engine_->getTensorShape(out_name.c_str());
        const std::vector<int64_t> sizes = dimsToSizes(d);
        const nvinfer1::DataType trt_dt = engine_->getTensorDataType(out_name.c_str());
        const torch::ScalarType st = trtDataTypeToTorch(trt_dt);
        torch::Tensor t = torch::empty(sizes, torch::TensorOptions().device(device_).dtype(st));
        if (!context_->setTensorAddress(out_name.c_str(), t.data_ptr())) {
            throw std::runtime_error("TensorRTBackend: setTensorAddress failed for output " + out_name);
        }
        outputs.push_back(std::move(t));
    }

    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRTBackend: enqueueV3 failed");
    }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        throw std::runtime_error("TensorRTBackend: cudaStreamSynchronize failed");
    }

    return outputs;
}

bool TensorRTBackend::isLoaded() const {
    return engine_ != nullptr && context_ != nullptr;
}

const std::string& TensorRTBackend::loadedPath() const {
    return path_;
}

}  // namespace nuketorch::torch_worker
