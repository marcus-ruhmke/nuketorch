#include <nuketorch/nuketorch_c.h>

#include <nuketorch/Errors.h>
#include <nuketorch/InferenceClient.h>

#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void copyMetricsToC(const nuketorch::InferenceMetrics& src, nuketorch_inference_metrics* dst) {
    if (!dst) {
        return;
    }
    dst->backend_forward_ms = src.backend_forward_ms;
    dst->gpu_compute_ms = src.gpu_compute_ms;
    dst->tensor_prep_ms = src.tensor_prep_ms;
    dst->output_copy_ms = src.output_copy_ms;
    dst->model_load_ms = src.model_load_ms;
    dst->peak_gpu_memory_bytes = src.peak_gpu_memory_bytes;
    dst->shm_write_ms = src.shm_write_ms;
    dst->round_trip_ms = src.round_trip_ms;
    dst->shm_read_ms = src.shm_read_ms;
    dst->total_ms = src.total_ms;

    std::strncpy(dst->backend, src.backend.c_str(), sizeof(dst->backend) - 1);
    dst->backend[sizeof(dst->backend) - 1] = '\0';
    std::strncpy(dst->device, src.device.c_str(), sizeof(dst->device) - 1);
    dst->device[sizeof(dst->device) - 1] = '\0';
    std::strncpy(dst->dtype, src.dtype.c_str(), sizeof(dst->dtype) - 1);
    dst->dtype[sizeof(dst->dtype) - 1] = '\0';
}

struct NuketorchClientOpaque {
    std::unique_ptr<nuketorch::InferenceClient> client;
    std::string last_error;
    nuketorch_error_code last_error_code = NUKETORCH_ERRC_OK;
};

void setError(NuketorchClientOpaque* w, const char* msg,
              nuketorch_error_code code = NUKETORCH_ERRC_INTERNAL) {
    if (w) {
        w->last_error = msg ? msg : "";
        w->last_error_code = code;
    }
}

void setErrorFromException(NuketorchClientOpaque* w, const std::exception& e) {
    nuketorch_error_code code = NUKETORCH_ERRC_INTERNAL;
    if (const auto* typed = dynamic_cast<const nuketorch::Error*>(&e)) {
        code = static_cast<nuketorch_error_code>(typed->code());
    }
    setError(w, e.what(), code);
}

}  // namespace

extern "C" {

nuketorch_client_t nuketorch_client_create(const char* worker_binary,
                                           const char* socket_path,
                                           int num_inputs) {
    if (!worker_binary || !socket_path) {
        return nullptr;
    }
    try {
        auto* w = new NuketorchClientOpaque();
        w->client = std::make_unique<nuketorch::InferenceClient>(
            std::string(worker_binary), std::string(socket_path), num_inputs);
        return reinterpret_cast<nuketorch_client_t>(w);
    } catch (const std::exception&) {
        return nullptr;
    }
}

void nuketorch_client_destroy(nuketorch_client_t client) {
    if (!client) {
        return;
    }
    delete reinterpret_cast<NuketorchClientOpaque*>(client);
}

const char* nuketorch_client_last_error(nuketorch_client_t client) {
    if (!client) {
        return "";
    }
    auto* w = reinterpret_cast<NuketorchClientOpaque*>(client);
    return w->last_error.c_str();
}

nuketorch_error_code nuketorch_client_last_error_code(nuketorch_client_t client) {
    if (!client) {
        return NUKETORCH_ERRC_INVALID_ARGUMENT;
    }
    auto* w = reinterpret_cast<NuketorchClientOpaque*>(client);
    return w->last_error_code;
}

int nuketorch_client_start(nuketorch_client_t client) {
    if (!client) {
        return -1;
    }
    auto* w = reinterpret_cast<NuketorchClientOpaque*>(client);
    w->last_error.clear();
    w->last_error_code = NUKETORCH_ERRC_OK;
    try {
        w->client->start();
        return 0;
    } catch (const std::exception& e) {
        setErrorFromException(w, e);
        return -1;
    }
}

int nuketorch_client_stop(nuketorch_client_t client) {
    if (!client) {
        return -1;
    }
    auto* w = reinterpret_cast<NuketorchClientOpaque*>(client);
    w->last_error.clear();
    w->last_error_code = NUKETORCH_ERRC_OK;
    try {
        w->client->stop();
        return 0;
    } catch (const std::exception& e) {
        setErrorFromException(w, e);
        return -1;
    }
}

int nuketorch_client_abort(nuketorch_client_t client) {
    if (!client) {
        return -1;
    }
    auto* w = reinterpret_cast<NuketorchClientOpaque*>(client);
    w->last_error.clear();
    w->last_error_code = NUKETORCH_ERRC_OK;
    try {
        w->client->abort();
        return 0;
    } catch (const std::exception& e) {
        setErrorFromException(w, e);
        return -1;
    }
}

int nuketorch_client_ping(nuketorch_client_t client) {
    if (!client) {
        return -1;
    }
    auto* w = reinterpret_cast<NuketorchClientOpaque*>(client);
    w->last_error.clear();
    w->last_error_code = NUKETORCH_ERRC_OK;
    try {
        return w->client->ping() ? 0 : -1;
    } catch (const std::exception& e) {
        setErrorFromException(w, e);
        return -1;
    }
}

int nuketorch_client_get_gpu_info(nuketorch_client_t client, char* buf, size_t buf_size) {
    if (!client || !buf || buf_size == 0) {
        return -1;
    }
    auto* w = reinterpret_cast<NuketorchClientOpaque*>(client);
    w->last_error.clear();
    w->last_error_code = NUKETORCH_ERRC_OK;
    try {
        const std::string s = w->client->getGpuInfo();
        std::strncpy(buf, s.c_str(), buf_size - 1);
        buf[buf_size - 1] = '\0';
        return 0;
    } catch (const std::exception& e) {
        setErrorFromException(w, e);
        return -1;
    }
}

int nuketorch_client_process_frame(nuketorch_client_t client,
                                   const nuketorch_frame_buffers* buffers,
                                   const nuketorch_inference_config* config,
                                   nuketorch_abort_fn abort_fn,
                                   void* abort_user_data,
                                   nuketorch_inference_metrics* metrics) {
    if (!client || !buffers || !config) {
        if (client) {
            setError(reinterpret_cast<NuketorchClientOpaque*>(client), "null argument",
                     NUKETORCH_ERRC_INVALID_ARGUMENT);
        }
        return -1;
    }
    auto* w = reinterpret_cast<NuketorchClientOpaque*>(client);
    w->last_error.clear();
    w->last_error_code = NUKETORCH_ERRC_OK;

    nuketorch::FrameBuffers fb;
    if (buffers->num_inputs < 1 || !buffers->inputs) {
        setError(w, "invalid frame buffers", NUKETORCH_ERRC_INVALID_ARGUMENT);
        return -1;
    }
    fb.inputs.assign(buffers->inputs, buffers->inputs + buffers->num_inputs);
    fb.output = buffers->output;
    fb.width = buffers->width;
    fb.height = buffers->height;
    fb.channels = buffers->channels;

    nuketorch::InferenceConfig cfg;
    if (config->model_path) {
        cfg.model_path = config->model_path;
    }
    cfg.use_gpu = config->use_gpu != 0;
    cfg.mixed_precision = config->mixed_precision != 0;
    cfg.debug = config->debug != 0;
    cfg.frame_timeout_ms = config->frame_timeout_ms;
    if (config->params && config->num_params > 0) {
        for (int i = 0; i < config->num_params; ++i) {
            const nuketorch_param& p = config->params[i];
            if (p.key && p.value) {
                cfg.params[p.key] = p.value;
            }
        }
    }

    std::function<bool()> is_aborted;
    if (abort_fn) {
        is_aborted = [abort_fn, abort_user_data]() {
            return abort_fn(abort_user_data) != 0;
        };
    }

    nuketorch::InferenceMetrics cpp_metrics;
    nuketorch::InferenceMetrics* metrics_ptr = metrics ? &cpp_metrics : nullptr;

    try {
        w->client->processFrame(fb, cfg, is_aborted, metrics_ptr);
        if (metrics) {
            copyMetricsToC(cpp_metrics, metrics);
        }
        return 0;
    } catch (const std::exception& e) {
        setErrorFromException(w, e);
        return -1;
    }
}

}  // extern "C"
