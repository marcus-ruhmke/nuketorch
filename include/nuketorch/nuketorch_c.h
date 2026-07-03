#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to host-side inference client (fork/exec worker + IPC + SHM). */
typedef struct nuketorch_client_opaque* nuketorch_client_t;

/** Maximum length for backend/device/dtype strings in metrics (including NUL). */
#define NUKETORCH_METRICS_STRING_MAX 64

#define NUKETORCH_BACKEND_TORCHSCRIPT "torchscript"
#define NUKETORCH_BACKEND_AOTINDUCTOR "aotinductor"
#define NUKETORCH_BACKEND_TENSORRT "tensorrt"

struct nuketorch_frame_buffers {
    const float* const* inputs;
    int num_inputs;
    float* output;
    int width;
    int height;
    int channels;
};

struct nuketorch_param {
    const char* key;
    const char* value;
};

struct nuketorch_inference_config {
    const char* model_path;
    int use_gpu;
    int mixed_precision;
    int debug;
    const struct nuketorch_param* params;
    int num_params;
};

struct nuketorch_inference_metrics {
    double backend_forward_ms;
    double gpu_compute_ms;
    double tensor_prep_ms;
    double output_copy_ms;
    double model_load_ms;
    char backend[NUKETORCH_METRICS_STRING_MAX];
    char device[NUKETORCH_METRICS_STRING_MAX];
    char dtype[NUKETORCH_METRICS_STRING_MAX];
    int64_t peak_gpu_memory_bytes;
    double shm_write_ms;
    double round_trip_ms;
    double shm_read_ms;
    double total_ms;
};

/** Return non-zero if the user requested cancellation (maps to C++ abort predicate). */
typedef int (*nuketorch_abort_fn)(void* user_data);

nuketorch_client_t nuketorch_client_create(const char* worker_binary,
                                           const char* socket_path,
                                           int num_inputs);

void nuketorch_client_destroy(nuketorch_client_t client);

int nuketorch_client_start(nuketorch_client_t client);
int nuketorch_client_stop(nuketorch_client_t client);
int nuketorch_client_abort(nuketorch_client_t client);
int nuketorch_client_ping(nuketorch_client_t client);

int nuketorch_client_get_gpu_info(nuketorch_client_t client, char* buf, size_t buf_size);

int nuketorch_client_process_frame(nuketorch_client_t client,
                                   const struct nuketorch_frame_buffers* buffers,
                                   const struct nuketorch_inference_config* config,
                                   nuketorch_abort_fn abort_fn,
                                   void* abort_user_data,
                                   struct nuketorch_inference_metrics* metrics);

const char* nuketorch_client_last_error(nuketorch_client_t client);

#ifdef __cplusplus
}
#endif
