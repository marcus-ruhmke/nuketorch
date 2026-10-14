#pragma once

#include <cstdint>
#include <string>

namespace nuketorch {

/// Profiling and metadata for one inference frame. Host-only fields are filled by `InferenceClient`;
/// worker fields are serialized over the wire after the `"OK"` response prefix.
struct InferenceMetrics {
    double backend_forward_ms = -1;
    double gpu_compute_ms = -1;
    double tensor_prep_ms = -1;
    double output_copy_ms = -1;
    double model_load_ms = -1;

    std::string backend;
    std::string device;
    std::string dtype;
    int64_t peak_gpu_memory_bytes = -1;

    /// Filled by the host (`InferenceClient`); not sent on the wire.
    double shm_write_ms = -1;
    double round_trip_ms = -1;
    double shm_read_ms = -1;
    double total_ms = -1;

    /// Worker remarks for this frame (e.g. a precision fallback, a recomputed
    /// non-finite result); empty when there is nothing to say. Sent on the
    /// wire only when non-empty; not part of the C API metrics struct.
    std::string notes;
};

/// Serializes worker-side fields only (binary blob, no `"OK"` prefix).
std::string serializeMetrics(const InferenceMetrics& m);

/// Parses a metrics blob produced by `serializeMetrics`. Host-side fields remain at defaults (-1 / empty).
bool parseMetricsPayload(const std::string& payload, InferenceMetrics& out, std::string& error);

/// Parses a full inference reply: must start with `"OK"`. If only `"OK"`, metrics default. If `"OK"` + blob, parses worker metrics.
bool parseInferenceOkResponse(const std::string& response, InferenceMetrics& out, std::string& error);

}  // namespace nuketorch
