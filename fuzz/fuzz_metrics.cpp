#include <nuketorch/InferenceMetrics.h>

#include <cstddef>
#include <cstdint>
#include <string>

// Fuzzes the metrics blob parser and the OK-response wrapper.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    const std::string payload(reinterpret_cast<const char*>(data), size);
    nuketorch::InferenceMetrics metrics;
    std::string error;
    (void)nuketorch::parseMetricsPayload(payload, metrics, error);
    (void)nuketorch::parseInferenceOkResponse(payload, metrics, error);
    return 0;
}
