#include <nuketorch/Protocol.h>

#include <cstddef>
#include <cstdint>
#include <string>

// Fuzzes the inference-request wire parser: must never crash or hang on
// arbitrary bytes, only return false with an error string.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    nuketorch::InferenceRequest req;
    std::string error;
    (void)nuketorch::deserialize(std::string(reinterpret_cast<const char*>(data), size), req,
                                 error);
    return 0;
}
