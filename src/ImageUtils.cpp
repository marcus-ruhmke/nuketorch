#include <nuketorch/ImageUtils.h>

#include <cstdint>
#include <cstring>
#include <limits>

namespace nuketorch {

bool computeFrameBytes(int width, int height, int channels, size_t& bytes_out) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        return false;
    }
    const uint64_t wh = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    if (wh > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(channels)) {
        return false;
    }
    const uint64_t pixels = wh * static_cast<uint64_t>(channels);
    if (pixels > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    bytes_out = static_cast<size_t>(pixels) * sizeof(float);
    return true;
}

void copyPlanarWithVerticalFlip(const float* src, float* dst, int width, int height, int channels) {
    const size_t row_bytes = static_cast<size_t>(width) * sizeof(float);
    const size_t channel_elements = static_cast<size_t>(width) * height;

    for (int c = 0; c < channels; ++c) {
        const float* src_channel = src + c * channel_elements;
        float* dst_channel = dst + c * channel_elements;

        for (int y = 0; y < height; ++y) {
            const float* src_row = src_channel + y * width;
            float* dst_row = dst_channel + (height - 1 - y) * width;
            std::memcpy(dst_row, src_row, row_bytes);
        }
    }
}

}  // namespace nuketorch
