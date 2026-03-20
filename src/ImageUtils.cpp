#include <nuketorch/ImageUtils.h>

#include <cstring>

namespace nuketorch {

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
