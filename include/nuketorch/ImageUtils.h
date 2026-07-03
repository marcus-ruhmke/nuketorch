#pragma once

#include <cstddef>

namespace nuketorch {

/// Copy planar float image (channels * height * width) with vertical flip (Nuke bottom-up to top-down worker layout).
void copyPlanarWithVerticalFlip(const float* src, float* dst, int width, int height, int channels);

/// Computes `width * height * channels * sizeof(float)` with overflow checking.
/// Returns false (leaving @p bytes_out untouched) when any dimension is <= 0 or
/// the product does not fit in size_t. Both sides of the wire use this so a
/// hostile or corrupt request can never size a buffer smaller than the frame.
bool computeFrameBytes(int width, int height, int channels, size_t& bytes_out);

}  // namespace nuketorch
