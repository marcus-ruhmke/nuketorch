#pragma once

namespace nuketorch {

/// Copy planar float image (channels * height * width) with vertical flip (Nuke bottom-up to top-down worker layout).
void copyPlanarWithVerticalFlip(const float* src, float* dst, int width, int height, int channels);

}  // namespace nuketorch
