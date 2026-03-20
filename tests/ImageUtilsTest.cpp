#include <gtest/gtest.h>

#include <nuketorch/ImageUtils.h>

#include <vector>

TEST(ImageUtilsTest, VerticalFlipTwoRows) {
    // planar CHW, h=2, w=2, c=1
    std::vector<float> src = {
        // channel 0: row0 then row1
        1.0f,
        2.0f,
        3.0f,
        4.0f,
    };
    std::vector<float> dst(4, 0.0f);
    nuketorch::copyPlanarWithVerticalFlip(src.data(), dst.data(), 2, 2, 1);
    // bottom row of src becomes top of dst
    EXPECT_FLOAT_EQ(dst[0], 3.0f);
    EXPECT_FLOAT_EQ(dst[1], 4.0f);
    EXPECT_FLOAT_EQ(dst[2], 1.0f);
    EXPECT_FLOAT_EQ(dst[3], 2.0f);
}

TEST(ImageUtilsTest, HeightOneIsIdentity) {
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> dst(6, 0.0f);
    nuketorch::copyPlanarWithVerticalFlip(src.data(), dst.data(), 2, 1, 3);
    EXPECT_EQ(dst, src);
}
