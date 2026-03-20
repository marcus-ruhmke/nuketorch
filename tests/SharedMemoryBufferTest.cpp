#include <gtest/gtest.h>

#include <nuketorch/SharedMemoryBuffer.h>

#include <unistd.h>

#include <string>

TEST(SharedMemoryTest, CanWriteAndReadFloats) {
    const std::string shm_name = "/nuketorch_test_shm_" + std::to_string(getpid());
    const size_t size = 1920 * 1080 * 3 * sizeof(float);

    {
        nuketorch::SharedMemoryBuffer buf(shm_name, size, true);
        float* data = static_cast<float*>(buf.data());
        ASSERT_NE(data, nullptr);

        data[0] = 42.0f;
        data[100] = 99.0f;

        EXPECT_EQ(data[0], 42.0f);
        EXPECT_EQ(data[100], 99.0f);

        nuketorch::SharedMemoryBuffer reader(shm_name, size, false);
        const float* rdata = static_cast<const float*>(reader.data());
        ASSERT_NE(rdata, nullptr);
        EXPECT_EQ(rdata[0], 42.0f);
        EXPECT_EQ(rdata[100], 99.0f);
    }
}
