#include <gtest/gtest.h>

#include <nuketorch/Errors.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <unistd.h>

#include <utility>

TEST(SharedMemoryTest, CanWriteAndReadFloats) {
    const size_t size = 1920 * 1080 * 3 * sizeof(float);

    auto buf = nuketorch::SharedMemoryBuffer::create(size);
    float* data = static_cast<float*>(buf.data());
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(buf.size(), size);
    EXPECT_GE(buf.fd(), 0);

    data[0] = 42.0f;
    data[100] = 99.0f;

    EXPECT_EQ(data[0], 42.0f);
    EXPECT_EQ(data[100], 99.0f);
}

TEST(SharedMemoryTest, AdoptSharesTheSameMemory) {
    auto creator = nuketorch::SharedMemoryBuffer::create(64);
    static_cast<float*>(creator.data())[3] = 7.0f;

    // Simulate the fd arriving on the other side of a socket.
    const int duplicate = dup(creator.fd());
    ASSERT_GE(duplicate, 0);
    auto adopted = nuketorch::SharedMemoryBuffer::adopt(duplicate, 64);

    EXPECT_FLOAT_EQ(static_cast<const float*>(adopted.data())[3], 7.0f);

    static_cast<float*>(adopted.data())[3] = 8.0f;
    EXPECT_FLOAT_EQ(static_cast<const float*>(creator.data())[3], 8.0f);
}

TEST(SharedMemoryTest, AdoptRejectsUndersizedSegment) {
    auto creator = nuketorch::SharedMemoryBuffer::create(16);
    const int duplicate = dup(creator.fd());
    ASSERT_GE(duplicate, 0);
    // adopt() takes ownership of the fd even on failure.
    EXPECT_THROW(nuketorch::SharedMemoryBuffer::adopt(duplicate, 1024), nuketorch::BadRequestError);
}

TEST(SharedMemoryTest, CreateRejectsZeroSize) {
    EXPECT_THROW(nuketorch::SharedMemoryBuffer::create(0), nuketorch::Error);
}

TEST(SharedMemoryTest, MoveTransfersOwnership) {
    auto a = nuketorch::SharedMemoryBuffer::create(32);
    static_cast<float*>(a.data())[0] = 5.0f;
    const int fd_before = a.fd();

    nuketorch::SharedMemoryBuffer b = std::move(a);
    EXPECT_EQ(b.fd(), fd_before);
    EXPECT_FLOAT_EQ(static_cast<const float*>(b.data())[0], 5.0f);
}
