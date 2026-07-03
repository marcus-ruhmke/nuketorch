#include <nuketorch/SharedMemoryBuffer.h>

#include <nuketorch/Errors.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace nuketorch {

SharedMemoryBuffer SharedMemoryBuffer::create(size_t size) {
    if (size == 0) {
        throw Error(ErrorCode::invalid_argument, "shared memory size must be > 0");
    }
    const int fd = memfd_create("nuketorch-frame", MFD_CLOEXEC);
    if (fd == -1) {
        throw Error(ErrorCode::internal,
                    std::string("memfd_create failed: ") + std::strerror(errno));
    }
    if (ftruncate(fd, static_cast<off_t>(size)) == -1) {
        const int err = errno;
        ::close(fd);
        throw Error(ErrorCode::internal,
                    std::string("ftruncate failed: ") + std::strerror(err));
    }
    return SharedMemoryBuffer(fd, size);
}

SharedMemoryBuffer SharedMemoryBuffer::adopt(int fd, size_t size) {
    if (fd < 0 || size == 0) {
        throw Error(ErrorCode::invalid_argument, "invalid fd or size for shared memory adopt");
    }
    struct stat st{};
    if (fstat(fd, &st) == -1) {
        const int err = errno;
        ::close(fd);
        throw Error(ErrorCode::internal, std::string("fstat failed: ") + std::strerror(err));
    }
    if (st.st_size < 0 || static_cast<size_t>(st.st_size) < size) {
        ::close(fd);
        throw BadRequestError("shared memory segment smaller than requested mapping");
    }
    return SharedMemoryBuffer(fd, size);
}

SharedMemoryBuffer::SharedMemoryBuffer(int fd, size_t size)
    : size_(size), ptr_(MAP_FAILED), fd_(fd) {
    ptr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (ptr_ == MAP_FAILED) {
        const int err = errno;
        ::close(fd_);
        fd_ = -1;
        throw Error(ErrorCode::internal, std::string("mmap failed: ") + std::strerror(err));
    }
}

SharedMemoryBuffer::~SharedMemoryBuffer() {
    cleanup();
}

SharedMemoryBuffer::SharedMemoryBuffer(SharedMemoryBuffer&& other) noexcept
    : size_(other.size_), ptr_(other.ptr_), fd_(other.fd_) {
    other.ptr_ = MAP_FAILED;
    other.fd_ = -1;
}

SharedMemoryBuffer& SharedMemoryBuffer::operator=(SharedMemoryBuffer&& other) noexcept {
    if (this != &other) {
        cleanup();
        size_ = other.size_;
        ptr_ = other.ptr_;
        fd_ = other.fd_;

        other.ptr_ = MAP_FAILED;
        other.fd_ = -1;
    }
    return *this;
}

void SharedMemoryBuffer::cleanup() {
    if (ptr_ != MAP_FAILED) {
        munmap(ptr_, size_);
        ptr_ = MAP_FAILED;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

}  // namespace nuketorch
