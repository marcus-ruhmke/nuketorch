#include <nuketorch/SharedMemoryBuffer.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace nuketorch {

SharedMemoryBuffer::SharedMemoryBuffer(const std::string& name, size_t size, bool create)
    : name_(name), size_(size), ptr_(MAP_FAILED), fd_(-1), is_creator_(create) {
    if (create) {
        shm_unlink(name_.c_str());

        fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
        if (fd_ == -1) {
            throw std::runtime_error("shm_open failed for " + name_ + ": " + std::strerror(errno));
        }

        if (ftruncate(fd_, size_) == -1) {
            cleanup();
            throw std::runtime_error("ftruncate failed for " + name_ + ": " + std::strerror(errno));
        }
    } else {
        fd_ = shm_open(name_.c_str(), O_RDWR, 0666);
        if (fd_ == -1) {
            throw std::runtime_error("shm_open (read) failed for " + name_ + ": " + std::strerror(errno));
        }
    }

    ptr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (ptr_ == MAP_FAILED) {
        cleanup();
        throw std::runtime_error("mmap failed for " + name_ + ": " + std::strerror(errno));
    }
}

SharedMemoryBuffer::~SharedMemoryBuffer() {
    cleanup();
}

SharedMemoryBuffer::SharedMemoryBuffer(SharedMemoryBuffer&& other) noexcept
    : name_(std::move(other.name_)),
      size_(other.size_),
      ptr_(other.ptr_),
      fd_(other.fd_),
      is_creator_(other.is_creator_) {
    other.ptr_ = MAP_FAILED;
    other.fd_ = -1;
    other.is_creator_ = false;
}

SharedMemoryBuffer& SharedMemoryBuffer::operator=(SharedMemoryBuffer&& other) noexcept {
    if (this != &other) {
        cleanup();
        name_ = std::move(other.name_);
        size_ = other.size_;
        ptr_ = other.ptr_;
        fd_ = other.fd_;
        is_creator_ = other.is_creator_;

        other.ptr_ = MAP_FAILED;
        other.fd_ = -1;
        other.is_creator_ = false;
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
    if (is_creator_) {
        shm_unlink(name_.c_str());
        is_creator_ = false;
    }
}

}  // namespace nuketorch
