#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace nuketorch {

/// POSIX shared memory object (`shm_open` + `mmap`). The creator (`create == true`) unlinks the object in the destructor.
class SharedMemoryBuffer {
public:
    /// @param name POSIX shared memory name (typically starts with `/`).
    /// @param size Mapping size in bytes.
    /// @param create If true, create/exclusive open and truncate; if false, open existing segment for read/write.
    SharedMemoryBuffer(const std::string& name, size_t size, bool create = true);
    ~SharedMemoryBuffer();

    SharedMemoryBuffer(const SharedMemoryBuffer&) = delete;
    SharedMemoryBuffer& operator=(const SharedMemoryBuffer&) = delete;

    SharedMemoryBuffer(SharedMemoryBuffer&& other) noexcept;
    SharedMemoryBuffer& operator=(SharedMemoryBuffer&& other) noexcept;

    /// Writable mapping; layout is owned by the caller (e.g. planar float CHW).
    void* data() const { return ptr_; }
    /// Size of the mapping in bytes.
    size_t size() const { return size_; }
    /// Same name passed to the constructor.
    const std::string& name() const { return name_; }

private:
    std::string name_;
    size_t size_;
    void* ptr_;
    int fd_;
    bool is_creator_;

    void cleanup();
};

}  // namespace nuketorch
