#pragma once

#include <cstddef>

namespace nuketorch {

/// Anonymous shared memory segment backed by `memfd_create` + `mmap`.
///
/// Segments have no name in any global namespace: they are shared by passing
/// the file descriptor over a Unix socket (SCM_RIGHTS), and the kernel reclaims
/// the memory when the last mapping and descriptor are gone — including after a
/// crash on either side. Descriptors are created close-on-exec.
class SharedMemoryBuffer {
public:
    /// Create a new segment of @p size bytes (creator side).
    static SharedMemoryBuffer create(size_t size);

    /// Map a segment received from the peer. Takes ownership of @p fd (closed on
    /// destruction, and immediately on failure). Throws BadRequestError if the
    /// segment is smaller than @p size.
    static SharedMemoryBuffer adopt(int fd, size_t size);

    ~SharedMemoryBuffer();

    SharedMemoryBuffer(const SharedMemoryBuffer&) = delete;
    SharedMemoryBuffer& operator=(const SharedMemoryBuffer&) = delete;

    SharedMemoryBuffer(SharedMemoryBuffer&& other) noexcept;
    SharedMemoryBuffer& operator=(SharedMemoryBuffer&& other) noexcept;

    /// Writable mapping; layout is owned by the caller (e.g. planar float CHW).
    void* data() const { return ptr_; }
    /// Size of the mapping in bytes.
    size_t size() const { return size_; }
    /// Underlying descriptor, e.g. for IPCServer::sendWithFds. Remains owned by this object.
    int fd() const { return fd_; }

private:
    SharedMemoryBuffer(int fd, size_t size);

    size_t size_;
    void* ptr_;
    int fd_;

    void cleanup();
};

}  // namespace nuketorch
