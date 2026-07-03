#pragma once

#include <stdexcept>
#include <string>

namespace nuketorch {

/// Stable error categories shared by the C++ exceptions, the wire protocol, and the C API.
enum class ErrorCode : int {
    ok = 0,
    /// The worker process exited or was killed (exit status is in the message).
    worker_died = 1,
    /// A configured deadline expired (worker bring-up, frame timeout, control reply).
    timeout = 2,
    /// The caller's abort predicate fired, or the worker acknowledged a cancel.
    cancelled = 3,
    /// Malformed or out-of-sequence wire traffic; the connection is unusable afterwards.
    protocol = 4,
    /// The worker rejected the request (bad dimensions, missing buffers, ...).
    bad_request = 5,
    /// The worker's inference callback reported an error (message carries its text).
    worker_error = 6,
    /// The worker executable could not be spawned.
    spawn_failed = 7,
    /// Operation requires a started worker.
    not_started = 8,
    /// Invalid argument passed by the caller of this library.
    invalid_argument = 9,
    /// Anything else (system call failures, allocation, ...).
    internal = 10,
};

/// Base class for all nuketorch exceptions. Derives from std::runtime_error so
/// existing `catch (const std::runtime_error&)` call sites keep working.
class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& what) : std::runtime_error(what), code_(code) {}
    ErrorCode code() const { return code_; }

private:
    ErrorCode code_;
};

/// Worker process exited or was killed while the host still needed it.
class WorkerDiedError : public Error {
public:
    explicit WorkerDiedError(const std::string& what) : Error(ErrorCode::worker_died, what) {}
};

/// A deadline expired (READY handshake, frame timeout, control reply).
class TimeoutError : public Error {
public:
    explicit TimeoutError(const std::string& what) : Error(ErrorCode::timeout, what) {}
};

/// The frame was cancelled via the abort predicate / cancel flag.
/// Worker inference callbacks may also throw this to acknowledge a cancel early.
class CancelledError : public Error {
public:
    explicit CancelledError(const std::string& what = "Cancelled by user")
        : Error(ErrorCode::cancelled, what) {}
};

/// Malformed, oversized, or out-of-sequence wire traffic. The connection is
/// considered poisoned; the client kills and forgets the worker before throwing.
class ProtocolError : public Error {
public:
    explicit ProtocolError(const std::string& what) : Error(ErrorCode::protocol, what) {}
};

/// The worker rejected the request as invalid.
class BadRequestError : public Error {
public:
    explicit BadRequestError(const std::string& what) : Error(ErrorCode::bad_request, what) {}
};

/// The worker's inference callback threw; the message carries the worker-side text.
class WorkerReportedError : public Error {
public:
    explicit WorkerReportedError(const std::string& what) : Error(ErrorCode::worker_error, what) {}
};

/// The worker executable could not be spawned (posix_spawn failed or exec failed).
class SpawnError : public Error {
public:
    explicit SpawnError(const std::string& what) : Error(ErrorCode::spawn_failed, what) {}
};

}  // namespace nuketorch
