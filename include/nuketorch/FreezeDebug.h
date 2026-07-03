// FreezeDebug.h
//
// Crash/freeze-survival logger. Every call opens the log file with O_APPEND |
// O_SYNC, writes a single line, fsync()s, and closes. That guarantees the OS
// has committed the byte to disk before we return -- so even if the kernel,
// X server or NVIDIA driver wedges immediately after, the last line in the
// log is the last instruction that actually executed.
//
// Default log path: /tmp/nnretime_freeze.log
// Override via env var:  NUKETORCH_FREEZE_LOG=/some/other/path
//
// Usage:
//   #include <nuketorch/FreezeDebug.h>
//   FREEZE_LOG("PLUGIN", "about to fork()");
//   FREEZE_LOG("PLUGIN", "fork returned pid=%d", pid);

#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace nuketorch::debug {

inline const char* freezeLogPath() {
    static const char* path = []() {
        const char* p = std::getenv("NUKETORCH_FREEZE_LOG");
        return (p && *p) ? p : "/tmp/nnretime_freeze.log";
    }();
    return path;
}

inline void freezeLog(const char* side, const char* fmt, ...) {
    const int fd = ::open(freezeLogPath(),
                          O_WRONLY | O_CREAT | O_APPEND | O_SYNC, 0644);
    if (fd < 0) {
        return;
    }

    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const auto secs = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - secs).count();
    const std::time_t tt = clock::to_time_t(now);
    std::tm tm_buf{};
    ::localtime_r(&tt, &tm_buf);

    char head[96];
    int n = std::snprintf(head, sizeof(head),
                          "[%04d-%02d-%02d %02d:%02d:%02d.%03lld] [%s] [pid=%d] ",
                          tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                          tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                          static_cast<long long>(ms),
                          side ? side : "?",
                          static_cast<int>(::getpid()));
    if (n < 0) n = 0;

    char body[1024];
    va_list ap;
    va_start(ap, fmt);
    int m = std::vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);
    if (m < 0) m = 0;

    char line[1200];
    int total = std::snprintf(line, sizeof(line), "%s%s\n", head, body);
    if (total < 0) total = 0;
    if (total >= static_cast<int>(sizeof(line))) total = sizeof(line) - 1;

    ssize_t written = 0;
    while (written < total) {
        ssize_t w = ::write(fd, line + written, total - written);
        if (w <= 0) break;
        written += w;
    }
    ::fsync(fd);
    ::close(fd);
}

}  // namespace nuketorch::debug

#define FREEZE_LOG(side, ...) ::nuketorch::debug::freezeLog((side), __VA_ARGS__)
