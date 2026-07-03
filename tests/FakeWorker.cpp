#include <nuketorch/Errors.h>
#include <nuketorch/WorkerHarness.h>

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

// Deterministic torch-free worker for integration tests.
//
// Recognized params:
//   timestep   float added to the blended output (default 0.25)
//   crash_now  if present, prints a marker to stderr and _exit(3)s mid-frame
//   sleep_ms   sleeps this long in 10 ms chunks, polling ctx.cancelled() and
//              throwing CancelledError when the host raises the cancel flag
int main(int argc, char** argv) {
    // Marker used by tests asserting that worker stderr reaches host error messages.
    std::fprintf(stderr, "FakeWorker started\n");

    return nuketorch::workerMain(
        argc,
        argv,
        [](const nuketorch::WorkerContext& ctx) {
            const auto& req = ctx.request;

            if (req.params.count("crash_now")) {
                std::fprintf(stderr, "FakeWorker crashing on request\n");
                _exit(3);
            }

            const auto sleep_it = req.params.find("sleep_ms");
            if (sleep_it != req.params.end()) {
                int remaining = std::stoi(sleep_it->second);
                while (remaining > 0) {
                    if (ctx.cancelled && ctx.cancelled()) {
                        throw nuketorch::CancelledError("fake worker cancelled");
                    }
                    const int chunk = remaining < 10 ? remaining : 10;
                    std::this_thread::sleep_for(std::chrono::milliseconds(chunk));
                    remaining -= chunk;
                }
            }

            const int w = req.header.width;
            const int h = req.header.height;
            const int c = req.header.channels;
            const size_t count = static_cast<size_t>(w) * h * c;

            float timestep = 0.25f;
            const auto it = req.params.find("timestep");
            if (it != req.params.end()) {
                timestep = std::stof(it->second);
            }

            if (ctx.input_ptrs.size() < 2 || !ctx.output_ptr) {
                throw std::runtime_error("fake worker expects 2 inputs");
            }

            const float* p0 = static_cast<const float*>(ctx.input_ptrs[0]);
            const float* p1 = static_cast<const float*>(ctx.input_ptrs[1]);
            float* po = static_cast<float*>(ctx.output_ptr);
            for (size_t i = 0; i < count; ++i) {
                po[i] = (p0[i] + p1[i]) * 0.5f + timestep;
            }

            ctx.metrics.backend_forward_ms = 42.0;
            ctx.metrics.backend = "fake";
            ctx.metrics.device = "cpu";
            ctx.metrics.dtype = "float32";
        },
        []() { return std::string("OK|FakeWorker"); });
}
