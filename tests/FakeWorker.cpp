#include <nuketorch/WorkerHarness.h>

#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    return nuketorch::workerMain(
        argc,
        argv,
        [](const nuketorch::WorkerContext& ctx) {
            const auto& req = ctx.request;
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
        },
        []() { return std::string("OK|FakeWorker"); });
}
