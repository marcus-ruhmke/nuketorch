#include <nuketorch/WorkerHarness.h>

int main(int argc, char** argv) {
    return nuketorch::workerMain(
        argc,
        argv,
        [](const nuketorch::WorkerContext&) {
            // no-op; not used in smoke test
        },
        []() { return std::string("OK|test-gpu"); });
}
