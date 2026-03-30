#include <nuketorch/WorkerHarness.h>

#include <nuketorch/InferenceMetrics.h>
#include <nuketorch/IPC.h>
#include <nuketorch/Protocol.h>
#include <nuketorch/SharedMemoryBuffer.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

namespace nuketorch {

int workerMain(int argc, char** argv, InferenceCallback inference, GpuInfoCallback gpu_info) {
    if (argc < 2) {
        std::cerr << "Usage: worker <socket_path>\n";
        return 2;
    }

    if (!inference) {
        std::cerr << "workerMain: inference callback is null\n";
        return 2;
    }

    const std::string socket_path = argv[1];

    try {
        IPCClient client(socket_path);
        client.send("READY");

        while (true) {
            const std::string msg = client.receive(-1);

            if (msg == "QUIT") {
                client.send("BYE");
                break;
            }
            if (msg == "PING") {
                client.send("PONG");
                continue;
            }
            if (msg == "GPUINFO") {
                if (gpu_info) {
                    client.send(gpu_info());
                } else {
                    client.send("ERROR|GPU info not available");
                }
                continue;
            }

            InferenceRequest req;
            std::string parse_error;
            if (!deserialize(msg, req, parse_error)) {
                client.send(std::string("ERROR|bad_request|") + parse_error);
                continue;
            }

            const int w = req.header.width;
            const int h = req.header.height;
            const int c = req.header.channels;
            if (w <= 0 || h <= 0 || c <= 0) {
                client.send("ERROR|bad_request|invalid dimensions");
                continue;
            }
            if (req.shm_inputs.empty()) {
                client.send("ERROR|bad_request|no inputs");
                continue;
            }
            if (req.shm_output.empty()) {
                client.send("ERROR|bad_request|no output");
                continue;
            }

            const size_t pixel_count =
                static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(c);
            const size_t bytes = pixel_count * sizeof(float);

            std::vector<std::unique_ptr<SharedMemoryBuffer>> input_bufs;
            input_bufs.reserve(req.shm_inputs.size());
            std::vector<void*> input_ptrs;
            input_ptrs.reserve(req.shm_inputs.size());

            for (const auto& name : req.shm_inputs) {
                auto buf = std::make_unique<SharedMemoryBuffer>(name, bytes, false);
                input_ptrs.push_back(buf->data());
                input_bufs.push_back(std::move(buf));
            }

            SharedMemoryBuffer out_buf(req.shm_output, bytes, false);

            WorkerContext ctx{req, std::move(input_ptrs), out_buf.data(), bytes};

            try {
                inference(ctx);
                client.send(std::string("OK") + serializeMetrics(ctx.metrics));
            } catch (const std::exception& e) {
                client.send(std::string("ERROR|std|") + e.what());
            } catch (...) {
                client.send("ERROR|unknown");
            }
        }
    } catch (const IPCClosedError&) {
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Worker fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

}  // namespace nuketorch
