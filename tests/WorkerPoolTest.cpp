#include <gtest/gtest.h>

#include <nuketorch/WorkerPool.h>

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#ifndef FAKE_WORKER_BIN
#define FAKE_WORKER_BIN "./FakeWorker"
#endif

using nuketorch::InferenceClient;
using nuketorch::WorkerPool;

TEST(WorkerPoolTest, SameKeySharesOneClient) {
    auto a = WorkerPool::instance().acquire(FAKE_WORKER_BIN, 2, "share-test");
    auto b = WorkerPool::instance().acquire(FAKE_WORKER_BIN, 2, "share-test");

    InferenceClient* pa = nullptr;
    InferenceClient* pb = nullptr;
    a.withClient([&](InferenceClient& c) { pa = &c; });
    b.withClient([&](InferenceClient& c) { pb = &c; });
    EXPECT_EQ(pa, pb);
}

TEST(WorkerPoolTest, DifferentShareKeysGetDifferentClients) {
    auto a = WorkerPool::instance().acquire(FAKE_WORKER_BIN, 2, "model-a");
    auto b = WorkerPool::instance().acquire(FAKE_WORKER_BIN, 2, "model-b");

    InferenceClient* pa = nullptr;
    InferenceClient* pb = nullptr;
    a.withClient([&](InferenceClient& c) { pa = &c; });
    b.withClient([&](InferenceClient& c) { pb = &c; });
    EXPECT_NE(pa, pb);
}

TEST(WorkerPoolTest, EntryRespawnsFreshAfterLastRelease) {
    {
        auto a = WorkerPool::instance().acquire(FAKE_WORKER_BIN, 2, "respawn");
        a.withClient([](InferenceClient& c) {
            c.start();
            EXPECT_TRUE(c.ping());
        });
    }  // last handle dropped: worker stopped, entry destroyed

    auto b = WorkerPool::instance().acquire(FAKE_WORKER_BIN, 2, "respawn");
    b.withClient([](InferenceClient& c) {
        EXPECT_FALSE(c.ping());  // fresh, unstarted client
        c.start();
        EXPECT_TRUE(c.ping());
    });
}

TEST(WorkerPoolTest, ConcurrentNodesSerializeOnOneWorker) {
    auto keepalive = WorkerPool::instance().acquire(FAKE_WORKER_BIN, 2, "concurrent");
    keepalive.withClient([](InferenceClient& c) { c.start(); });

    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&failures]() {
            auto handle = WorkerPool::instance().acquire(FAKE_WORKER_BIN, 2, "concurrent");
            for (int i = 0; i < 5; ++i) {
                std::vector<float> in0{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
                std::vector<float> in1{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
                std::vector<float> out(6, 0.0f);

                nuketorch::FrameBuffers fb;
                fb.inputs = {in0.data(), in1.data()};
                fb.output = out.data();
                fb.width = 2;
                fb.height = 1;
                fb.channels = 3;

                nuketorch::InferenceConfig cfg;
                cfg.model_path = "unused.pt";
                cfg.params["timestep"] = "0.25";

                try {
                    handle.withClient([&](InferenceClient& c) { c.processFrame(fb, cfg); });
                    for (size_t j = 0; j < out.size(); ++j) {
                        const float expected = ((in0[j] + in1[j]) * 0.5f) + 0.25f;
                        if (std::fabs(out[j] - expected) > 1e-5f) {
                            ++failures;
                        }
                    }
                } catch (...) {
                    ++failures;
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(failures.load(), 0);
}
