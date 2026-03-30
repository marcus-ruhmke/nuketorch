#include <gtest/gtest.h>

#include <nuketorch/ScopedTimer.h>

#include <thread>

TEST(ScopedTimerTest, MeasuresElapsedTime) {
    double ms = -1;
    {
        nuketorch::ScopedTimer t(ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_GE(ms, 5.0);
    EXPECT_LE(ms, 50.0);
}

TEST(ScopedTimerTest, DefaultTargetWrittenOnScopeExit) {
    double ms = -1;
    { nuketorch::ScopedTimer t(ms); }
    EXPECT_GE(ms, 0.0);
}
