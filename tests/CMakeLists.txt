cmake_minimum_required(VERSION 3.25)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(SharedMemoryTest SharedMemoryBufferTest.cpp)
target_link_libraries(SharedMemoryTest PRIVATE nuketorch::nuketorch gtest_main rt)

add_executable(IPCTest IPCTest.cpp)
target_link_libraries(IPCTest PRIVATE nuketorch::nuketorch gtest_main)

add_executable(ProtocolTest ProtocolTest.cpp)
target_link_libraries(ProtocolTest PRIVATE nuketorch::nuketorch gtest_main)

add_executable(InferenceMetricsTest InferenceMetricsTest.cpp)
target_link_libraries(InferenceMetricsTest PRIVATE nuketorch::nuketorch gtest_main)

add_executable(ScopedTimerTest ScopedTimerTest.cpp)
target_link_libraries(ScopedTimerTest PRIVATE nuketorch::nuketorch gtest_main)

add_executable(ImageUtilsTest ImageUtilsTest.cpp)
target_link_libraries(ImageUtilsTest PRIVATE nuketorch::nuketorch gtest_main)

add_executable(HarnessSmokeWorker HarnessSmokeWorker.cpp)
target_link_libraries(HarnessSmokeWorker PRIVATE nuketorch::nuketorch)

add_executable(FakeWorker FakeWorker.cpp)
target_link_libraries(FakeWorker PRIVATE nuketorch::nuketorch)

add_executable(WorkerHarnessTest WorkerHarnessTest.cpp)
target_link_libraries(WorkerHarnessTest PRIVATE nuketorch::nuketorch gtest_main)
target_compile_definitions(WorkerHarnessTest PRIVATE
  HARNESS_SMOKE_BIN="$<TARGET_FILE:HarnessSmokeWorker>"
  FAKE_WORKER_BIN="$<TARGET_FILE:FakeWorker>")
add_dependencies(WorkerHarnessTest HarnessSmokeWorker FakeWorker)

add_executable(InferenceClientTest InferenceClientTest.cpp)
target_link_libraries(InferenceClientTest PRIVATE nuketorch::nuketorch gtest_main rt)
target_compile_definitions(InferenceClientTest PRIVATE FAKE_WORKER_BIN="$<TARGET_FILE:FakeWorker>")
add_dependencies(InferenceClientTest FakeWorker)

set(NUKETORCH_TEST_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/TestPaths.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/TestPaths.h"
  @ONLY
)

if(TARGET nuketorch_torch_worker)
  add_executable(TorchWorkerUtilsTest TorchWorkerUtilsTest.cpp)
  target_link_libraries(TorchWorkerUtilsTest PRIVATE nuketorch::nuketorch_torch_worker gtest_main)
  target_include_directories(TorchWorkerUtilsTest PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
endif()

include(GoogleTest)
gtest_discover_tests(SharedMemoryTest)
gtest_discover_tests(IPCTest)
gtest_discover_tests(ProtocolTest)
gtest_discover_tests(InferenceMetricsTest)
gtest_discover_tests(ScopedTimerTest)
gtest_discover_tests(ImageUtilsTest)
gtest_discover_tests(WorkerHarnessTest)
gtest_discover_tests(InferenceClientTest)
if(TARGET TorchWorkerUtilsTest)
  gtest_discover_tests(TorchWorkerUtilsTest)
endif()
