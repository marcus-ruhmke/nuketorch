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

include(GoogleTest)
gtest_discover_tests(SharedMemoryTest)
gtest_discover_tests(IPCTest)
gtest_discover_tests(ProtocolTest)
gtest_discover_tests(ImageUtilsTest)
gtest_discover_tests(WorkerHarnessTest)
gtest_discover_tests(InferenceClientTest)
