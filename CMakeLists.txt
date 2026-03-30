cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(nuketorch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(nuketorch STATIC
  src/IPC.cpp
  src/SharedMemoryBuffer.cpp
  src/Protocol.cpp
  src/ImageUtils.cpp
  src/InferenceClient.cpp
  src/WorkerHarness.cpp
)

target_include_directories(nuketorch
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_library(nuketorch::nuketorch ALIAS nuketorch)

include(GNUInstallDirs)

install(TARGETS nuketorch
  EXPORT nuketorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/nuketorch DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT nuketorchTargets
  FILE nuketorchTargets.cmake
  NAMESPACE nuketorch::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nuketorch
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nuketorchConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/nuketorchConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nuketorch
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/nuketorchConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/nuketorchConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/nuketorchConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nuketorch
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchLibtorch.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nuketorch
)

option(NUKETORCH_BUILD_TORCH_WORKER "Build torch_worker helpers (requires libtorch)" OFF)
if(NUKETORCH_BUILD_TORCH_WORKER)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(FetchLibtorch)
  find_package(Torch REQUIRED)

  add_library(nuketorch_torch_worker STATIC
    src/torch_worker/BackendParams.cpp
    src/torch_worker/CudaPinnedMemory.cpp
    src/torch_worker/TorchWorkerUtils.cpp
    src/torch_worker/TorchScriptBackend.cpp
    src/torch_worker/AOTInductorBackend.cpp
    src/torch_worker/InferenceBackend.cpp
    src/torch_worker/BackendCache.cpp
  )
  target_include_directories(nuketorch_torch_worker PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(nuketorch_torch_worker PUBLIC nuketorch::nuketorch ${TORCH_LIBRARIES})
  target_compile_options(nuketorch_torch_worker PUBLIC ${TORCH_CXX_FLAGS})
  add_library(nuketorch::nuketorch_torch_worker ALIAS nuketorch_torch_worker)

  install(TARGETS nuketorch_torch_worker
    EXPORT nuketorchTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

option(NUKETORCH_BUILD_TESTING "Build nuketorch unit tests" ON)
if(BUILD_TESTING AND NUKETORCH_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
