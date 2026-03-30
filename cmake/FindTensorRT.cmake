# FindTensorRT.cmake
#
# Locates TensorRT headers and libnvinfer. Set TENSORRT_ROOT to the SDK root
# (contains include/NvInfer.h and lib/libnvinfer.so).
#
# Provides imported target TensorRT::nvinfer (INTERFACE links CUDA::cudart).

set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT SDK root (include/ + lib/)")

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS "${TENSORRT_ROOT}/include"
  NO_DEFAULT_PATH
)
if(NOT TENSORRT_INCLUDE_DIR)
  find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h)
endif()

find_library(TENSORRT_NVINFER_LIBRARY
  NAMES nvinfer
  PATHS "${TENSORRT_ROOT}/lib"
  NO_DEFAULT_PATH
)
if(NOT TENSORRT_NVINFER_LIBRARY)
  find_library(TENSORRT_NVINFER_LIBRARY NAMES nvinfer)
endif()

find_package(CUDAToolkit REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TENSORRT_INCLUDE_DIR TENSORRT_NVINFER_LIBRARY
)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
  add_library(TensorRT::nvinfer SHARED IMPORTED)
  set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "CUDA::cudart"
  )
endif()
