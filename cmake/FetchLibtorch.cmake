# FetchLibtorch.cmake
#
# Ensures libtorch is discoverable before find_package(Torch):
# - If LIBTORCH_ROOT is set, prepends it to CMAKE_PREFIX_PATH.
# - Else if Torch is already on CMAKE_PREFIX_PATH, records LIBTORCH_ROOT from Torch_DIR.
# - Else downloads and extracts a zip from download.pytorch.org into the build tree.
#
# Cache variables (all may be set on the cmake command line):
#   LIBTORCH_ROOT   Path to existing libtorch; empty triggers fetch/search.
#   TORCH_VERSION   e.g. 2.10.0
#   CUDA_VARIANT    e.g. cu130, cu126, cu124, cpu
#   LIBTORCH_ABI    cxx11 (default) or pre-cxx11

set(LIBTORCH_ROOT "" CACHE PATH
  "Path to libtorch root (lib/ + share/cmake/Torch). Empty: use CMAKE_PREFIX_PATH or auto-download.")
set(TORCH_VERSION "2.10.0" CACHE STRING "libtorch version for auto-download")
set(CUDA_VARIANT "cu130" CACHE STRING "CUDA segment for URL (cu124, cu126, cu130, cpu, ...)")
set(LIBTORCH_ABI "cxx11" CACHE STRING "pre-cxx11 or cxx11 (matches libtorch zip flavor)")

if(LIBTORCH_ROOT)
  list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
  return()
endif()

find_package(Torch QUIET)
if(Torch_FOUND)
  get_filename_component(_fetch_libtorch_root "${Torch_DIR}/../../.." ABSOLUTE)
  set(LIBTORCH_ROOT "${_fetch_libtorch_root}" CACHE PATH
    "Path to libtorch root (lib/ + share/cmake/Torch). Empty: use CMAKE_PREFIX_PATH or auto-download."
    FORCE)
  return()
endif()

string(TOLOWER "${LIBTORCH_ABI}" _fetch_libtorch_abi_lower)
if(_fetch_libtorch_abi_lower STREQUAL "pre-cxx11"
    OR _fetch_libtorch_abi_lower STREQUAL "precxx11")
  set(_fetch_libtorch_abi_slug "shared-with-deps")
else()
  set(_fetch_libtorch_abi_slug "cxx11-abi-shared-with-deps")
endif()

set(_fetch_libtorch_url
  "https://download.pytorch.org/libtorch/${CUDA_VARIANT}/libtorch-${_fetch_libtorch_abi_slug}-${TORCH_VERSION}%2B${CUDA_VARIANT}.zip")

set(_fetch_libtorch_staging
  "${CMAKE_CURRENT_BINARY_DIR}/_deps/libtorch-${TORCH_VERSION}-${CUDA_VARIANT}")
set(_fetch_libtorch_zip "${_fetch_libtorch_staging}/libtorch.zip")
set(_fetch_libtorch_extracted "${_fetch_libtorch_staging}/libtorch")

if(NOT EXISTS "${_fetch_libtorch_extracted}/share/cmake/Torch/TorchConfig.cmake")
  message(STATUS "Downloading libtorch ${TORCH_VERSION}+${CUDA_VARIANT} (${LIBTORCH_ABI} ABI)...")
  message(STATUS "  URL: ${_fetch_libtorch_url}")
  file(MAKE_DIRECTORY "${_fetch_libtorch_staging}")
  file(DOWNLOAD "${_fetch_libtorch_url}" "${_fetch_libtorch_zip}"
    SHOW_PROGRESS
    STATUS _fetch_libtorch_dl_status
    TLS_VERIFY ON)
  list(GET _fetch_libtorch_dl_status 0 _fetch_libtorch_dl_code)
  if(NOT _fetch_libtorch_dl_code EQUAL 0)
    list(GET _fetch_libtorch_dl_status 1 _fetch_libtorch_dl_msg)
    message(FATAL_ERROR
      "libtorch download failed (code ${_fetch_libtorch_dl_code}): ${_fetch_libtorch_dl_msg}\n"
      "  URL: ${_fetch_libtorch_url}")
  endif()
  file(ARCHIVE_EXTRACT INPUT "${_fetch_libtorch_zip}" DESTINATION "${_fetch_libtorch_staging}")
  file(REMOVE "${_fetch_libtorch_zip}")
  if(NOT EXISTS "${_fetch_libtorch_extracted}/share/cmake/Torch/TorchConfig.cmake")
    message(FATAL_ERROR
      "libtorch extract did not produce expected layout at \"${_fetch_libtorch_extracted}\"")
  endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${_fetch_libtorch_extracted}")
set(LIBTORCH_ROOT "${_fetch_libtorch_extracted}" CACHE PATH
  "Path to libtorch root (lib/ + share/cmake/Torch). Empty: use CMAKE_PREFIX_PATH or auto-download."
  FORCE)
