# This cmake does:
# - Set CUDA_PATH
# - Find libcudart
# - Util functions like cuda_add_library, cuda_add_executable


# CUDA_PATH can be specified through below means shown in priority order 1.
# cmake command line argument, -DCUDA_PATH=/path/to/cuda 2. bash environment
# variable, export CUDA_PATH=/path/to/cuda
if(DEFINED ENV{CUDA_PATH})
  set(CUDA_PATH "$ENV{CUDA_PATH}")
else()
  set(CUDA_PATH
      "/opt/sw_home/local/cuda"
      CACHE PATH "cuda installation root path")
endif()
message(STATUS "Use CUDA_PATH=${CUDA_PATH} ")

# GPU arch
if(NOT "${CUDA_ARCH}" STREQUAL "")
  set(CUDA_ARCH
      ${CUDA_ARCH}
      CACHE STRING "GPU architecture tag, ivcore11")
else("${CUDA_ARCH}" STREQUAL "")
  set(CUDA_ARCH
      "ivcore11"
      CACHE STRING "GPU architecture tag, ivcore11")
endif()
message(STATUS "Use CUDA_ARCH=${CUDA_ARCH}")

macro(cuda_add_executable)
  foreach(File ${ARGN})
    if(${File} MATCHES ".*\.cu$")
      set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
    endif()
  endforeach()
  add_executable(${ARGV})
endmacro()

macro(cuda_add_library)
  foreach(File ${ARGN})
    if(${File} MATCHES ".*\.cu$")
      set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
    endif()
  endforeach()
  add_library(${ARGV})
endmacro()

find_library(
  CUDART_LIBRARY cudart
  PATHS ${CUDA_PATH}
  PATH_SUFFIXES lib/x64 lib64 lib
  NO_DEFAULT_PATH)

if (NOT USE_TRT)
  set(CUDA_LIBRARIES cudart)
endif()
