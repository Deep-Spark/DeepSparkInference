cmake_minimum_required(VERSION 3.10)

project(nv_plugin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")
set(CMAKE_CXX_EXTENSIONS OFF)

set(TARGET_NAME ixrt_plugin)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)
set(PLUGIN_REPO_PATH ${PROJECT_SOURCE_DIR})

if(DEFINED USE_TENSORRT)
  find_package(CUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_75)

  include_directories(
        ${CUDA_PATH}/include)

  message(STATUS "Plugin lib use TRT 8.6.1")
  set(TRT_INC_PATH /usr/include/x86_64-linux-gnu/)
  set(TRT_LIB_PATH /usr/lib/x86_64-linux-gnu/ /usr/local/cuda/targets/x86_64-linux/lib)
  set(TRT_LIBRARY nvinfer cublasLt)
  
  message(STATUS "cuda_libs = ${CUDA_LIBRARIES}")
  message(STATUS "cudadevrt_libs = ${CUDA_cudadevrt_LIBRARY}")
else()
  include(FindIxrt)
  include(FindCompiler)
  include(FindCuda)
  set(TRT_LIBRARY cublasLt cudart ixrt)
  include_directories(${IXRT_INCLUDE_DIR}
        ${CUDA_PATH}/include)
  add_definitions(-D__ILUVATAR__)

  string(APPEND CMAKE_CXX_FLAGS " -std=c++17")
endif()

include(FindPluginFiles)

################################## Compile Options ######################################
cuda_add_library(${SHARED_TARGET} SHARED
        ${PLUGIN_FILES}
)

target_link_libraries(${SHARED_TARGET} ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY} ${TRT_LIBRARY})
target_link_directories(${SHARED_TARGET} PUBLIC ${CUDA_PATH}/lib64 ${TRT_LIB_PATH} ${IXRT_LIB_DIR})
target_include_directories(${SHARED_TARGET}  PUBLIC ${CUDA_PATH}/include ${TRT_INC_PATH} src PUBLIC src/common)
