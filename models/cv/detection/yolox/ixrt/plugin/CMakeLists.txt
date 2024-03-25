cmake_minimum_required(VERSION 3.10)

project(ixrt_plugin_unittest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")
set(CMAKE_CXX_EXTENSIONS OFF)

set(IXRT_HOME)

set(TARGET_NAME ixrt_plugin)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)

set(PLUGIN_REPO_PATH ${PROJECT_SOURCE_DIR})
set(PLUGIN_SOURCES)

if (USE_TRT)
    # cuda
    find_package(CUDA)
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    # tensorrt
    include_directories(/usr/include/x86_64-linux-gnu/)
    link_directories(/usr/lib/x86_64-linux-gnu)
    include_directories(/usr/local/TensorRT/include)
    link_directories(/usr/local/TensorRT/lib)
else()
    include(FindIxrt)
    include(FindCompiler)
    include(FindCuda)
endif ()

include(FindPluginFiles)
list(APPEND PLUGIN_FILES ${PLUGIN_SOURCES})

if (USE_TRT)
    include_directories(
        ${CUDA_PATH}/include)
else()
    include_directories(${IXRT_INCLUDE_DIR}
        ${CUDA_PATH}/include)
    ################################## Compile Options ######################################
    # For cuda files
    string(APPEND CMAKE_CXX_FLAGS " --cuda-path=${CUDA_PATH}")
    string(APPEND CMAKE_CXX_FLAGS " --cuda-gpu-arch=${CUDA_ARCH}")
endif ()

cuda_add_library(${SHARED_TARGET} SHARED
        ${PLUGIN_FILES}
)

if (USE_TRT)
    target_link_libraries(${SHARED_TARGET}  cublasLt cudart nvinfer)
else()
    target_link_libraries(${SHARED_TARGET} PUBLIC  cublasLt cudart ixrt)
endif ()

target_link_directories(${SHARED_TARGET} PUBLIC ${IXRT_LIB_DIR})
target_include_directories(${SHARED_TARGET} PUBLIC src PUBLIC src/common PUBLIC  src/common/kernels)
# add_subdirectory(unit_test)
