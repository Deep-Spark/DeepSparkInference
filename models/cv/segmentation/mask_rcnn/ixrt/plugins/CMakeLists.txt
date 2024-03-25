cmake_minimum_required(VERSION 3.10)

project(maskrcnn LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)

include(SetFlags)
include(FindCompiler)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--extended-lambda)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/plugin_src)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
if (USE_TRT)
    find_package(CUDA)
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
else()
    include(FindCuda)
endif ()

# tensorrt
if (USE_TRT)
    include_directories(/usr/include/x86_64-linux-gnu/)
    link_directories(/usr/lib/x86_64-linux-gnu)
    if (NV_TRT_PATH)
        include_directories(${NV_TRT_PATH}/include)
        link_directories(${NV_TRT_PATH}/lib)
    else ()
        include_directories(/usr/local/TensorRT/include)
        link_directories(/usr/local/TensorRT/lib)
    endif ()
else()
    include(FindIxrt)
endif ()

if (USE_TRT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED --cuda-gpu-arch=ivcore11")
endif ()

cuda_add_library(maskrcnn SHARED ${PROJECT_SOURCE_DIR}/plugin_src/RpnDecode.cu
    ${PROJECT_SOURCE_DIR}/plugin_src/RoiAlign.cu
    ${PROJECT_SOURCE_DIR}/plugin_src/PredictorDecode.cu
    ${PROJECT_SOURCE_DIR}/plugin_src/BatchedNms.cu
    ${PROJECT_SOURCE_DIR}/plugin_src/MaskRcnnInference.cu
    ${PROJECT_SOURCE_DIR}/plugin_src/RpnNms.cu)

if (USE_TRT)
    target_link_libraries(maskrcnn cudart nvinfer)
else ()
	target_link_directories(maskrcnn PUBLIC ${IXRT_LIB_DIR})
    target_link_libraries(maskrcnn cudart ixrt)
endif ()

add_definitions(-pthread)
