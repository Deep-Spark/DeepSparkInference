option(USE_TRT "Use TensorRT for API comparison" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS   ON)
if (USE_TRT)
    set(LIBRT nvinfer)
    set(LIBPLUGIN nvinfer_plugin)
    set(LIBPARSER nvonnxparser)
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_TRT")
else()
    set(LIBRT ixrt)
    set(LIBPLUGIN ixrt_plugin)
    set(LIBPARSER "")
endif ()
