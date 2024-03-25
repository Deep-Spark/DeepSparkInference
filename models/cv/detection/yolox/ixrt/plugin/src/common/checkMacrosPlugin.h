/* Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
* All Rights Reserved.
*
*    Licensed under the Apache License, Version 2.0 (the "License"); you may
*    not use this file except in compliance with the License. You may obtain
*    a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0
*
*    Unless required by applicable law or agreed to in writing, software
*    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
*    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
*    License for the specific language governing permissions and limitations
*    under the License.
*/
#pragma once
#include <cublasLt.h>

#include <cassert>
#include <iostream>
#include <mutex>
#include <sstream>

#include "NvInfer.h"
#include "NvInferRuntime.h"

// Logs failed assertion and aborts.
// Aborting is undesirable and will be phased-out from the plugin module, at which point
// PLUGIN_ASSERT will perform the same function as PLUGIN_VALIDATE.
using namespace std;

namespace nvinfer1 {
namespace plugin {

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#define IXRT_PLUGIN_CHECK_VALUE(value, msg)                            \
    {                                                                  \
        if (not(value)) {                                              \
            std::cerr << __FILE__ << " (" << __LINE__ << ")"           \
                      << "-" << __FUNCTION__ << " : "                  \
                      << " Plugin assert error: " << msg << std::endl; \
            std::exit(EXIT_FAILURE);                                   \
        }                                                              \
    }

#define IXRT_PLUGIN_ASSERT(value)                             \
    {                                                         \
        if (not(value)) {                                     \
            std::cerr << __FILE__ << " (" << __LINE__ << ")"  \
                      << "-" << __FUNCTION__ << " : "         \
                      << " Plugin assert false" << std::endl; \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    }

#define IXRT_PLUGIN_CHECK_CUDA(call)                                        \
    do {                                                                    \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess) {                                    \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

inline void caughtError(const std::exception& e) { std::cerr << e.what() << std::endl; }

#define IXRT_PLUGIN_FAIL(msg)                         \
    do {                                              \
        std::ostringstream stream;                    \
        stream << "Assertion failed: " << msg << "\n" \
               << __FILE__ << ':' << __LINE__ << "\n" \
               << "Aborting..."                       \
               << "\n";                               \
        IXRT_PLUGIN_CHECK_CUDA(cudaDeviceReset());    \
        abort;                                        \
    } while (0)

inline void throwCudaError(char const* file, char const* function, int32_t line, int32_t status, char const* msg) {
    std::cerr << file << " (" << line << ")"
              << "-" << function << " : " << msg << std::endl;
    std::exit(EXIT_FAILURE);
}

#define IXRT_PLUGIN_CUASSERT(status_)                             \
    {                                                             \
        auto s_ = status_;                                        \
        if (s_ != cudaSuccess) {                                  \
            const char* msg = cudaGetErrorString(s_);             \
            throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg); \
        }                                                         \
    }

#undef CUINFER_CHECK
#define CUINFER_CHECK(func)                                                              \
    do {                                                                                 \
        cuinferStatus_t status = (func);                                                 \
        if (status != CUINFER_STATUS_SUCCESS) {                                          \
            std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " \
                      << cuinferGetErrorString(status) << std::endl;                     \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

static std::string _cudaGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "CUBLAS_UNKNOW";
}

template <typename T>
void check_gpu_error(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        throw std::runtime_error(std::string("[CUDA][ERROR] ") + +file + "(" + std::to_string(line) +
                                 "): " + (_cudaGetErrorString(result)) + "\n");
    }
}

#define CHECK_GPU_ERROR(val) check_gpu_error((val), #val, __FILE__, __LINE__)

template <ILogger::Severity kSeverity>
class LogStream : public std::ostream {
    class Buf : public std::stringbuf {
       public:
        int32_t sync() override;
    };

    Buf buffer;
    std::mutex mLogStreamMutex;

   public:
    std::mutex& getMutex() { return mLogStreamMutex; }
    LogStream() : std::ostream(&buffer){};
};

// Use mutex to protect multi-stream write to buffer
template <ILogger::Severity kSeverity, typename T>
LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, T const& msg) {
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << msg;
    return stream;
}

// Special handling static numbers
template <ILogger::Severity kSeverity>
inline LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, int32_t num) {
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << num;
    return stream;
}

// Special handling std::endl
template <ILogger::Severity kSeverity>
inline LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, std::ostream& (*f)(std::ostream&)) {
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << f;
    return stream;
}

extern LogStream<ILogger::Severity::kERROR> gLogError;
extern LogStream<ILogger::Severity::kWARNING> gLogWarning;
extern LogStream<ILogger::Severity::kINFO> gLogInfo;
extern LogStream<ILogger::Severity::kVERBOSE> gLogVerbose;
}  // namespace plugin
}  // namespace nvinfer1
