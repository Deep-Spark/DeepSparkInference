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
*
* SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "checkMacrosPlugin.h"

namespace nvinfer1 {
namespace ixrt_plugin {
namespace bert {

constexpr uint32_t BDIM = 0;  // batch dimension
constexpr uint32_t SDIM = 1;  // seq len dimension
constexpr uint32_t HDIM = 2;  // hidden dimension

#define TRT_UNUSED (void)

template <typename T>
struct CudaDeleter {
    void operator()(T* buf) { IXRT_PLUGIN_CUASSERT(cudaFree(buf)); }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter<T>>;

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept {
    switch (t) {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
        // case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:
            return 1;
        default:
            break;
        // case DataType::kUNKNOWN:
        // case DataType::kINT64:
        // case DataType::kFLOAT64:
            // break;
    }
    return 0;
}

inline int64_t getWeightsSize(nvinfer1::Weights const& w, nvinfer1::DataType type) {
    return w.count * getElementSize(type);
}

template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>& ptr, void* cudaMem) {
    ptr.reset(static_cast<T*>(cudaMem), bert::CudaDeleter<T>());
}

struct WeightsWithOwnership : public nvinfer1::Weights {
    ILogger* logger_;
    WeightsWithOwnership() {
        values = nullptr;
        count = 0;
    }
    ~WeightsWithOwnership() { operator delete[](const_cast<void*>(values)); }

    WeightsWithOwnership(WeightsWithOwnership const&) = delete;
    WeightsWithOwnership operator=(WeightsWithOwnership const&) = delete;
    WeightsWithOwnership(WeightsWithOwnership const&&) = delete;
    WeightsWithOwnership operator=(WeightsWithOwnership const&&) = delete;

    void convertAndCopy(nvinfer1::Weights const& src, nvinfer1::DataType type, float scale = 1) {
        this->type = type;
        this->count = src.count;

        if (type == nvinfer1::DataType::kFLOAT) {
            auto destBuf = new float[src.count];
            this->values = destBuf;

            if (src.type == nvinfer1::DataType::kFLOAT) {
                ixrt_plugin::gLogInfo << "Float Weights(Host) => Float Array(Host)" << endl;
                std::copy_n(static_cast<float const*>(src.values), src.count, destBuf);
            } else {
                IXRT_PLUGIN_ASSERT(src.type == nvinfer1::DataType::kHALF);

                ixrt_plugin::gLogInfo << "Half Weights(Host) => Float Array(Host)" << endl;
                auto const s = static_cast<half const*>(src.values);
                auto d = static_cast<float*>(const_cast<void*>(this->values));

                for (auto it = 0; it < src.count; it++) {
                    d[it] = __half2float(s[it]);
                }
            }
        } else if (type == nvinfer1::DataType::kHALF) {
            auto destBuf = new half[src.count];
            this->values = destBuf;

            if (src.type == nvinfer1::DataType::kHALF) {
                ixrt_plugin::gLogInfo << "Half Weights(Host) => Half Array(Host)" << endl;
                std::copy_n(static_cast<half const*>(src.values), src.count, destBuf);
            } else {
                IXRT_PLUGIN_ASSERT(src.type == nvinfer1::DataType::kFLOAT);

                ixrt_plugin::gLogInfo << "Float Weights(Host) => Half Array(Host)" << endl;
                auto const s = static_cast<float const*>(src.values);
                auto d = static_cast<half*>(const_cast<void*>(this->values));

                for (auto it = 0; it < src.count; it++) {
                    d[it] = __float2half(s[it]);
                }
            }
        } else if (type == nvinfer1::DataType::kINT8) {
            auto destBuf = new int8_t[src.count];
            this->values = destBuf;

            if (src.type == nvinfer1::DataType::kFLOAT) {
                ixrt_plugin::gLogInfo << "Float Weights(Host) => Int8 Array(Host)" << endl;
                auto const s = static_cast<float const*>(src.values);
                auto d = static_cast<int8_t*>(const_cast<void*>(this->values));

                for (auto it = 0; it < src.count; it++) {
                    int32_t v = static_cast<int32_t>(std::roundf(s[it] / scale));
                    d[it] = v <= -127 ? -127 : (v >= 127 ? 127 : v);
                }
            } else if (src.type == nvinfer1::DataType::kINT8) {
                ixrt_plugin::gLogInfo << "Int8 Weights(Host) => Int8 Array(Host)" << endl;
                std::copy_n(static_cast<int8_t const*>(src.values), src.count, destBuf);
            } else {
                throw std::runtime_error("Unsupported DataType specified for plugin.");
            }
        } else {
            throw std::runtime_error("Unsupported DataType specified for plugin.");
        }
    }

    void convertAndCopy(char const*& srcBuf, size_t count, nvinfer1::DataType type) noexcept {
        this->type = type;
        this->count = count;
        auto const nbBytes = getWeightsSize(*this, type);
        auto destBuf = new char[nbBytes];
        this->values = destBuf;

        std::copy_n(srcBuf, nbBytes, destBuf);
        srcBuf += nbBytes;
    }
};

template <typename T>
inline void copyToDevice(WeightsWithOwnership& hostWeights, size_t nbBytes, cuda_unique_ptr<T>& cudaWeights) {
    if (hostWeights.values) {
        void* cudaMem{nullptr};
        IXRT_PLUGIN_CUASSERT(cudaMalloc(&cudaMem, nbBytes));
        IXRT_PLUGIN_CUASSERT(cudaMemcpy(cudaMem, hostWeights.values, nbBytes, cudaMemcpyHostToDevice));
        cudaWeights.reset(static_cast<T*>(cudaMem));
    }
}

template <typename T>
inline void serFromDev(char*& buffer, T const* data, size_t nbElem) {
    const size_t len = sizeof(T) * nbElem;
    IXRT_PLUGIN_CUASSERT(cudaMemcpy(buffer, static_cast<void const*>(data), len, cudaMemcpyDeviceToHost));
    buffer += len;
}

template <typename T>
inline T* deserToDev(char const*& buffer, size_t nbElem) {
    void* dev{nullptr};
    const size_t len = sizeof(T) * nbElem;
    IXRT_PLUGIN_CUASSERT(cudaMalloc(&dev, len));
    IXRT_PLUGIN_CUASSERT(cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice));

    buffer += len;
    return static_cast<T*>(dev);
}

inline nvinfer1::DataType fieldTypeToDataType(const nvinfer1::PluginFieldType ftype) {
    switch (ftype) {
        case nvinfer1::PluginFieldType::kFLOAT32: {
            gLogInfo << "PluginFieldType is Float32" << endl;
            return nvinfer1::DataType::kFLOAT;
        }
        case nvinfer1::PluginFieldType::kFLOAT16: {
            gLogInfo << "PluginFieldType is Float16" << endl;
            return nvinfer1::DataType::kHALF;
        }
        case nvinfer1::PluginFieldType::kINT32: {
            gLogInfo << "PluginFieldType is Int32" << endl;
            return nvinfer1::DataType::kINT32;
        }
        case nvinfer1::PluginFieldType::kINT8: {
            gLogInfo << "PluginFieldType is Int8" << endl;
            return nvinfer1::DataType::kINT8;
        }
        default:
            throw std::invalid_argument("No corresponding datatype for plugin field type");
    }
}

inline int64_t volume(nvinfer1::Dims const& d) {
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}
}  // namespace bert
}  // namespace ixrt_plugin
}  // namespace nvinfer1
