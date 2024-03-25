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

#include "NvInferRuntimeCommon.h"
#include "bertCommon.h"
#include "checkMacrosPlugin.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "fcPlugin.h"
#include "plugin.h"
#include "serialize.h"
#include <cassert>

using namespace nvinfer1;
using namespace nvinfer1::ixrt_plugin;
using namespace nvinfer1::ixrt_plugin::bert;
using namespace nvinfer1::ixrt_plugin::backend;

namespace {
char const* const kFC_VERSION{"2"};
char const* const kFC_NAME{"CustomFCPluginDynamic_IxRT"};
}  // namespace

// Static class fields initialization
PluginFieldCollection FCInt8PluginDynamicCreator::mFC{};
std::vector<PluginField> FCInt8PluginDynamicCreator::mPluginAttributes;

FCInt8PluginDynamicCreator::FCInt8PluginDynamicCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("out_dims", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("W", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("fc_amax", nullptr, PluginFieldType::kFLOAT32, 2));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* FCInt8PluginDynamicCreator::getPluginName() const noexcept { return kFC_NAME; }

char const* FCInt8PluginDynamicCreator::getPluginVersion() const noexcept { return kFC_VERSION; }

PluginFieldCollection const* FCInt8PluginDynamicCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* FCInt8PluginDynamicCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept {
    try {
        gLogInfo << "Creating FCInt8PluginDynamicCreator..." << endl;
        IXRT_PLUGIN_ASSERT(name != nullptr);
        IXRT_PLUGIN_ASSERT(fc != nullptr);

        int32_t outDims = 0;
        Weights W{DataType::kINT8, nullptr, 0LL};
        Weights Bias{DataType::kFLOAT, nullptr, 0LL};
        ixrt_plugin::validateRequiredAttributesExist({"out_dims", "W", "fc_amax"}, fc);
        vector<float> weight_scale;

        for (int32_t i = 0; i < fc->nbFields; i++) {
            std::string fieldName(fc->fields[i].name);
            if (fieldName.compare("out_dims") == 0) {
                outDims = static_cast<int32_t const*>(fc->fields[i].data)[0];
                gLogInfo << "Building outDims: " << outDims << endl;
            }

            if (fieldName.compare("W") == 0) {
                gLogInfo << "Building W..." << endl;
                W.values = fc->fields[i].data;
                W.count = fc->fields[i].length;
                W.type = fieldTypeToDataType(fc->fields[i].type);
                gLogInfo << "Is W int8: " << (W.type == DataType::kINT8) << endl;
            }

            if (fieldName.compare("Bias") == 0) {
                gLogInfo << "Building Bias..." << endl;
                Bias.values = fc->fields[i].data;
                Bias.count = fc->fields[i].length;
                Bias.type = fieldTypeToDataType(fc->fields[i].type);
                gLogInfo << "Is Bias float32: " << (Bias.type == DataType::kFLOAT) << endl;
            }

            if (fieldName.compare("fc_amax") == 0) {
                gLogInfo << "Building fc_amax..." << endl;
                for (auto j = 0; j < fc->fields[i].length; j++) {
                    auto value = static_cast<float const*>(fc->fields[i].data)[j];
                    weight_scale.emplace_back(value / 127.0);
                }
            }
        }

        if (outDims <= 0) {
            gLogInfo << "Invalid output dimension" << endl;
        }
        if (W.count == 0 || W.values == nullptr || W.count < outDims) {
            gLogInfo << "Invalid weights" << endl;
        }

        DataType type = DataType::kINT8;
        return new FCInt8PluginDynamic(name, type, outDims, W, Bias, weight_scale);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* FCInt8PluginDynamicCreator::deserializePlugin(char const* name, void const* serialData,
                                                         size_t serialLength) noexcept {
    // This object will be deleted when the network is destroyed, which will
    // call FCInt8PluginDynamic::destroy()
    try {
        return new FCInt8PluginDynamic(name, serialData, serialLength);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void FCInt8PluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* FCInt8PluginDynamicCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// REGISTER_TENSORRT_PLUGIN(FCInt8PluginDynamicCreator);
//#########################################################################//
FCInt8PluginDynamic::FCInt8PluginDynamic(std::string const name, DataType const type, int32_t const outDim,
                                         Weights const& W, Weights const& Bias, vector<float> const& scale)
    : mLayerName(name),
      mType(type),
      mOutDim(outDim),
      mNumParams(W.count),
      mNmax(0),
      mK(0),
      mWdev(nullptr),
      mNumBias(Bias.count),
      mScale(scale),
      mBiasdev(nullptr) {
    if (W.type == nvinfer1::DataType::kFLOAT) {
        float weight_max = std::numeric_limits<float>::min();
        for (int64_t wb = 0, we = W.count; wb < we; ++wb) {
            float val = static_cast<const float*>(W.values)[wb];
            weight_max = std::max(weight_max, std::abs(val));
        }
        // mWeightScale = 127 / weight_max;
    }

    mW.convertAndCopy(W, DataType::kINT8, scale[0]);
    copyToDevice(mW, getWeightsSize(mW, DataType::kINT8), mWdev);
    if (Bias.values != nullptr) {
        mBias.convertAndCopy(Bias, DataType::kFLOAT);
        copyToDevice(mBias, getWeightsSize(mBias, DataType::kFLOAT), mBiasdev);
    }
}

FCInt8PluginDynamic::FCInt8PluginDynamic(std::string const name, void const* data, size_t length)
    : mLayerName(name), mWdev(nullptr), mBiasdev(nullptr) {
    gLogInfo << "FCInt8PluginDynamic deserialize" << endl;

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mOutDim);
    deserialize_value(&data, &length, &mNumParams);
    deserialize_value(&data, &length, &mNmax);
    deserialize_value(&data, &length, &mK);
    deserialize_value(&data, &length, &mNumBias);
    deserialize_value(&data, &length, &mScale);

    char const* d = static_cast<char const*>(data);

    mW.convertAndCopy(d, mNumParams, DataType::kINT8);
    copyToDevice(mW, getWeightsSize(mW, DataType::kINT8), mWdev);
    if (mNumBias > 0) {
        mBias.convertAndCopy(d, mNumBias, DataType::kFLOAT);
        copyToDevice(mBias, getWeightsSize(mBias, DataType::kFLOAT), mBiasdev);
    }
}

// IPluginV2 Methods
char const* FCInt8PluginDynamic::getPluginType() const noexcept { return kFC_NAME; }

char const* FCInt8PluginDynamic::getPluginVersion() const noexcept { return kFC_VERSION; }

int32_t FCInt8PluginDynamic::getNbOutputs() const noexcept { return 1; }

int32_t FCInt8PluginDynamic::initialize() noexcept {
    gLogInfo << "FCInt8PluginDynamic initialize" << endl;
    return 0;
}

void FCInt8PluginDynamic::terminate() noexcept { gLogInfo << "FCInt8PluginDynamic terminate" << endl; }

size_t FCInt8PluginDynamic::getSerializationSize() const noexcept {
    return sizeof(mType) + sizeof(mOutDim) + sizeof(mNumParams) + sizeof(mNmax) + sizeof(mK) + sizeof(mNumBias) +
           mScale.size() * sizeof(float) + sizeof(mScale.size()) + getElementSize(DataType::kINT8) * mNumParams +
           getElementSize(DataType::kFLOAT) * mNumBias;
}

void FCInt8PluginDynamic::serialize(void* buffer) const noexcept {
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mOutDim);
    serialize_value(&buffer, mNumParams);
    serialize_value(&buffer, mNmax);
    serialize_value(&buffer, mK);
    serialize_value(&buffer, mNumBias);
    serialize_value(&buffer, mScale);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mWdev.get()), mNumParams * getElementSize(DataType::kINT8));

    if (mNumBias > 0) {
        serFromDev(d, static_cast<char*>(mBiasdev.get()), mNumBias * getElementSize(DataType::kFLOAT));
    }
}

void FCInt8PluginDynamic::destroy() noexcept {
    gLogInfo << "FCInt8PluginDynamic destroy" << endl;
    mWdev.reset(nullptr);
    if (mNumBias > 0) {
        mBiasdev.reset(nullptr);
    }
    delete this;
}

void FCInt8PluginDynamic::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* FCInt8PluginDynamic::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// IPluginV2Ext Methods
DataType FCInt8PluginDynamic::getOutputDataType(int32_t index, DataType const* inputTypes,
                                                int32_t nbInputs) const noexcept {
    IXRT_PLUGIN_ASSERT(index == 0);
    IXRT_PLUGIN_ASSERT(nbInputs == 1);
    IXRT_PLUGIN_ASSERT(inputTypes != nullptr);
    // IXRT_PLUGIN_ASSERT(inputTypes[0] == DataType::kINT8);
    return inputTypes[0];
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* FCInt8PluginDynamic::clone() const noexcept {
    try {
        gLogInfo << "FCInt8PluginDynamic clone" << endl;

        auto* p = new FCInt8PluginDynamic(mLayerName, mType, mOutDim, mW, mBias, mScale);
        p->setPluginNamespace(mNamespace.c_str());

        return p;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs FCInt8PluginDynamic::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs,
                                                   IExprBuilder& exprBuilder) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(nbInputs == 1);
        IXRT_PLUGIN_ASSERT(outputIndex == 0);
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        DimsExprs ret;
        ret.nbDims = 5;
        ret.d[0] = inputs[0].d[0];
        ret.d[1] = inputs[0].d[1];
        ret.d[2] = exprBuilder.constant(mOutDim);
        ret.d[3] = exprBuilder.constant(1);
        ret.d[4] = exprBuilder.constant(1);
        return ret;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return DimsExprs{};
}

bool FCInt8PluginDynamic::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                                    int32_t nbOutputs) noexcept {
    IXRT_PLUGIN_ASSERT(nbInputs == 1);
    IXRT_PLUGIN_ASSERT(nbOutputs == 1);
    IXRT_PLUGIN_ASSERT(inOut != nullptr);

    PluginTensorDesc const& in = inOut[pos];
    if (pos == 0) {
        return (in.type == mType) && (in.format == TensorFormat::kLINEAR);
    }
    PluginTensorDesc const& prev = inOut[pos - 1];

    // output
    return in.type == prev.type && in.format == prev.format;
}

void FCInt8PluginDynamic::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                          DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept {
    try {
        // Validate input arguments
        IXRT_PLUGIN_ASSERT(nbOutputs == 1);
        IXRT_PLUGIN_ASSERT(nbInputs == 1);
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        IXRT_PLUGIN_ASSERT(outputs != nullptr);
        IXRT_PLUGIN_ASSERT(mType == inputs[0].desc.type);
        auto const& inDims0 = inputs[0].desc.dims;

        IXRT_PLUGIN_ASSERT(inDims0.nbDims == 5);
        mK = inDims0.d[HDIM];  // hiddensize
        // IXRT_PLUGIN_ASSERT(hiddenSize * mOutDim == mNumParams);
        IXRT_PLUGIN_ASSERT(inDims0.d[3] == 1);
        IXRT_PLUGIN_ASSERT(inDims0.d[4] == 1);
#ifdef __ILUVATAR__
        CUINFER_CHECK(cuinferCreate(&cuinfer_handle));
#else
        CHECK_GPU_ERROR(cublasLtCreate(&blaslt_handle));
#endif
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

size_t FCInt8PluginDynamic::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                                             PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept {
    int32_t const B = inputs[0].dims.d[BDIM];
    int32_t const S = inputs[0].dims.d[SDIM];
    int32_t const oE = outputs[0].dims.d[HDIM];
#ifdef __ILUVATAR__
        return B * S * oE * sizeof(int8_t);
#else 
        return B * S * oE * sizeof(int32_t);
#endif
}

int32_t FCInt8PluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                     void const* const* inputs, void* const* outputs, void* workSpace,
                                     cudaStream_t stream) noexcept {
    try {
#ifdef __ILUVATAR__
        CUINFER_CHECK(cuinferSetStream(cuinfer_handle, stream));
#endif
        int32_t const S = inputDesc->dims.d[SDIM];
        int32_t const B = inputDesc->dims.d[BDIM];
        int32_t const E = inputDesc->dims.d[HDIM];
        int32_t const oE = outputDesc->dims.d[HDIM];
        int32_t const n = S * B;
        IXRT_PLUGIN_ASSERT(n >= 0);

        float qkv_in_scale = inputDesc[0].scale;
        float qkv_wei_scale = mScale[0];
        float output_scale = outputDesc[0].scale;
        float qkv_out_scale;
        if (mScale.size() == 2) {
            qkv_out_scale = mScale[1];
        } else {
            qkv_out_scale = output_scale;
        }
#ifdef __ILUVATAR__
        int8_t* buffer = static_cast<int8_t*>(workSpace);
#else
        int32_t* buffer = static_cast<int32_t*>(workSpace);
#endif
        if (mType == DataType::kINT8) {
            auto const* const input = static_cast<int8_t const*>(inputs[0]);
            auto* output = static_cast<int8_t*>(outputs[0]);
            auto weight = static_cast<int8_t*>(mWdev.get());

            float dequant_scale = (qkv_in_scale * qkv_wei_scale) / qkv_out_scale;

            if (mBiasdev.get() != nullptr) {
#ifdef __ILUVATAR__
                cuinfer_i8_gemm(weight, input, nullptr, buffer, 1, oE, n, E, 0, 0, 0, dequant_scale, 0.0, 0,
                                cuinfer_handle, stream);
                dequantGemmWithBias(buffer, static_cast<float*>(mBiasdev.get()), output, B * S, oE, qkv_out_scale,
                                    1.0 / output_scale, stream);
#else
                cublaslt_gemm(weight, input, buffer, 1, oE, n, E, 0, 0, 0, 1, blaslt_handle, stream);
                dequantGemmWithBias(buffer, static_cast<float*>(mBiasdev.get()), output, B * S, oE,  dequant_scale, qkv_out_scale,
                                    1.0 / output_scale, stream);
#endif
                
            } else {
#ifdef __ILUVATAR__
                cuinfer_i8_gemm(weight, input, nullptr, output, 1, oE, n, E, 0, 0, 0, dequant_scale, 0.0, 0,
                                cuinfer_handle, stream);
#else
                
                cublaslt_gemm(weight, input, buffer, 1, oE, n, E, 0, 0, 0, 1, blaslt_handle, stream);
                quantGemm(buffer, output, B * S, oE, dequant_scale, stream);
#endif
            }
        } else {
            gLogError << "Unsupported type error, expected [kINT8], but received " << static_cast<int32_t>(mType)
                      << endl;
            return STATUS_FAILURE;
        }
        return STATUS_SUCCESS;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return STATUS_FAILURE;
}
