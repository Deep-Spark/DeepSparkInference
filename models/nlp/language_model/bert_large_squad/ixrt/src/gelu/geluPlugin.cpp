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
#include "geluPlugin.h"
#include "bertCommon.h"
#include "checkMacrosPlugin.h"
#include "plugin.h"
#include "serialize.h"

#include <cstdint>

using namespace nvinfer1;
using namespace nvinfer1::ixrt_plugin;
using namespace nvinfer1::ixrt_plugin::bert;

namespace {
char const* const kGELU_IXRT_PLUGIN_VERSION{"1"};
char const* const kGELU_IXRT_PLUGIN_NAME{"CustomGeluPluginDynamic_IxRT"};
}  // namespace

// Static class fields initialization
PluginFieldCollection GeluPluginDynamicCreator::mFC{};
std::vector<PluginField> GeluPluginDynamicCreator::mPluginAttributes;

GeluPluginDynamicCreator::GeluPluginDynamicCreator() {
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GeluPluginDynamicCreator::getPluginName() const noexcept { return kGELU_IXRT_PLUGIN_NAME; }

char const* GeluPluginDynamicCreator::getPluginVersion() const noexcept { return kGELU_IXRT_PLUGIN_VERSION; }

PluginFieldCollection const* GeluPluginDynamicCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* GeluPluginDynamicCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept {
    try {
        gLogVerbose << "GeluPluginDynamicCreator createPlugin\n";
        IXRT_PLUGIN_ASSERT(fc != nullptr);

        Weights bias{DataType::kFLOAT, nullptr, 0};
        int32_t typeId = -1;
        ixrt_plugin::validateRequiredAttributesExist({"type_id", "ld"}, fc);
        int32_t ld = 0;

        for (int32_t i = 0; i < fc->nbFields; i++) {
            IXRT_PLUGIN_ASSERT(fc->fields[i].name != nullptr);
            std::string fieldName(fc->fields[i].name);

            if (fieldName.compare("type_id") == 0) {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            if (fieldName.compare("bias") == 0) {
                bias.values = fc->fields[i].data;
                bias.count = fc->fields[i].length;
                bias.type = fieldTypeToDataType(fc->fields[i].type);
            }
            if (fieldName.compare("ld") == 0) {
                ld = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }

        if (typeId < 0 || typeId > 3) {
            gLogError << "GeluPluginDynamicCreator: invalid typeId " << typeId << std::endl;
            return nullptr;
        }

        return new GeluPluginDynamic(name, static_cast<DataType>(typeId), bias, ld);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GeluPluginDynamicCreator::deserializePlugin(char const* name, void const* serialData,
                                                       size_t serialLength) noexcept {
    // This object will be deleted when the network is destroyed, which will
    // call GeluPluginDynamic::destroy()
    try {
        return new GeluPluginDynamic(name, serialData, serialLength);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void GeluPluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* GeluPluginDynamicCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// REGISTER_TENSORRT_PLUGIN(GeluPluginDynamicCreator);
//#########################################################################//
GeluPluginDynamic::GeluPluginDynamic(const std::string name, const DataType type, Weights const& bias, const int ld)
    : mLayerName(name), mType(type), mLd(ld), mNumBias(bias.count) {
    if (mNumBias > 0) {
        mBias.convertAndCopy(bias, DataType::kHALF);
        copyToDevice(mBias, getWeightsSize(mBias, DataType::kHALF), mBiasDev);
    }
}

GeluPluginDynamic::GeluPluginDynamic(const std::string name, void const* data, size_t length) : mLayerName(name) {
    gLogVerbose << "GeluPluginDynamic deserialize\n";
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mNumBias);

    if (mNumBias > 0) {
        IXRT_PLUGIN_ASSERT(mLd > 0);
        char const* d = static_cast<char const*>(data);
        mBias.convertAndCopy(d, mNumBias, DataType::kHALF);
        copyToDevice(mBias, getWeightsSize(mBias, DataType::kHALF), mBiasDev);
    }
}

// IPluginV2 Methods

char const* GeluPluginDynamic::getPluginType() const noexcept { return kGELU_IXRT_PLUGIN_NAME; }

char const* GeluPluginDynamic::getPluginVersion() const noexcept { return kGELU_IXRT_PLUGIN_VERSION; }

int32_t GeluPluginDynamic::getNbOutputs() const noexcept { return 1; }

int32_t GeluPluginDynamic::initialize() noexcept {
    gLogVerbose << "GeluPluginDynamic initalize\n";
    return 0;
}

void GeluPluginDynamic::terminate() noexcept { gLogVerbose << "GeluPluginDynamic terminate\n"; }

size_t GeluPluginDynamic::getSerializationSize() const noexcept {
    const size_t wordSize = getElementSize(mType);
    return sizeof(mType) + sizeof(mLd) + sizeof(mNumBias) + mNumBias * sizeof(half);
}

void GeluPluginDynamic::serialize(void* buffer) const noexcept {
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mNumBias);
    if (mNumBias > 0) {
        IXRT_PLUGIN_ASSERT(mLd > 0);
        char* d = static_cast<char*>(buffer);

        serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * getElementSize(DataType::kHALF));
    }
}

void GeluPluginDynamic::destroy() noexcept {
    gLogVerbose << "GeluPluginDynamic destroy\n";
    // This gets called when the network containing plugin is destroyed
    if (mNumBias > 0) {
        mBiasDev.reset();
    }
    delete this;
}

void GeluPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* GeluPluginDynamic::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// IPluginV2Ext Methods
nvinfer1::DataType GeluPluginDynamic::getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                                        int32_t nbInputs) const noexcept {
    try {
        IXRT_PLUGIN_ASSERT(index == 0);
        IXRT_PLUGIN_ASSERT(inputTypes != nullptr);
        IXRT_PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF ||
                           inputTypes[0] == DataType::kINT8);
        return inputTypes[0];
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return DataType{};
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GeluPluginDynamic::clone() const noexcept {
    try {
        gLogVerbose << "GeluPluginDynamic clone\n";
        auto* plugin = new GeluPluginDynamic(mLayerName, mType, mBias, mLd);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs GeluPluginDynamic::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
                                                           int32_t nbInputs,
                                                           nvinfer1::IExprBuilder& exprBuilder) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        IXRT_PLUGIN_ASSERT(nbInputs == 1);
        IXRT_PLUGIN_ASSERT(outputIndex == 0);
        return inputs[0];
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return DimsExprs{};
}

bool GeluPluginDynamic::supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut,
                                                  int32_t nbInputs, int32_t nbOutputs) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(inOut != nullptr);
        IXRT_PLUGIN_ASSERT(nbInputs == 1);
        IXRT_PLUGIN_ASSERT(nbOutputs == 1);
        IXRT_PLUGIN_ASSERT(pos >= 0);
        IXRT_PLUGIN_ASSERT(pos < nbInputs + nbOutputs);
    } catch (std::exception const& e) {
        caughtError(e);
        return false;
    }

    PluginTensorDesc const& input = inOut[0];
    if (pos == 0) {
        return (input.type == mType) && (input.format == TensorFormat::kLINEAR);
    }
    if (pos == 1) {
        PluginTensorDesc const& output = inOut[1];
        return (input.type == output.type) && (output.format == TensorFormat::kLINEAR) && (output.type == mType);
    }
    return false;
}

void GeluPluginDynamic::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
                                        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    gLogVerbose << "GeluPluginDynamic configurePlugin\n";

    try {
        IXRT_PLUGIN_ASSERT(in != nullptr);
        IXRT_PLUGIN_ASSERT(nbInputs == 1);
        IXRT_PLUGIN_ASSERT(mType == in[0].desc.type);
        IXRT_PLUGIN_ASSERT(mType == DataType::kHALF || mType == DataType::kINT8);
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

size_t GeluPluginDynamic::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
                                           nvinfer1::PluginTensorDesc const* outputs,
                                           int32_t nbOutputs) const noexcept {
    return 0;
}

template <typename TDataType>
int32_t GeluPluginDynamic::enqueueTyped(void const* input_, void* output_, int32_t const inputVolume,
                                        cudaStream_t stream) noexcept {
    TDataType const* input = static_cast<TDataType const*>(input_);
    TDataType* output = static_cast<TDataType*>(output_);
    int32_t const cols = inputVolume / mLd;
    int32_t const rows = mLd;

    if (mNumBias > 0) {
        TDataType const* bias = static_cast<TDataType*>(mBiasDev.get());
        return computeGeluBias(output, input, bias, rows, cols, stream);
    } else {
        return computeGelu(stream, inputVolume, input, output);
    }
}

int32_t GeluPluginDynamic::enqueueInt8(void const* input_, void* output_, float dequant_scale, float quant_scale,
                                       int32_t const inputVolume, cudaStream_t stream) noexcept {
    int8_t const* input = static_cast<int8_t const*>(input_);
    int8_t* output = static_cast<int8_t*>(output_);
    int32_t const cols = inputVolume / mLd;
    int32_t const rows = mLd;

    if (mNumBias > 0) {
        half const* bias = static_cast<half*>(mBiasDev.get());
        return computeGeluI8O8Bias(output, input, bias, rows, cols, dequant_scale, quant_scale, stream);
    } else {
        return computeGeluI8O8(stream, inputVolume, input, output, dequant_scale, quant_scale);
    }
}

int32_t GeluPluginDynamic::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                   nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
                                   void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(inputDesc != nullptr);
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        IXRT_PLUGIN_ASSERT(outputs != nullptr);
    } catch (std::exception const& e) {
        caughtError(e);
        return STATUS_FAILURE;
    }

    int32_t const inputVolume = volume(inputDesc[0].dims);
    int32_t batch_token_num = inputDesc[0].dims.d[BDIM] * inputDesc[0].dims.d[SDIM];

    // Our plugin outputs only one tensor.
    // Launch CUDA kernel wrapper and save its return value.
    switch (mType) {
        case DataType::kFLOAT:
            return enqueueTyped<float>(inputs[0], outputs[0], inputVolume, stream);
        case DataType::kHALF:
            return enqueueTyped<half>(inputs[0], outputs[0], inputVolume, stream);
        case DataType::kINT8: {
            int8_t* input = (int8_t*)(inputs[0]);
            int8_t* output = (int8_t*)(outputs[0]);
            IxinferBiasGeluI8II8O(batch_token_num, stream, (int8_t*)input, (int8_t*)output,
                                           static_cast<half*>(mBiasDev.get()), mLd,  inputDesc[0].scale,
                                           1.0/outputDesc[0].scale);
            return STATUS_SUCCESS;
        }
        default:
            return STATUS_FAILURE;
    }
}
