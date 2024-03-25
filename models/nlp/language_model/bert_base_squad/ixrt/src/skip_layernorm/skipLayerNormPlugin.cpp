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
#include "skipLayerNormPlugin.h"

#include "bertCommon.h"
#include "checkMacrosPlugin.h"
#include "plugin.h"
#include "serialize.h"

using namespace nvinfer1;
using namespace nvinfer1::ixrt_plugin;
using namespace nvinfer1::ixrt_plugin::bert;

namespace {
char const* kSKIP_LAYER_NORM_VERSION{"1"};
char const* kSKIP_LAYER_NORM_NAME{"CustomSkipLayerNormPluginDynamic_IxRT"};
char const* kSKIP_LAYER_NORM_VAR_SEQLEN_VERSION{"2"};
}  // namespace

// Static class fields initialization
PluginFieldCollection SkipLayerNormPluginDynamicCreator::mFC{};
std::vector<PluginField> SkipLayerNormPluginDynamicCreator::mPluginAttributes;

// REGISTER_TENSORRT_PLUGIN(SkipLayerNormPluginDynamicCreator);

static inline DataType getParamWordType(DataType cfgType) noexcept {
    if (cfgType == DataType::kINT8) {
        return DataType::kHALF;
    }

    return cfgType;
}

SkipLayerNormPluginDynamicCreator::SkipLayerNormPluginDynamicCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("ld"));
    mPluginAttributes.emplace_back(PluginField("type_id"));
    mPluginAttributes.emplace_back(PluginField("beta"));
    mPluginAttributes.emplace_back(PluginField("gamma"));
    mPluginAttributes.emplace_back(PluginField("bias"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* SkipLayerNormPluginDynamicCreator::getPluginName() const noexcept { return kSKIP_LAYER_NORM_NAME; }

char const* SkipLayerNormPluginDynamicCreator::getPluginVersion() const noexcept { return kSKIP_LAYER_NORM_VERSION; }

PluginFieldCollection const* SkipLayerNormPluginDynamicCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* SkipLayerNormPluginDynamicCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept {
    try {
        gLogInfo << "SkipLayerNormPluginDynamicCreator createPlugin" << endl;

        int32_t ld = 0;
        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        Weights bias{DataType::kFLOAT, nullptr, 0};
        int32_t typeId = -1;

        IXRT_PLUGIN_ASSERT(fc != nullptr);

        ixrt_plugin::validateRequiredAttributesExist({"type_id", "beta", "ld", "gamma"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("ld") == 0) {
                ld = *static_cast<int32_t const*>(fc->fields[i].data);
                gLogInfo << "Building ld: " << ld << endl;
            }

            if (field_name.compare("type_id") == 0) {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
                gLogInfo << "Building typeId: " << typeId << endl;
            }

            if (field_name.compare("beta") == 0) {
                gLogInfo << "Building beta..." << endl;
                beta.values = fc->fields[i].data;
                beta.count = fc->fields[i].length;
                beta.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("gamma") == 0) {
                gLogInfo << "Building gamma..." << endl;
                gamma.values = fc->fields[i].data;
                gamma.count = fc->fields[i].length;
                gamma.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bias") == 0) {
                gLogInfo << "Building bias..." << endl;
                bias.values = fc->fields[i].data;
                bias.count = fc->fields[i].length;
                bias.type = fieldTypeToDataType(fc->fields[i].type);
            }
        }
        gLogInfo << "Type " << typeId << endl;

        IXRT_PLUGIN_CHECK_VALUE(typeId >= 0 && typeId <= 3,
                                ("SkipLayerNorm: Invalid type ID: " + std::to_string(typeId)).c_str());

        IXRT_PLUGIN_CHECK_VALUE(beta.values != nullptr, "SkipLayerNorm: invalid beta");
        IXRT_PLUGIN_CHECK_VALUE(beta.count > 0, "SkipLayerNorm: invalid beta");

        IXRT_PLUGIN_CHECK_VALUE(gamma.values != nullptr, "SkipLayerNorm: invalid gamma");
        IXRT_PLUGIN_CHECK_VALUE(gamma.count > 0, "SkipLayerNorm: invalid gamma");

        IXRT_PLUGIN_CHECK_VALUE(typeId == (int)DataType::kHALF, "typeId != DataType::kHALF error");

        return new SkipLayerNormPluginDynamic(name, static_cast<DataType>(typeId), ld, beta, gamma, bias);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV2* SkipLayerNormPluginDynamicCreator::deserializePlugin(char const* name, void const* serialData,
                                                                          size_t serialLength) noexcept {
    try {
        return new SkipLayerNormPluginDynamic(name, serialData, serialLength);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void SkipLayerNormPluginDynamicCreator::setPluginNamespace(char const* pluginNamespace) noexcept {
    try {
        mNamespace = pluginNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* SkipLayerNormPluginDynamicCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

//#########################################################################//
SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const DataType type, int32_t const ld,
                                                       Weights const& beta, Weights const& gamma, Weights const& bias)
    : mLayerName(name), mGammaDev(nullptr), mBetaDev(nullptr), mHiddenSize(ld), mType(type), mBiasDev(nullptr) {
    IXRT_PLUGIN_ASSERT(mType == nvinfer1::DataType::kFLOAT || mType == nvinfer1::DataType::kHALF ||
                       mType == nvinfer1::DataType::kINT8);

    mCfgType = mType == DataType::kINT8 ? DataType::kHALF : mType;
    mParamWordsize = getElementSize(mCfgType);

    mBeta.convertAndCopy(beta, mCfgType);
    mGamma.convertAndCopy(gamma, mCfgType);

    mHasBias = (bias.values != nullptr);
    if (mHasBias) {
        mBias.convertAndCopy(bias, mCfgType);
    }

    copyToDevice(mGamma, getWeightsSize(mGamma, mCfgType), mGammaDev);
    copyToDevice(mBeta, getWeightsSize(mBeta, mCfgType), mBetaDev);
    if (mHasBias) {
        copyToDevice(mBias, getWeightsSize(mBias, mCfgType), mBiasDev);
    }
}

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string& name, void const* data, size_t length)
    : mLayerName(name), mGammaDev(nullptr), mBetaDev(nullptr), mBiasDev(nullptr) {
    gLogInfo << "SkipLayerNormPluginDynamic deserialize" << endl;

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mCfgType);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mHasBias);

    IXRT_PLUGIN_ASSERT(mCfgType == nvinfer1::DataType::kFLOAT || mCfgType == nvinfer1::DataType::kHALF);
    mParamWordsize = getElementSize(mCfgType);

    char const* d = static_cast<char const*>(data);
    mBeta.convertAndCopy(d, mHiddenSize, mCfgType);
    mGamma.convertAndCopy(d, mHiddenSize, mCfgType);
    if (mHasBias) {
        mBias.convertAndCopy(d, mHiddenSize, mCfgType);
    }

    copyToDevice(mGamma, getWeightsSize(mGamma, mCfgType), mGammaDev);
    copyToDevice(mBeta, getWeightsSize(mBeta, mCfgType), mBetaDev);
    if (mHasBias) {
        copyToDevice(mBias, getWeightsSize(mBias, mCfgType), mBiasDev);
    }
}

// IPluginV2Ext Methods
DataType SkipLayerNormPluginDynamic::getOutputDataType(int32_t index, DataType const* inputTypes,
                                                       int32_t nbInputs) const noexcept {
    try {
        IXRT_PLUGIN_ASSERT(inputTypes != nullptr);
        IXRT_PLUGIN_ASSERT(index == 0);
        IXRT_PLUGIN_ASSERT(nbInputs == 2);
        return inputTypes[0];
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return DataType{};
}

// IPluginV2 Methods
char const* SkipLayerNormPluginDynamic::getPluginType() const noexcept { return kSKIP_LAYER_NORM_NAME; }

char const* SkipLayerNormPluginDynamic::getPluginVersion() const noexcept { return kSKIP_LAYER_NORM_VERSION; }

int32_t SkipLayerNormPluginDynamic::getNbOutputs() const noexcept { return 1; }
int32_t SkipLayerNormPluginDynamic::initialize() noexcept {
    gLogInfo << "SkipLayerNormPluginDynamic initialize" << endl;
    return 0;
}

void SkipLayerNormPluginDynamic::terminate() noexcept { gLogInfo << "SkipLayerNormPluginDynamic terminate" << endl; }

size_t SkipLayerNormPluginDynamic::getSerializationSize() const noexcept {
    const size_t biasSize = mHasBias ? (mHiddenSize * mParamWordsize) : 0;
    return 2 * mParamWordsize * mHiddenSize + 2 * sizeof(DataType) + sizeof(mHiddenSize) + biasSize + sizeof(mHasBias);
}

void SkipLayerNormPluginDynamic::serialize(void* buffer) const noexcept {
    try {
        serialize_value(&buffer, mType);
        serialize_value(&buffer, mCfgType);
        serialize_value(&buffer, mHiddenSize);
        serialize_value(&buffer, mHasBias);

        char* d = static_cast<char*>(buffer);
        serFromDev(d, static_cast<char*>(mBetaDev.get()), mHiddenSize * mParamWordsize);
        serFromDev(d, static_cast<char*>(mGammaDev.get()), mHiddenSize * mParamWordsize);
        if (mHasBias) {
            serFromDev(d, static_cast<char*>(mBiasDev.get()), mHiddenSize * mParamWordsize);
        }
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

void SkipLayerNormPluginDynamic::destroy() noexcept {
    try {
        gLogInfo << "SkipLayerNormPluginDynamic destroy" << endl;
        // This gets called when the network containing plugin is destroyed
        mGammaDev.reset(nullptr);
        mBetaDev.reset(nullptr);
        if (mHasBias) {
            mBiasDev.reset(nullptr);
        }
        delete this;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

void SkipLayerNormPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* SkipLayerNormPluginDynamic::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* SkipLayerNormPluginDynamic::clone() const noexcept {
    try {
        gLogInfo << "SkipLayerNormPluginDynamic clone" << endl;

        auto* p = new SkipLayerNormPluginDynamic(mLayerName, mType, mHiddenSize, mBeta, mGamma, mBias);
        p->initialize();
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs SkipLayerNormPluginDynamic::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs,
                                                          int32_t nbInputs, IExprBuilder& exprBuilder) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        IXRT_PLUGIN_ASSERT(nbInputs == 2);
        IXRT_PLUGIN_ASSERT(outputIndex == 0);
        IXRT_PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);
        return inputs[0];
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return DimsExprs{};
}

bool SkipLayerNormPluginDynamic::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                                           int32_t nbOutputs) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(inOut != nullptr);
        IXRT_PLUGIN_ASSERT(nbInputs == 2);
        IXRT_PLUGIN_ASSERT(nbOutputs == 1);
        IXRT_PLUGIN_ASSERT(pos >= 0 && pos < (nbInputs + nbOutputs));

        PluginTensorDesc const& in = inOut[pos];
        if (pos == 0) {
            return (in.type == mType) && (in.format == TensorFormat::kLINEAR);
        }
        PluginTensorDesc const& prev = inOut[pos - 1];

        return in.type == prev.type && in.format == prev.format && (in.type == DataType::kHALF);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return false;
}

void SkipLayerNormPluginDynamic::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                                 DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept {
    try {
        gLogInfo << "SkipLayerNormPluginDynamic configurePlugin" << endl;

        // Validate input arguments
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        IXRT_PLUGIN_ASSERT(outputs != nullptr);
        IXRT_PLUGIN_ASSERT(nbOutputs == 1);
        IXRT_PLUGIN_ASSERT(nbInputs == 2);
        if (mType == DataType::kFLOAT || mType == DataType::kHALF) {
            IXRT_PLUGIN_ASSERT(mType == inputs[0].desc.type);
            IXRT_PLUGIN_ASSERT(mType == inputs[1].desc.type);
        } else {
            IXRT_PLUGIN_ASSERT(mType == inputs[0].desc.type || DataType::kFLOAT == inputs[0].desc.type);
            IXRT_PLUGIN_ASSERT(mType == inputs[1].desc.type || DataType::kFLOAT == inputs[1].desc.type);
        }
        auto const& inDims0 = inputs[0].desc.dims;
        auto const& inDims1 = inputs[1].desc.dims;
        IXRT_PLUGIN_ASSERT(inDims0.nbDims == inDims1.nbDims);

        IXRT_PLUGIN_ASSERT(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

        IXRT_PLUGIN_ASSERT(inDims0.nbDims == 5);
        mHiddenSize = inDims0.d[HDIM];  // hiddensize
        IXRT_PLUGIN_ASSERT(mHiddenSize != 0U);
        IXRT_PLUGIN_ASSERT(inDims0.d[3] == 1);
        IXRT_PLUGIN_ASSERT(inDims0.d[4] == 1);
        IXRT_PLUGIN_ASSERT(outputs[0].desc.type == DataType::kHALF);

        mCfgType = inputs[0].desc.type == DataType::kINT8 ? DataType::kHALF : inputs[0].desc.type;

        auto const paramType = getParamWordType(mCfgType);
        mParamWordsize = getElementSize(paramType);
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

size_t SkipLayerNormPluginDynamic::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                                                    PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t SkipLayerNormPluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                            void const* const* inputs, void* const* outputs, void* workspace,
                                            cudaStream_t stream) noexcept {
    gLogInfo << "in SkipLayerNormPluginDynamic.." << endl;
    int32_t status = -1;
    try {
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        IXRT_PLUGIN_ASSERT(outputs != nullptr);
        int32_t const inputVolume = volume(inputDesc[0].dims);
        DataType iType = inputDesc->type;

        // Our plugin outputs only one tensor
        // Launch CUDA kernel wrapper and save its return value
        if (iType == DataType::kFLOAT) {
            gLogInfo << "SkipLayerNormPlugin fp32 not supported yet!" << endl;
            return STATUS_NOT_SUPPORTED;
        } else if (iType == DataType::kHALF) {
            auto const* input = static_cast<half const*>(inputs[0]);
            auto skip = (half*)(inputs[1]);
            auto* output = static_cast<half*>(outputs[0]);
            auto const* const bias = static_cast<half const*>(mBiasDev.get());
            auto const* const beta = static_cast<half const*>(mBetaDev.get());
            auto const* const gamma = static_cast<half const*>(mGammaDev.get());

            if (mHasBias) {
                status = computeSkipLayerNorm<half, true>(stream, static_cast<int32_t>(mHiddenSize), inputVolume, input,
                                                          gamma, beta, bias, skip, output);
            } else {
                status = computeSkipLayerNorm<half, false>(stream, static_cast<int32_t>(mHiddenSize), inputVolume,
                                                           input, gamma, beta, bias, skip, output);
            }
        } else {
            IXRT_PLUGIN_CHECK_VALUE(false, "Unsupported type error, expected [kHALF,kFLOAT], but received " +
                                               std::to_string(static_cast<int32_t>(iType)));
        }
        if (status != cudaSuccess) {
            return STATUS_FAILURE;
        }
        return STATUS_SUCCESS;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return STATUS_FAILURE;
}
