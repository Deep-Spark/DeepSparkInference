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
#include "ffnPlugin.h"

#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#ifdef __ILUVATAR__
#include "backend/ixinfer/ixinfer_gemm_helper.h"
#endif
#include "bertCommon.h"
#include "checkMacrosPlugin.h"
#include "gelu/geluPlugin.h"
#include "plugin.h"
#include "serialize.h"

using namespace nvinfer1;
using namespace nvinfer1::ixrt_plugin;
using namespace nvinfer1::ixrt_plugin::bert;
using namespace nvinfer1::ixrt_plugin::backend;

namespace {
char const* const kFFN_VERSION{"1"};
char const* const kFFN_NAME{"CustomFFNPluginDynamic_IxRT"};
}  // namespace

// Static class fields initialization
PluginFieldCollection FFNPluginDynamicCreator::mFFN{};
std::vector<PluginField> FFNPluginDynamicCreator::mPluginAttributes;

FFNPluginDynamicCreator::FFNPluginDynamicCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("out_dims", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_type", nullptr, PluginFieldType::kINT32, 1));

    mFFN.nbFields = mPluginAttributes.size();
    mFFN.fields = mPluginAttributes.data();
}

char const* FFNPluginDynamicCreator::getPluginName() const noexcept { return kFFN_NAME; }

char const* FFNPluginDynamicCreator::getPluginVersion() const noexcept { return kFFN_VERSION; }

PluginFieldCollection const* FFNPluginDynamicCreator::getFieldNames() noexcept { return &mFFN; }

IPluginV2* FFNPluginDynamicCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept {
    try {
        gLogInfo << "Creating FFNPluginDynamicCreator..." << endl;
        IXRT_PLUGIN_ASSERT(name != nullptr);
        IXRT_PLUGIN_ASSERT(fc != nullptr);

        int32_t outDims = 0;
        int32_t typeId = -1;
        int32_t act_type = -1;
        Weights W1{DataType::kFLOAT, nullptr, 0LL};
        Weights W2{DataType::kFLOAT, nullptr, 0LL};
        Weights B1{DataType::kFLOAT, nullptr, 0LL};
        ixrt_plugin::validateRequiredAttributesExist({"out_dims", "type_id", "W1", "W2", "B1"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++) {
            std::string fieldName(fc->fields[i].name);
            if (fieldName.compare("out_dims") == 0) {
                outDims = static_cast<int32_t const*>(fc->fields[i].data)[0];
                gLogInfo << "Building outDims: " << outDims << endl;
            }

            if (fieldName.compare("type_id") == 0) {
                typeId = static_cast<int32_t const*>(fc->fields[i].data)[0];
                gLogInfo << "Building typeId: " << typeId << endl;
            }

            if (fieldName.compare("W1") == 0) {
                gLogInfo << "Building W1..." << endl;
                W1.values = fc->fields[i].data;
                W1.count = fc->fields[i].length;
                W1.type = fieldTypeToDataType(fc->fields[i].type);
                gLogInfo << "Is W1 float32: " << (W1.type == DataType::kFLOAT) << endl;
            }

            if (fieldName.compare("W2") == 0) {
                gLogInfo << "Building W2..." << endl;
                W2.values = fc->fields[i].data;
                W2.count = fc->fields[i].length;
                W2.type = fieldTypeToDataType(fc->fields[i].type);
                gLogInfo << "Is W2 float32: " << (W2.type == DataType::kFLOAT) << endl;
            }

            if (fieldName.compare("B1") == 0) {
                gLogInfo << "Building B1..." << endl;
                B1.values = fc->fields[i].data;
                B1.count = fc->fields[i].length;
                B1.type = fieldTypeToDataType(fc->fields[i].type);
                gLogInfo << "Is B1 float32: " << (B1.type == DataType::kFLOAT) << endl;
            }

            if (fieldName.compare("act_type") == 0) {
                gLogInfo << "Building act_type..." << endl;
                act_type = static_cast<int32_t const*>(fc->fields[i].data)[0];
                gLogInfo << "Building act_type: " << act_type << endl;
            }
        }

        if (outDims <= 0) {
            gLogInfo << "Invalid output dimension" << endl;
        }
        if (typeId < 0 || typeId > 1) {
            gLogInfo << "Invalid type id" << typeId << endl;
        }
        if (W1.count == 0 || W1.values == nullptr) {
            gLogInfo << "Invalid weights W1" << endl;
        }
        if (W2.count == 0 || W2.values == nullptr) {
            gLogInfo << "Invalid weights W2" << endl;
        }
        if (B1.count == 0 || B1.values == nullptr) {
            gLogInfo << "Invalid weights B1" << endl;
        }

        DataType type = typeId == 0 ? DataType::kFLOAT : DataType::kHALF;
        return new FFNPluginDynamic(name, type, outDims, act_type, W1, W2, B1);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* FFNPluginDynamicCreator::deserializePlugin(char const* name, void const* serialData,
                                                      size_t serialLength) noexcept {
    // This object will be deleted when the network is destroyed, which will
    // call FFNPluginDynamic::destroy()
    try {
        return new FFNPluginDynamic(name, serialData, serialLength);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void FFNPluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* FFNPluginDynamicCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// REGISTER_TENSORRT_PLUGIN(FFNPluginDynamicCreator);
//#########################################################################//
FFNPluginDynamic::FFNPluginDynamic(std::string const name, DataType const type, int32_t const outDim,
                                   int32_t const act_type, Weights const& W1, Weights const& W2, Weights const& B1)
    : mLayerName(name),
      mType(type),
      mHiddenSize(outDim),
      mActType(act_type),
      mWdev1(nullptr),
      mWdev2(nullptr),
      mBdev1(nullptr) {
    mW1.convertAndCopy(W1, mType);
    mW2.convertAndCopy(W2, mType);
    mB1.convertAndCopy(B1, mType);
    copyToDevice(mW1, getWeightsSize(mW1, mType), mWdev1);
    copyToDevice(mW2, getWeightsSize(mW2, mType), mWdev2);
    copyToDevice(mB1, getWeightsSize(mB1, mType), mBdev1);
}

FFNPluginDynamic::FFNPluginDynamic(std::string const name, void const* data, size_t length)
    : mLayerName(name), mWdev1(nullptr), mWdev2(nullptr), mBdev1(nullptr) {
    gLogInfo << "FFNPluginDynamic deserialize" << endl;

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mActType);

    char const* d = static_cast<char const*>(data);

    mW1.convertAndCopy(d, mHiddenSize * mHiddenSize * 4, mType);
    copyToDevice(mW1, getWeightsSize(mW1, mType), mWdev1);

    mW2.convertAndCopy(d, mHiddenSize * mHiddenSize * 4, mType);
    copyToDevice(mW2, getWeightsSize(mW2, mType), mWdev2);

    mB1.convertAndCopy(d, mHiddenSize * 4, mType);
    copyToDevice(mB1, getWeightsSize(mB1, mType), mBdev1);
}

// IPluginV2 Methods
char const* FFNPluginDynamic::getPluginType() const noexcept { return kFFN_NAME; }

char const* FFNPluginDynamic::getPluginVersion() const noexcept { return kFFN_VERSION; }

int32_t FFNPluginDynamic::getNbOutputs() const noexcept { return 1; }

int32_t FFNPluginDynamic::initialize() noexcept {
    gLogInfo << "FFNPluginDynamic initialize" << endl;
    return 0;
}

void FFNPluginDynamic::terminate() noexcept { gLogInfo << "FFNPluginDynamic terminate" << endl; }

size_t FFNPluginDynamic::getSerializationSize() const noexcept {
    size_t wordSize = getElementSize(mType);
    return wordSize * (mHiddenSize * mHiddenSize * 8 + mHiddenSize * 4) + sizeof(mType) + sizeof(mHiddenSize) +
           sizeof(mActType);
}

void FFNPluginDynamic::serialize(void* buffer) const noexcept {
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mHiddenSize);
    serialize_value(&buffer, mActType);

    size_t wordSize = getElementSize(mType);
    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mWdev1.get()), 4 * mHiddenSize * mHiddenSize * wordSize);
    serFromDev(d, static_cast<char*>(mWdev2.get()), 4 * mHiddenSize * mHiddenSize * wordSize);
    serFromDev(d, static_cast<char*>(mBdev1.get()), 4 * mHiddenSize * wordSize);
}

void FFNPluginDynamic::destroy() noexcept {
    gLogInfo << "FFNPluginDynamic destroy" << endl;
    mWdev1.reset(nullptr);
    mWdev2.reset(nullptr);
    mBdev1.reset(nullptr);
    delete this;
}

void FFNPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* FFNPluginDynamic::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// IPluginV2Ext Methods
DataType FFNPluginDynamic::getOutputDataType(int32_t index, DataType const* inputTypes,
                                             int32_t nbInputs) const noexcept {
    IXRT_PLUGIN_ASSERT(index == 0);
    IXRT_PLUGIN_ASSERT(nbInputs == 1);
    IXRT_PLUGIN_ASSERT(inputTypes != nullptr);
    IXRT_PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* FFNPluginDynamic::clone() const noexcept {
    try {
        gLogInfo << "FFNPluginDynamic clone" << endl;

        auto* p = new FFNPluginDynamic(mLayerName, mType, mHiddenSize, mActType, mW1, mW2, mB1);
        p->setPluginNamespace(mNamespace.c_str());

        return p;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs FFNPluginDynamic::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs,
                                                IExprBuilder& exprBuilder) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(nbInputs == 1);
        IXRT_PLUGIN_ASSERT(outputIndex == 0);
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        DimsExprs ret;
        ret.nbDims = 5;
        ret.d[0] = inputs[0].d[0];
        ret.d[1] = inputs[0].d[1];
        ret.d[2] = exprBuilder.constant(mHiddenSize);
        ret.d[3] = exprBuilder.constant(1);
        ret.d[4] = exprBuilder.constant(1);
        return ret;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return DimsExprs{};
}

bool FFNPluginDynamic::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
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

void FFNPluginDynamic::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
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

size_t FFNPluginDynamic::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                                          PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept {
    int32_t const S = inputs[0].dims.d[SDIM];
    int32_t const B = inputs[0].dims.d[BDIM];
    return B * S * 4 * mHiddenSize * sizeof(half);
}

int32_t FFNPluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                  void const* const* inputs, void* const* outputs, void* workSpace,
                                  cudaStream_t stream) noexcept {
    gLogInfo << "in FFNPluginDynamic.." << endl;
    try {
#ifdef __ILUVATAR__
        CUINFER_CHECK(cuinferSetStream(cuinfer_handle, stream));
#endif
        int32_t const S = inputDesc->dims.d[SDIM];
        int32_t const B = inputDesc->dims.d[BDIM];
        int32_t const n = S * B;
        IXRT_PLUGIN_ASSERT(n >= 0);

        if (mType == DataType::kHALF) {
            auto const* const input = static_cast<half const*>(inputs[0]);
            auto* output = static_cast<half*>(outputs[0]);
            auto weight1 = static_cast<half*>(mWdev1.get());
            auto weight2 = static_cast<half*>(mWdev2.get());
            auto bias1 = static_cast<half*>(mBdev1.get());
            auto buffer = static_cast<half*>(workSpace);

#ifdef __ILUVATAR__
            cuinfer_gemm(weight1, input, bias1, buffer, 1, mHiddenSize * 4, n, mHiddenSize, 0, 0, 0, 1.0f, mActType,
                         stream, cuinfer_handle);
            cuinfer_gemm(weight2, buffer, nullptr, output, 1, mHiddenSize, n, 4 * mHiddenSize, 0, 0, 0, 1.0f, -1,
                         stream, cuinfer_handle);
#else
            cublaslt_gemm(weight1, input, buffer, 1, mHiddenSize * 4, n, mHiddenSize, 0, 0, 0, 1.0f, blaslt_handle,
                          stream);
            computeGeluBias(buffer, buffer, bias1, 4 * mHiddenSize, n, stream);
            cublaslt_gemm(weight2, buffer, output, 1, mHiddenSize, n, mHiddenSize * 4, 0, 0, 0, 1.0f, blaslt_handle,
                          stream);
#endif
        } else {
            gLogError << "Unsupported type error, expected [kHALF], but received " << static_cast<int32_t>(mType)
                      << endl;
            return STATUS_FAILURE;
        }
        return STATUS_SUCCESS;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return STATUS_FAILURE;
}