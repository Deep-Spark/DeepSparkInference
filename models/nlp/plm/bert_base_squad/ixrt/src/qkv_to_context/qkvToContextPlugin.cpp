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
#include "qkvToContextPlugin.h"

#include "bertCommon.h"
#include "checkMacrosPlugin.h"
#include "common_def.cuh"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "plugin.h"
#include "serialize.h"
#include <cstddef>
#include <cstdint>

using namespace nvinfer1;
using namespace nvinfer1::ixrt_plugin;
using namespace nvinfer1::ixrt_plugin::bert;

namespace {
char const* const kQKV_TO_CONTEXT_IXRT_PLUGIN_VERSION{"1"};
char const* const kQKV_TO_CONTEXT_VAR_SEQLEN_IXRT_PLUGIN_VERSION{"2"};
char const* const kQKV_TO_CONTEXT_IXRT_PLUGIN_NAME{"CustomQKVToContextPluginDynamic_IxRT"};
}  // namespace

// Static class fields initialization
PluginFieldCollection QKVToContextPluginDynamicCreator::mFC{};
std::vector<PluginField> QKVToContextPluginDynamicCreator::mPluginAttributes;

constexpr uint32_t IIDX = 0;  // index of the input tensor
constexpr uint32_t MIDX = 1;  // index of the mask

QKVToContextPluginDynamicCreator::QKVToContextPluginDynamicCreator() {
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_mask", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dq_probs", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QKVToContextPluginDynamicCreator::getPluginName() const noexcept {
    return kQKV_TO_CONTEXT_IXRT_PLUGIN_NAME;
}

char const* QKVToContextPluginDynamicCreator::getPluginVersion() const noexcept {
    return kQKV_TO_CONTEXT_IXRT_PLUGIN_VERSION;
}

PluginFieldCollection const* QKVToContextPluginDynamicCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* QKVToContextPluginDynamicCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept {
    try {
        gLogInfo << "Creating QKV2ContextPlugin..." << endl;
        IXRT_PLUGIN_ASSERT(fc != nullptr);
        int32_t hiddenSize = 0;
        // Since numHeads must always exist or validateRequiredAttributes will fail,
        // we can set numHeads to -1 so that static analysis tools don't warn about
        // a division by zero in QKVToContextPluginDynamic constructor.
        int32_t numHeads{-1};
        bool hasMask = false;
        int32_t typeId = -1;

        float dqProbs = -1;

        IXRT_PLUGIN_ASSERT(fc->fields != nullptr);
        ixrt_plugin::validateRequiredAttributesExist({"type_id", "hidden_size", "num_heads", "has_mask"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++) {
            IXRT_PLUGIN_ASSERT(fc->fields[i].name != nullptr);
            IXRT_PLUGIN_ASSERT(fc->fields[i].data != nullptr);
            std::string field_name(fc->fields[i].name);

            if (field_name.compare("type_id") == 0) {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
                IXRT_PLUGIN_CHECK_VALUE(typeId >= 0 && typeId <= 2,
                                        ("QKV: Invalid TypeId " + std::to_string(typeId)).c_str());
                gLogInfo << "Building typeId: " << typeId << endl;
            }
            if (field_name.compare("hidden_size") == 0) {
                hiddenSize = *static_cast<int32_t const*>(fc->fields[i].data);
                IXRT_PLUGIN_CHECK_VALUE(hiddenSize > 0,
                                        ("QKV: Invalid hiddenSize " + std::to_string(hiddenSize)).c_str());
                gLogInfo << "Building hiddenSize: " << hiddenSize << endl;
            }
            if (field_name.compare("num_heads") == 0) {
                numHeads = *static_cast<int32_t const*>(fc->fields[i].data);
                IXRT_PLUGIN_CHECK_VALUE(numHeads > 0, ("QKV: Invalid numHeads " + std::to_string(numHeads)).c_str());
                gLogInfo << "Building numHeads: " << numHeads << endl;
            }
            if (field_name.compare("has_mask") == 0) {
                auto hasMaskValue = *static_cast<int32_t const*>(fc->fields[i].data);
                IXRT_PLUGIN_CHECK_VALUE(hasMaskValue == 0 || hasMaskValue == 1,
                                        ("QKV: Invalid hasMask " + std::to_string(hasMaskValue)).c_str());
                hasMask = static_cast<bool>(hasMaskValue);
                gLogInfo << "Building hasMask: " << hasMask << endl;
            }
        }

        gLogInfo << "Building the Plugin..." << endl;
        auto type = static_cast<DataType>(typeId);
        auto* p = new QKVToContextPluginDynamic(name, type, hiddenSize, numHeads, dqProbs, hasMask);
        return p;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QKVToContextPluginDynamicCreator::deserializePlugin(char const* name, void const* serialData,
                                                               size_t serialLength) noexcept {
    // This object will be deleted when the network is destroyed, which will
    // call QKVToContextPluginDynamic::destroy()
    return new QKVToContextPluginDynamic(name, serialData, serialLength);
}

void QKVToContextPluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept {
    mNamespace = libNamespace;
}

char const* QKVToContextPluginDynamicCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// REGISTER_TENSORRT_PLUGIN(QKVToContextPluginDynamicCreator);
//#########################################################################//
QKVToContextPluginDynamic::QKVToContextPluginDynamic(const std::string name, const DataType type,
                                                     const int32_t hiddenSize, const int32_t numHeads,
                                                     float const dqProbs, bool hasImask)
    : mLayerName(name),
      mS(0),
      mB(0),
      mHeadSize(hiddenSize / numHeads),
      mHiddenSize(hiddenSize),
      mNumHeads(numHeads),
      mHasImask(hasImask),
      mType(type)

{
    //
}

QKVToContextPluginDynamic::QKVToContextPluginDynamic(const std::string name, void const* data, size_t length)
    : mLayerName(name) {
    gLogInfo << "QKV Deser Start" << endl;
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mNumHeads);
    deserialize_value(&data, &length, &mHeadSize);
    deserialize_value(&data, &length, &mHasImask);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mB);

    gLogInfo << "QKV Deser done" << endl;
}

// IPluginV2 Methods
char const* QKVToContextPluginDynamic::getPluginType() const noexcept { return kQKV_TO_CONTEXT_IXRT_PLUGIN_NAME; }

char const* QKVToContextPluginDynamic::getPluginVersion() const noexcept { return kQKV_TO_CONTEXT_IXRT_PLUGIN_VERSION; }

int32_t QKVToContextPluginDynamic::getNbOutputs() const noexcept { return 1; }

int32_t QKVToContextPluginDynamic::initialize() noexcept { return 0; }

void QKVToContextPluginDynamic::terminate() noexcept {}

size_t QKVToContextPluginDynamic::getSerializationSize() const noexcept {
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mHasImask) + sizeof(mHiddenSize) +
           sizeof(mS) + sizeof(mB);
}

void QKVToContextPluginDynamic::serialize(void* buffer) const noexcept {
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mNumHeads);
    serialize_value(&buffer, mHeadSize);
    serialize_value(&buffer, mHasImask);
    serialize_value(&buffer, mHiddenSize);
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mB);
}

void QKVToContextPluginDynamic::destroy() noexcept { delete this; }

void QKVToContextPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept { mNamespace = libNamespace; }

char const* QKVToContextPluginDynamic::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// IPluginV2Ext Methods
DataType QKVToContextPluginDynamic::getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                                      int32_t /*nbInputs*/) const noexcept {
    IXRT_PLUGIN_ASSERT(index == 0);
    IXRT_PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF ||
                       inputTypes[0] == DataType::kINT8);
    return inputTypes[0];
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextPluginDynamic::clone() const noexcept {
    gLogInfo << "QKV Clone" << endl;

    QKVToContextPluginDynamic* ret = nullptr;
    ret = new QKVToContextPluginDynamic(mLayerName, mType, mHiddenSize, mNumHeads, mDqProbs, mHasImask);

    ret->setPluginNamespace(mNamespace.c_str());
    gLogInfo << "QKV Clone done" << endl;
    return ret;
}

DimsExprs QKVToContextPluginDynamic::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs,
                                                         int32_t /*nbInputs*/, IExprBuilder& exprBuilder) noexcept {
    // Input is BxSx3*N*H, output should be BxSxN*H
    IXRT_PLUGIN_ASSERT(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[IIDX]);
    // Divide last dim by three
    auto const* three = exprBuilder.constant(3);
    output.d[HDIM] = exprBuilder.constant(mHiddenSize);
    return output;
}
bool QKVToContextPluginDynamic::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                                          int32_t /*nbOutputs*/) noexcept {
    IXRT_PLUGIN_ASSERT(pos >= 0);
    IXRT_PLUGIN_ASSERT(pos < 2 + mHasImask);
    IXRT_PLUGIN_ASSERT(nbInputs == 1 + mHasImask);
    auto const* in = inOut;
    auto const* out = inOut + nbInputs;

    if (pos == 0) {
        return (in->type == mType) && (in->format == TensorFormat::kLINEAR);
    }

    // pos==1
    if ((mHasImask && pos == 1))  // pos 1 is the mask
    {
        auto const* inMask = &inOut[1];

        // detect full mask and check that it was produced
        return (inMask->type == DataType::kINT32) &&       // precision
               (inMask->format == TensorFormat::kLINEAR);  // format
    }

    if (!mHasImask || pos == 2)  // output pos
    {
        return (in->type == out->type) && (out->format == TensorFormat::kLINEAR);
    }

    return false;
}
void QKVToContextPluginDynamic::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                                                DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    IXRT_PLUGIN_ASSERT(nbInputs == 1 + mHasImask);
    IXRT_PLUGIN_ASSERT(nbOutputs == 1);
    PluginTensorDesc const& inDesc = in[IIDX].desc;
    TRT_UNUSED inDesc;
    PluginTensorDesc const& outDesc = out->desc;
    TRT_UNUSED outDesc;
    IXRT_PLUGIN_ASSERT(mType == inDesc.type);
    IXRT_PLUGIN_ASSERT(mType == outDesc.type);
    IXRT_PLUGIN_ASSERT(inDesc.dims.nbDims == 5)
    IXRT_PLUGIN_ASSERT(inDesc.dims.d[HDIM] == 3 * mHiddenSize);
    IXRT_PLUGIN_ASSERT(inDesc.dims.d[3] == 1);
    IXRT_PLUGIN_ASSERT(inDesc.dims.d[4] == 1);
    if (mHasImask) {
        PluginTensorDesc const& maskDesc = in[MIDX].desc;
        TRT_UNUSED maskDesc;
        IXRT_PLUGIN_ASSERT(maskDesc.dims.nbDims == 2);
        IXRT_PLUGIN_ASSERT(maskDesc.dims.d[0] == inDesc.dims.d[0]);
        IXRT_PLUGIN_ASSERT(maskDesc.dims.d[1] == inDesc.dims.d[1]);
    }

    const int32_t S = inDesc.dims.d[SDIM];
    const int32_t B = inDesc.dims.d[BDIM] <= 0 ? in->max.d[BDIM] : inDesc.dims.d[BDIM];
    mS = S;
    mB = B;

    IXRT_PLUGIN_ASSERT(outDesc.dims.nbDims == 5);
    IXRT_PLUGIN_ASSERT(outDesc.dims.d[BDIM] == inDesc.dims.d[BDIM]);
    IXRT_PLUGIN_ASSERT(outDesc.dims.d[SDIM] == mS);
    IXRT_PLUGIN_ASSERT(outDesc.dims.d[HDIM] == mHiddenSize);
    IXRT_PLUGIN_ASSERT(outDesc.dims.d[3] == 1);
    IXRT_PLUGIN_ASSERT(outDesc.dims.d[4] == 1);
#ifdef __ILUVATAR__
    CUINFER_CHECK(cuinferCreate(&cuinfer_handle));
#else
    CHECK_GPU_ERROR(cublasLtCreate(&blaslt_handle));
#endif
}

size_t QKVToContextPluginDynamic::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                                                   PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept {
    const int32_t B = inputs->dims.d[BDIM];
    const int32_t S = inputs->dims.d[SDIM];
    const int32_t E = inputs->dims.d[2];
    int32_t fmha_S = S;
    int64_t buffer_size = B * fmha_S * E;
#ifndef __ILUVATAR__
    buffer_size += B * S * S * mNumHeads;
#endif
    return 4 * buffer_size * sizeof(mType);
}

inline void print_element(half* x, int num, string name) {
    printf("%s: \n", name.c_str());
    half* out = (half*)malloc(num * sizeof(half));
    cudaMemcpy(out, x, num * sizeof(half), cudaMemcpyDeviceToHost);
    for (auto i = 0; i < num; i++) {
        printf("%f\n", __half2float(out[i]));
    }
    printf("\n");
}

int32_t QKVToContextPluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                           void const* const* inputs, void* const* outputs, void* workspace,
                                           cudaStream_t stream) noexcept {
    gLogInfo << "in QKVToContextPluginDynamic.." << endl;
    int32_t S = inputDesc->dims.d[SDIM];
    int32_t B = inputDesc->dims.d[BDIM];
    int32_t status = STATUS_SUCCESS;
#ifdef __ILUVATAR__
    CUINFER_CHECK(cuinferSetStream(cuinfer_handle, stream));
#endif

    try {
        if (mType != DataType::kHALF) {
            gLogError << "embLayerNormPlugin infer type{" << int(mType) << "} not supported!" << endl;
            return STATUS_NOT_SUPPORTED;
        }
        half* qkv_buffer_ = (half*)inputs[0];
        half* qkv_out_ = (half*)outputs[0];
        // [B, fmha_S]
        int32_t* mask_ = mHasImask ? (int32_t*)inputs[1] : nullptr;
        int fmha_seq_len = S;

        int64_t buffer_size = B * fmha_seq_len * mHiddenSize;
        half* q_buffer_ = reinterpret_cast<half*>(workspace);
        half* k_buffer_ = q_buffer_ + buffer_size;
        half* v_buffer_ = k_buffer_ + buffer_size;
        

        // [B, S, 3*E, 1, 1] [B, fmha_S]
#ifdef __ILUVATAR__
        auto status =
            fused_multihead_attetion(qkv_buffer_, mask_, q_buffer_, k_buffer_, v_buffer_, qkv_out_, B, mHeadSize,
                                     mNumHeads, mHiddenSize, S, fmha_seq_len, cuinfer_handle, stream);
#else    
        half* qk_out_ = v_buffer_ + buffer_size;
        auto status =
            fused_multihead_attetion(qkv_buffer_, mask_, q_buffer_, k_buffer_, v_buffer_, qk_out_, qkv_out_, B, mHeadSize,
                                     mNumHeads, mHiddenSize, S, fmha_seq_len, blaslt_handle, stream);
#endif
        if (status != cudaSuccess) {
            return STATUS_FAILURE;
        }
        return STATUS_SUCCESS;

    } catch (std::exception const& e) {
        caughtError(e);
        return STATUS_FAILURE;
    }
}
