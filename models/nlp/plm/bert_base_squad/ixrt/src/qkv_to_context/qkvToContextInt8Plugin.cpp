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
#include "qkvToContextInt8Plugin.h"

#include "NvInferRuntime.h"
#include "bertCommon.h"
#include "checkMacrosPlugin.h"
#include "driver_types.h"
#include "plugin.h"
#include "serialize.h"
#include <iomanip>

using namespace nvinfer1;
using namespace nvinfer1::ixrt_plugin;
using namespace nvinfer1::ixrt_plugin::bert;

namespace {
char const* const kQKV_TO_CONTEXT_INT8_IXRT_PLUGIN_VERSION{"3"};
char const* const kQKV_TO_CONTEXT_INT8_IXRT_PLUGIN_NAME{"CustomQKVToContextPluginDynamic_IxRT"};
}  // namespace

PluginFieldCollection QKVToContextInt8PluginDynamicCreator::mFC{};
std::vector<PluginField> QKVToContextInt8PluginDynamicCreator::mPluginAttributes;

constexpr uint32_t IIDX = 0;  // index of the input tensor
constexpr uint32_t MIDX = 1;  // index of the mask
/*
dq_probs:
_arrange_qkv_amax
_softmax_in_amax
_softmax_out_amax
*/
QKVToContextInt8PluginDynamicCreator::QKVToContextInt8PluginDynamicCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dq_probs", nullptr, PluginFieldType::kFLOAT32, 3));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QKVToContextInt8PluginDynamicCreator::getPluginName() const noexcept {
    return kQKV_TO_CONTEXT_INT8_IXRT_PLUGIN_NAME;
}

char const* QKVToContextInt8PluginDynamicCreator::getPluginVersion() const noexcept {
    return kQKV_TO_CONTEXT_INT8_IXRT_PLUGIN_VERSION;
}

PluginFieldCollection const* QKVToContextInt8PluginDynamicCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* QKVToContextInt8PluginDynamicCreator::createPlugin(char const* name,
                                                              PluginFieldCollection const* fc) noexcept {
    try {
        int32_t hiddenSize = 0;
        // Since numHeads must always exist or validateRequiredAttributes will fail,
        // we can set numHeads to -1 so that static analysis tools don't warn about
        // a division by zero in QKVToContextInt8PluginDynamic constructor.
        int32_t numHeads{-1};

        vector<float> dqProbs;

        ixrt_plugin::validateRequiredAttributesExist({"hidden_size", "num_heads"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++) {
            std::string field_name(fc->fields[i].name);

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
            if (field_name.compare("dq_probs") == 0) {
                IXRT_PLUGIN_CHECK_VALUE(fc->fields[i].length > 0,
                                        ("QKV: dpProbs can not be empty, error: [dpProbs.length == 0]!"));
                gLogInfo << "Building dqProbs: [";
                for (auto j = 0; j < fc->fields[i].length; j++) {
                    dqProbs.emplace_back(static_cast<float const*>((fc->fields[i].data))[j]);
                    gLogInfo << std::setprecision(5) << dqProbs[j];
                }
                gLogInfo << "]" << endl;
            }
        }

        QKVToContextInt8PluginDynamic* p = new QKVToContextInt8PluginDynamic(name, hiddenSize, numHeads, dqProbs);
        return p;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QKVToContextInt8PluginDynamicCreator::deserializePlugin(char const* name, void const* serialData,
                                                                   size_t serialLength) noexcept {
    try {
        // This object will be deleted when the network is destroyed, which will
        // call QKVToContextInt8PluginDynamic::destroy() noexcept
        return new QKVToContextInt8PluginDynamic(name, serialData, serialLength);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void QKVToContextInt8PluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept {
    mNamespace = libNamespace;
}

char const* QKVToContextInt8PluginDynamicCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// REGISTER_TENSORRT_PLUGIN(QKVToContextInt8PluginDynamicCreator);
//#########################################################################//
QKVToContextInt8PluginDynamic::QKVToContextInt8PluginDynamic(std::string const& name, int32_t const hiddenSize,
                                                             int32_t const numHeads, vector<float> const dqProbs)
    : mLayerName(name),
      mS(0),
      mB(0),
      mHeadSize(hiddenSize / numHeads),
      mHiddenSize(hiddenSize),
      mNumHeads(numHeads),
      mDqProbs(dqProbs) {}

QKVToContextInt8PluginDynamic::QKVToContextInt8PluginDynamic(std::string const& name, void const* data, size_t length)
    : mLayerName(name) {
    gLogInfo << "deserialize QKVToContextInt8PluginDynamic" << endl;
    deserialize_value(&data, &length, &mNumHeads);
    deserialize_value(&data, &length, &mHeadSize);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mDqProbs);
}

// IPluginV2 Methods
char const* QKVToContextInt8PluginDynamic::getPluginType() const noexcept {
    return kQKV_TO_CONTEXT_INT8_IXRT_PLUGIN_NAME;
}

char const* QKVToContextInt8PluginDynamic::getPluginVersion() const noexcept {
    return kQKV_TO_CONTEXT_INT8_IXRT_PLUGIN_VERSION;
}

int32_t QKVToContextInt8PluginDynamic::getNbOutputs() const noexcept { return 1; }

int32_t QKVToContextInt8PluginDynamic::initialize() noexcept { return 0; }

void QKVToContextInt8PluginDynamic::terminate() noexcept {}

size_t QKVToContextInt8PluginDynamic::getSerializationSize() const noexcept {
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mHiddenSize) + mDqProbs.size() * sizeof(float) +
           sizeof(mDqProbs.size());
}

void QKVToContextInt8PluginDynamic::serialize(void* buffer) const noexcept {
    serialize_value(&buffer, mNumHeads);
    serialize_value(&buffer, mHeadSize);
    serialize_value(&buffer, mHiddenSize);
    serialize_value(&buffer, mDqProbs);
}

void QKVToContextInt8PluginDynamic::destroy() noexcept { delete this; }

void QKVToContextInt8PluginDynamic::setPluginNamespace(char const* libNamespace) noexcept { mNamespace = libNamespace; }

char const* QKVToContextInt8PluginDynamic::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// IPluginV2Ext Methods
DataType QKVToContextInt8PluginDynamic::getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                                          int32_t nbInputs) const noexcept {
    IXRT_PLUGIN_ASSERT(index == 0)
    return DataType::kINT8;
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextInt8PluginDynamic::clone() const noexcept {
    try {
        QKVToContextInt8PluginDynamic* ret =
            new QKVToContextInt8PluginDynamic(mLayerName, mHiddenSize, mNumHeads, mDqProbs);

        ret->setPluginNamespace(mNamespace.c_str());
        return ret;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs QKVToContextInt8PluginDynamic::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs,
                                                             int32_t nbInputs, IExprBuilder& exprBuilder) noexcept {
    // input [B, S, 3*E] int8
    // pad_mask [B, S] int8
    
    // output [B, S, E] int8
    IXRT_PLUGIN_ASSERT(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[IIDX]);
    // Divide last dim by three
    auto const* three = exprBuilder.constant(3);
    output.d[HDIM] = exprBuilder.constant(mHiddenSize);
    return output;
}
bool QKVToContextInt8PluginDynamic::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut,
                                                              int32_t nbInputs, int32_t nbOutputs) noexcept {
    IXRT_PLUGIN_ASSERT(nbInputs == 2);
    IXRT_PLUGIN_ASSERT(nbOutputs == 1);
    return (inOut[pos].type == DataType::kINT8) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void QKVToContextInt8PluginDynamic::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                                                    DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    IXRT_PLUGIN_ASSERT(nbInputs == 2);
    IXRT_PLUGIN_ASSERT(nbOutputs == 1);
    PluginTensorDesc const& inDesc = in[IIDX].desc;
    PluginTensorDesc const& outDesc = out[0].desc;
    IXRT_PLUGIN_ASSERT(inDesc.dims.nbDims == 5)
    IXRT_PLUGIN_ASSERT(inDesc.dims.d[HDIM] == 3 * mHiddenSize);
    IXRT_PLUGIN_ASSERT(inDesc.dims.d[3] == 1);
    IXRT_PLUGIN_ASSERT(inDesc.dims.d[4] == 1);

    PluginTensorDesc const& maskDesc = in[MIDX].desc;
    IXRT_PLUGIN_ASSERT(maskDesc.dims.nbDims == 2);
    IXRT_PLUGIN_ASSERT(maskDesc.dims.d[0] == inDesc.dims.d[0]);
    IXRT_PLUGIN_ASSERT(maskDesc.dims.d[1] == inDesc.dims.d[1]);

    const int32_t S = inDesc.dims.d[SDIM];

    IXRT_PLUGIN_ASSERT(outDesc.dims.nbDims == 5);
    IXRT_PLUGIN_ASSERT(outDesc.dims.d[BDIM] == inDesc.dims.d[BDIM]);
    IXRT_PLUGIN_ASSERT(outDesc.dims.d[SDIM] == S);
    IXRT_PLUGIN_ASSERT(outDesc.dims.d[HDIM] == mHiddenSize);
    IXRT_PLUGIN_ASSERT(outDesc.dims.d[3] == 1);
    IXRT_PLUGIN_ASSERT(outDesc.dims.d[4] == 1);

#ifdef __ILUVATAR__
    CUINFER_CHECK(cuinferCreate(&cuinfer_handle));
#else
    CHECK_GPU_ERROR(cublasLtCreate(&blaslt_handle));
#endif
}

size_t QKVToContextInt8PluginDynamic::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                                                       PluginTensorDesc const* outputs,
                                                       int32_t nbOutputs) const noexcept {
    const int32_t B = inputs[0].dims.d[BDIM];
    const int32_t S = inputs->dims.d[SDIM];
    const int32_t E = inputs->dims.d[HDIM];
    IXRT_PLUGIN_ASSERT(E == 3 * mHiddenSize);
    int64_t buffer_size = B * S * E * sizeof(int8_t) + B * S * S * mNumHeads * sizeof(int8_t);
#ifndef __ILUVATAR__
    buffer_size += B * S * S * mNumHeads * sizeof(int32_t);
#endif
    return buffer_size;
}

int32_t QKVToContextInt8PluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                               void const* const* inputs, void* const* outputs, void* workspace,
                                               cudaStream_t stream) noexcept {
    try {
#ifdef __ILUVATAR__
        CUINFER_CHECK(cuinferSetStream(cuinfer_handle, 0));
#endif
        int32_t const B = inputDesc[0].dims.d[BDIM];
        int32_t const S = inputDesc[0].dims.d[SDIM];

        float qkv_out_amax_ = inputDesc[0].scale * 127;
        float linear_in_amax_ = outputDesc[0].scale * 127;
        float arrange_qkv_amax_ = mDqProbs[0];
        float softmax_in_amax_ = mDqProbs[1];
        float softmax_out_amax_ = mDqProbs[2];

        int8_t* qkv_buffer_ = (int8_t*)inputs[0];
        int8_t* qkv_out_ = (int8_t*)outputs[0];
        int8_t* mask_ = (int8_t*)inputs[1];

        int64_t buffer_size = B * S * mHiddenSize;
        int64_t buffer_size2 = B * S * S * mNumHeads;
        int8_t* q_buffer_ = static_cast<int8_t*>(workspace);
        int8_t* k_buffer_ = q_buffer_ + buffer_size;
        int8_t* v_buffer_ = k_buffer_ + buffer_size;
        int8_t* qk_buffer_ = v_buffer_ + buffer_size;
        
#ifdef __ILUVATAR__
        auto status =
            fused_multihead_attetion_int8(qkv_buffer_, mask_, q_buffer_, k_buffer_, v_buffer_, qkv_out_,
                                          qk_buffer_, B, S, mHeadSize, mNumHeads, mHiddenSize, arrange_qkv_amax_,
                                          softmax_in_amax_, softmax_out_amax_, linear_in_amax_, cuinfer_handle, stream);
#else
        int32_t* qk_out_ = reinterpret_cast<int32_t*>(qk_buffer_ + buffer_size2);
        auto status =
            fused_multihead_attetion_int8(qkv_buffer_, mask_, q_buffer_, k_buffer_, v_buffer_, qk_out_, qkv_out_,
                                          qk_buffer_, B, S, mHeadSize, mNumHeads, mHiddenSize, arrange_qkv_amax_,
                                          softmax_in_amax_, softmax_out_amax_, linear_in_amax_, blaslt_handle, stream);
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
