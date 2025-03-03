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
#ifdef __ILUVATAR__
#include <ixinfer.h>
#endif
#include <vector>

#include "NvInferRuntime.h"
#include "bertCommon.h"


namespace nvinfer1::ixrt_plugin {
namespace bert {
#ifdef __ILUVATAR__
cudaError_t fused_multihead_attetion(half* qkv_buffer, int32_t* mask, 
                              half* q_buffer, half* k_buffer, half* v_buffer, half* qkv_out,
                              int bsz, int head_dim, int head_num, int hsz, int ori_seq_len, int fmha_seq_len,
                              cuinferHandle_t &cuinfer_handle, cudaStream_t &stream);
#else
cudaError_t fused_multihead_attetion(half* qkv_buffer, int32_t* mask, 
                              half* q_buffer, half* k_buffer, half* v_buffer, half* qk_out, half* qkv_out,
                              int bsz, int head_dim, int head_num, int hsz, int ori_seq_len, int fmha_seq_len,
                              cublasLtHandle_t &blaslt_handle, cudaStream_t &stream);
#endif

void IxinferArrangeEncQkv(half *ori_qkv, half *new_q, half *new_k, half *new_v, int bsz,
                          int head_num, int head_dim, int ori_seq_len, int fmha_seq_len, cudaStream_t stream);

void IxinferEncAttnOutArrange(half *ori_q, half *new_q, int bsz, int ori_seq_len, int fmha_seq_len, int head_num,
                              int head_dim, cudaStream_t stream);

void IxinferCorrelationSoftmaxEncself(int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
                                      half *correlation, const int *src_padding_mask);

class QKVToContextPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextPluginDynamic(const std::string name, const nvinfer1::DataType type, const int32_t hiddenSize,
        const int32_t numHeads, float const dqProbs, bool hasImask = false);

    QKVToContextPluginDynamic(const std::string name, void const* data, size_t length);

    // It doesn't make sense to make QKVToContextPluginDynamic without arguments, so we
    // delete default constructor.
    QKVToContextPluginDynamic() = delete;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    const std::string mLayerName;
    std::string mNamespace;

    int32_t mS;
    int32_t mB;
    int32_t mSM;
    int32_t mHeadSize;
    int32_t mHiddenSize;
    int32_t mNumHeads;
    bool mHasImask;
    nvinfer1::DataType mType;
    float mDqProbs;
#ifdef __ILUVATAR__
    cuinferHandle_t cuinfer_handle;
#else
    cublasLtHandle_t blaslt_handle;
#endif
    cudaStream_t stream;
    
    half* query_;
};

class QKVToContextPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextPluginDynamicCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace bert
} // namespace nvinfer1::ixrt_plugin