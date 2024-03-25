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
#include <cublasLt.h>
#include "NvInferRuntime.h"
#include "bertCommon.h"
#include <string>
#include <vector>
#ifdef __ILUVATAR__
#include "ixinfer.h"
#endif

namespace nvinfer1::ixrt_plugin
{
namespace bert
{

#ifdef __ILUVATAR__
cudaError_t fused_multihead_attetion_int8(int8_t* qkv_buffer, int8_t* mask, int8_t* q_buffer, int8_t* k_buffer,
                                          int8_t* v_buffer, int8_t* qkv_out, int8_t* qk_buffer,
                                          int batch_size, int batch_seq_len, int head_dim, int head_num,
                                          int hidden_size, float arrange_qkv_amax, float softmax_in_amax,
                                          float softmax_out_amax, float linear_in_amax, cuinferHandle_t& cuinfer_handle,
                                          cudaStream_t& stream);
#else
cudaError_t fused_multihead_attetion_int8(int8_t* qkv_buffer, int8_t* mask, int8_t* q_buffer, int8_t* k_buffer,
                                          int8_t* v_buffer, int32_t* qk_out, int8_t* qkv_out, int8_t* qk_buffer,
                                          int batch_size, int batch_seq_len, int head_dim, int head_num,
                                          int hidden_size, float arrange_qkv_amax, float softmax_in_amax,
                                          float softmax_out_amax, float linear_in_amax,
                                          cublasLtHandle_t& cublas_lt_handle, cudaStream_t& stream);
#endif

void IxinferCorrelationSoftmaxEncselfI8II8O(int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
                                            int8_t *correlation, const int8_t *src_padding_mask, float quant_scale,
                                            float dequant_scale);

void IxinferArrangeAttenOutputI8II8O(int batch_token_num, int hidden_size, cudaStream_t stream, const int8_t *ori_q,
                                     int8_t *new_q, int beam_size, int dim_per_head, int head_num,
                                     int max_thread_per_block, float quant_scale, float dequant_scale);
class QKVToContextInt8PluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextInt8PluginDynamic(std::string const& name, int32_t const hiddenSize, int32_t const numHeads,
        vector<float> const dqProbs);

    QKVToContextInt8PluginDynamic(std::string const& name, void const* data, size_t length);

    // It doesn't make sense to make QKVToContextInt8PluginDynamic without arguments, so we
    // delete default constructor.
    QKVToContextInt8PluginDynamic() = delete;

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

protected:
    void createMHARunner() noexcept;
    int32_t getSMVersion() const noexcept;

private:
    std::string const& mLayerName;
    std::string mNamespace;

    int32_t mS;
    int32_t mB;
    int32_t mSM;
    int32_t mHeadSize;
    int32_t mHiddenSize;
    int32_t mNumHeads;

    cuda_unique_ptr<half> mQkvBias;

    vector<float> mDqProbs;
    bool mUseInt8ScaleMax{true};

#ifdef __ILUVATAR__
    cuinferHandle_t cuinfer_handle;
#else
    cublasLtHandle_t blaslt_handle;
#endif
};

class QKVToContextInt8PluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextInt8PluginDynamicCreator();

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
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace bert
} // namespace nvinfer1::ixrt_plugin