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
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>


#include "bertCommon.h"

namespace nvinfer1::ixrt_plugin {
namespace bert {

cudaError embLayerNorm(cudaStream_t stream, int E, int B, int S, int32_t const* inputIds, int32_t const* segmentIds,
    half const* beta, half const* gamma, half const* wordEmb, half const* posEmb, half const* tokEmb, int32_t const wordSize,
    int32_t const tokSize, half* output, int32_t* maskIdx, int32_t padId);

void IxinferMaskPad(int32_t* mask, int32_t* new_mask, int bsz, int ori_seq_len, int hsz,
                   int fmha_seq_len, int batch_tokens, cudaStream_t stream);

void IxinferBertEmbedLn(const half *token_emb, const half *pos_emb, const half *type_emb, const int *tokens, half *output,
                        int *pad_mask, int *type_ids, int pad_id, int batch_size, int seq_len, int hidden_size,
                        const half *scale, const half *bias, cudaStream_t stream);;

class EmbLayerNormPluginDynamic : public IPluginV2DynamicExt {
   public:
    EmbLayerNormPluginDynamic(std::string const& name, nvinfer1::DataType const type, nvinfer1::DataType const mhaType,
        nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma, nvinfer1::Weights const& word_emb,
        nvinfer1::Weights const& pos_emb, nvinfer1::Weights const& tok_emb, bool const useFullMask, int32_t padId = 0);
    EmbLayerNormPluginDynamic(std::string const& name, void const* data, size_t length);
    EmbLayerNormPluginDynamic() noexcept = delete;
    ~EmbLayerNormPluginDynamic() override = default;

    // IPluginV2 methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    DataType getOutputDataType(int32_t index, DataType const* inputType, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs,
                                  IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
                         int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

   private:
    const std::string mLayerName;
    std::string mNamespace;
    size_t mHiddenSize;
    size_t mSeqLen;
    size_t mPadId;
    DataType mEmbType;
    bool mUseFullMask;
    DataType mMhaType;
    size_t mWordVocabSize, mPosVocabSize, mTokVocabSize;
    cuda_unique_ptr<half> mGammaDev;
    cuda_unique_ptr<half> mBetaDev;
    cuda_unique_ptr<void> mWordEmbDev;
    cuda_unique_ptr<void> mTokEmbDev;
    cuda_unique_ptr<void> mPosEmbDev;
    WeightsWithOwnership mBeta;
    WeightsWithOwnership mGamma;
    WeightsWithOwnership mWordEmb;
    WeightsWithOwnership mTokEmb;
    WeightsWithOwnership mPosEmb; 
};

class EmbLayerNormPluginDynamicCreator : public IPluginCreator {
   public:
    EmbLayerNormPluginDynamicCreator();

    ~EmbLayerNormPluginDynamicCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(char const* name, void const* serialData,
                                           size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

   private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;

};

} // namespace bert
} // namespace nvinfer1::ixrt_plugin