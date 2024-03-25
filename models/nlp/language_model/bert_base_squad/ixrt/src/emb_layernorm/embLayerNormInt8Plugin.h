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
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "bertCommon.h"

namespace nvinfer1::ixrt_plugin {
namespace bert {

void IxinferBertEmbedLn(const float *token_emb, const float *pos_emb, const float *type_emb, const int *tokens, float *output,
                        int *pad_mask, int *type_ids, int pad_id, int batch_size, int seq_len, int hidden_size,
                        const float *scale, const float *bias, cudaStream_t stream);

cudaError_t embLayerNorm(cudaStream_t stream, int E, int B, int S, int32_t const* inputIds, int32_t const* segmentIds,
    float const* beta, float const* gamma, float const* wordEmb, float const* posEmb, float const* tokEmb, int32_t const wordSize,
    int32_t const tokSize, float* buffer, int8_t* output, int32_t* maskIdx, int32_t padId, float token_embed_amax_);

void IxinferMaskPad(int32_t* mask, int8_t* new_mask, int bsz, int ori_seq_len, int hsz,
                   int fmha_seq_len, int batch_tokens, cudaStream_t stream);

class EmbLayerNormInt8PluginDynamic : public IPluginV2DynamicExt {
   public:
    EmbLayerNormInt8PluginDynamic(std::string const& name, nvinfer1::DataType const type, nvinfer1::DataType const mhaType,
        nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma, nvinfer1::Weights const& word_emb,
        nvinfer1::Weights const& pos_emb, nvinfer1::Weights const& tok_emb, bool const useFullMask, int32_t padId = 0);
    EmbLayerNormInt8PluginDynamic(std::string const& name, void const* data, size_t length);
    EmbLayerNormInt8PluginDynamic() noexcept = delete;
    ~EmbLayerNormInt8PluginDynamic() override = default;

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
    cuda_unique_ptr<float> mGammaDev;
    cuda_unique_ptr<float> mBetaDev;
    cuda_unique_ptr<void> mWordEmbDev;
    cuda_unique_ptr<void> mTokEmbDev;
    cuda_unique_ptr<void> mPosEmbDev;
    // cuda_unique_ptr<int32_t> mNewMask;
    WeightsWithOwnership mBeta;
    WeightsWithOwnership mGamma;
    WeightsWithOwnership mWordEmb;
    WeightsWithOwnership mTokEmb;
    WeightsWithOwnership mPosEmb; 
};

class EmbLayerNormInt8PluginDynamicCreator : public IPluginCreator {
   public:
    EmbLayerNormInt8PluginDynamicCreator();

    ~EmbLayerNormInt8PluginDynamicCreator() override = default;

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
} //namespace nvinfer1::ixrt_plugin