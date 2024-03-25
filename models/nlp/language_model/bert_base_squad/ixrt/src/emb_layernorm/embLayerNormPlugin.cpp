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
#include "embLayerNormPlugin.h"

#include "NvInferImpl.h"
#include "checkMacrosPlugin.h"
#include "common_def.cuh"
#include "driver_types.h"

#include "plugin.h"
#include "serialize.h"

using namespace nvinfer1;
using namespace nvinfer1::ixrt_plugin;
using namespace nvinfer1::ixrt_plugin::bert;

namespace {
char const* EMB_LAYER_NORM_VERSION{"1"};
char const* EMB_LAYER_NORM_NAME{"CustomEmbLayerNormPluginDynamic_IxRT"};
}  // namespace

// Static class fields initialization
PluginFieldCollection EmbLayerNormPluginDynamicCreator::mFC{};
std::vector<PluginField> EmbLayerNormPluginDynamicCreator::mPluginAttributes;

EmbLayerNormPluginDynamicCreator::EmbLayerNormPluginDynamicCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_layernorm_beta"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_layernorm_gamma"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_word_embeddings"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_token_type_embeddings"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_position_embeddings"));
    mPluginAttributes.emplace_back(PluginField("output_fp16"));
    mPluginAttributes.emplace_back(PluginField("full_mask"));
    mPluginAttributes.emplace_back(PluginField("mha_type_id"));
    mPluginAttributes.emplace_back(PluginField("pad_id"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* EmbLayerNormPluginDynamicCreator::getPluginName() const noexcept { return EMB_LAYER_NORM_NAME; }

char const* EmbLayerNormPluginDynamicCreator::getPluginVersion() const noexcept { return EMB_LAYER_NORM_VERSION; }

PluginFieldCollection const* EmbLayerNormPluginDynamicCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2DynamicExt* EmbLayerNormPluginDynamicCreator::createPlugin(char const* name,
                                                                    PluginFieldCollection const* fc) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(fc != nullptr);
        gLogInfo << "EmbLayerNormPluginDynamic createPlugin." << endl;
        std::set<std::string> const requiredAttributes{
            "bert_embeddings_layernorm_beta",      "bert_embeddings_layernorm_gamma",
            "bert_embeddings_word_embeddings",     "bert_embeddings_token_type_embeddings",
            "bert_embeddings_position_embeddings",
        };

        bool output_fp16 = false;
        bool useFullMask = false;
        Weights beta{};
        Weights gamma{};
        Weights word_emb{};
        Weights pos_emb{};
        Weights tok_emb{};
        int32_t mhaTypeId = 0;
        int32_t pad_id = 0;

        for (auto i = 0; i < fc->nbFields; i++) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("bert_embeddings_layernorm_beta") == 0) {
                gLogInfo << "Building bert_embeddings_layernorm_beta..." << endl;
                beta.values = fc->fields[i].data;
                beta.count = fc->fields[i].length;
                beta.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_layernorm_gamma") == 0) {
                gLogInfo << "Building bert_embeddings_layernorm_gamma..." << endl;
                gamma.values = fc->fields[i].data;
                gamma.count = fc->fields[i].length;
                gamma.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_word_embeddings") == 0) {
                gLogInfo << "Building bert_embeddings_word_embeddings..." << endl;
                word_emb.values = fc->fields[i].data;
                word_emb.count = fc->fields[i].length;
                word_emb.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_token_type_embeddings") == 0) {
                gLogInfo << "Building bert_embeddings_token_type_embeddings..." << endl;
                tok_emb.values = fc->fields[i].data;
                tok_emb.count = fc->fields[i].length;
                tok_emb.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_position_embeddings") == 0) {
                gLogInfo << "Building bert_embeddings_position_embeddings..." << endl;
                pos_emb.values = fc->fields[i].data;
                pos_emb.count = fc->fields[i].length;
                pos_emb.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("output_fp16") == 0) {
                IXRT_PLUGIN_ASSERT(fc->fields[i].type == PluginFieldType::kINT32);
                output_fp16 = static_cast<int32_t const*>(fc->fields[i].data)[0] != 0;
                gLogInfo << "Building output_fp16: " << output_fp16 << endl;
            }

            if (field_name.compare("full_mask") == 0) {
                IXRT_PLUGIN_ASSERT(fc->fields[i].type == PluginFieldType::kINT32);
                useFullMask = static_cast<int32_t const*>(fc->fields[i].data)[0] != 0;
                gLogInfo << "Building full_mask: " << useFullMask << endl;
            }

            if (field_name.compare("mha_type_id") == 0) {
                mhaTypeId = *static_cast<int32_t const*>(fc->fields[i].data);
                IXRT_PLUGIN_ASSERT(mhaTypeId >= 0 && mhaTypeId < 3);
                gLogInfo << "Building mha typeId: " << mhaTypeId << endl;
            }

            if (field_name.compare("pad_id") == 0) {
                IXRT_PLUGIN_ASSERT(fc->fields[i].type == PluginFieldType::kINT32)
                pad_id = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }
        gLogInfo << "Building EmbLayerNormPluginDynamic Plugin..." << endl;
        DataType mhaType = static_cast<DataType>(mhaTypeId);
        EmbLayerNormPluginDynamic* p =
            new EmbLayerNormPluginDynamic(name, output_fp16 ? DataType::kHALF : DataType::kFLOAT, mhaType, beta, gamma,
                                          word_emb, pos_emb, tok_emb, useFullMask, pad_id);

        return p;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* EmbLayerNormPluginDynamicCreator::deserializePlugin(char const* name, void const* serialData,
                                                                         size_t serialLength) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(serialData != nullptr);
        return new EmbLayerNormPluginDynamic(name, serialData, serialLength);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void EmbLayerNormPluginDynamicCreator::setPluginNamespace(char const* pluginNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(pluginNamespace != nullptr);
        mNamespace = pluginNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* EmbLayerNormPluginDynamicCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// REGISTER_TENSORRT_PLUGIN(EmbLayerNormPluginDynamicCreator);

//#########################################################################//
EmbLayerNormPluginDynamic::EmbLayerNormPluginDynamic(std::string const& name, DataType const type,
                                                     DataType const mhaType, Weights const& beta, Weights const& gamma,
                                                     Weights const& wordEmb, Weights const& posEmb,
                                                     Weights const& tokEmb, bool const useFullMask, int32_t padId)
    : mLayerName(name),
      mHiddenSize(beta.count),
      mEmbType(type),
      mUseFullMask(useFullMask),
      mMhaType(mhaType),
      mPadId(padId) {
    IXRT_PLUGIN_ASSERT(beta.count == gamma.count);
    IXRT_PLUGIN_ASSERT(mHiddenSize > 0U);
    IXRT_PLUGIN_ASSERT(wordEmb.count % mHiddenSize == 0);
    IXRT_PLUGIN_ASSERT(posEmb.count % mHiddenSize == 0);
    IXRT_PLUGIN_ASSERT(tokEmb.count % mHiddenSize == 0);
    mWordVocabSize = wordEmb.count / mHiddenSize;
    mPosVocabSize = posEmb.count / mHiddenSize;
    mTokVocabSize = tokEmb.count / mHiddenSize;

    mBeta.convertAndCopy(beta, nvinfer1::DataType::kHALF);
    mGamma.convertAndCopy(gamma, nvinfer1::DataType::kHALF);
    mWordEmb.convertAndCopy(wordEmb, mEmbType);
    mTokEmb.convertAndCopy(tokEmb, mEmbType);
    mPosEmb.convertAndCopy(posEmb, mEmbType);

    copyToDevice(mGamma, sizeof(half) * mGamma.count, mGammaDev);
    copyToDevice(mBeta, sizeof(half) * mBeta.count, mBetaDev);
    copyToDevice(mWordEmb, getWeightsSize(mWordEmb, mEmbType), mWordEmbDev);
    copyToDevice(mPosEmb, getWeightsSize(mPosEmb, mEmbType), mPosEmbDev);
    copyToDevice(mTokEmb, getWeightsSize(mTokEmb, mEmbType), mTokEmbDev);
}

EmbLayerNormPluginDynamic::EmbLayerNormPluginDynamic(std::string const& name, void const* data, size_t length)
    : mLayerName(name),
      mGammaDev(nullptr),
      mBetaDev(nullptr),
      mWordEmbDev(nullptr),
      mTokEmbDev(nullptr),
      mPosEmbDev(nullptr) {
    gLogInfo << "EmbLayerNormPluginDynamic deserialize." << endl;

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mEmbType);
    deserialize_value(&data, &length, &mMhaType);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mSeqLen);
    deserialize_value(&data, &length, &mPadId);
    deserialize_value(&data, &length, &mWordVocabSize);
    deserialize_value(&data, &length, &mPosVocabSize);
    deserialize_value(&data, &length, &mTokVocabSize);
    deserialize_value(&data, &length, &mUseFullMask);

    char const* d = static_cast<char const*>(data);
    mBeta.convertAndCopy(d, mHiddenSize, nvinfer1::DataType::kHALF);
    mGamma.convertAndCopy(d, mHiddenSize, nvinfer1::DataType::kHALF);
    mWordEmb.convertAndCopy(d, mHiddenSize * mWordVocabSize, mEmbType);
    mPosEmb.convertAndCopy(d, mHiddenSize * mPosVocabSize, mEmbType);
    mTokEmb.convertAndCopy(d, mHiddenSize * mTokVocabSize, mEmbType);

    copyToDevice(mGamma, sizeof(half) * mGamma.count, mGammaDev);
    copyToDevice(mBeta, sizeof(half) * mBeta.count, mBetaDev);
    copyToDevice(mWordEmb, getWeightsSize(mWordEmb, mEmbType), mWordEmbDev);
    copyToDevice(mPosEmb, getWeightsSize(mPosEmb, mEmbType), mPosEmbDev);
    copyToDevice(mTokEmb, getWeightsSize(mTokEmb, mEmbType), mTokEmbDev);
}

// IPluginV2 Methods
char const* EmbLayerNormPluginDynamic::getPluginType() const noexcept { return EMB_LAYER_NORM_NAME; }

char const* EmbLayerNormPluginDynamic::getPluginVersion() const noexcept { return EMB_LAYER_NORM_VERSION; }

int32_t EmbLayerNormPluginDynamic::getNbOutputs() const noexcept { return 2; }

int32_t EmbLayerNormPluginDynamic::initialize() noexcept { return 0; }

void EmbLayerNormPluginDynamic::terminate() noexcept {  gLogInfo << "EmbLayerNormPluginDynamic terminate." << endl; }

size_t EmbLayerNormPluginDynamic::getSerializationSize() const noexcept {
    size_t const wordSize = getElementSize(mEmbType);
    return sizeof(mEmbType) * 2                       // mEmbType, mMhaType
           + sizeof(mHiddenSize) * 6                  // mHiddenSize, mSeqLen, 3*VocabSize, mPadId
           + sizeof(mUseFullMask)                     // mask type
           + 2 * sizeof(half) * mHiddenSize           // beta + gamma
           + wordSize * mHiddenSize * mWordVocabSize  // word emb
           + wordSize * mHiddenSize * mPosVocabSize   // pos emb
           + wordSize * mHiddenSize * mTokVocabSize   // tok emb
        ;
}

void EmbLayerNormPluginDynamic::serialize(void* buffer) const noexcept {
    serialize_value(&buffer, mEmbType);
    serialize_value(&buffer, mMhaType);
    serialize_value(&buffer, mHiddenSize);
    serialize_value(&buffer, mSeqLen);
    serialize_value(&buffer, mPadId);
    serialize_value(&buffer, mWordVocabSize);
    serialize_value(&buffer, mPosVocabSize);
    serialize_value(&buffer, mTokVocabSize);
    serialize_value(&buffer, mUseFullMask);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, mBetaDev.get(), mHiddenSize);
    serFromDev(d, mGammaDev.get(), mHiddenSize);
    size_t const wordSize = getElementSize(mEmbType);
    serFromDev(d, static_cast<char*>(mWordEmbDev.get()), mHiddenSize * mWordVocabSize * wordSize);
    serFromDev(d, static_cast<char*>(mPosEmbDev.get()), mHiddenSize * mPosVocabSize * wordSize);
    serFromDev(d, static_cast<char*>(mTokEmbDev.get()), mHiddenSize * mTokVocabSize * wordSize);
}

void EmbLayerNormPluginDynamic::destroy() noexcept {
    gLogInfo << "EmbLayerNormPluginDynamic destroy." << endl;
    // This gets called when the network containing plugin is destroyed
    mGammaDev.reset(nullptr);
    mBetaDev.reset(nullptr);
    mWordEmbDev.reset(nullptr);
    mPosEmbDev.reset(nullptr);
    mTokEmbDev.reset(nullptr);
    delete this;
}

void EmbLayerNormPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* EmbLayerNormPluginDynamic::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// IPluginV2Ext Methods
DataType EmbLayerNormPluginDynamic::getOutputDataType(int32_t index, DataType const* inputTypes,
                                                      int32_t nbInputs) const noexcept {
    IXRT_PLUGIN_ASSERT(index == 0 || index == 1);
    if (index == 0) {
        IXRT_PLUGIN_ASSERT(mMhaType == DataType::kHALF || mMhaType == DataType::kFLOAT);
        return mMhaType;
    }
    return DataType::kINT32;
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* EmbLayerNormPluginDynamic::clone() const noexcept {
    try {
        gLogInfo << "EmbLayerNormPluginDynamic clone." << endl;

        auto p = new EmbLayerNormPluginDynamic(mLayerName, mEmbType, mMhaType, mBeta, mGamma, mWordEmb, mPosEmb,
                                               mTokEmb, mUseFullMask);
        p->mSeqLen = mSeqLen;
        p->setPluginNamespace(mNamespace.c_str());

        return p;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs EmbLayerNormPluginDynamic::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs,
                                                         IExprBuilder& exprBuilder) noexcept {
    try {
        // Input should be input ids and token ids and the input mask
        // Output should be the embeddings tensor and mask indices
        IXRT_PLUGIN_ASSERT(nbInputs == 3);

        IXRT_PLUGIN_ASSERT(inputs[0].nbDims == 2);  // BxS
        IXRT_PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);
        IXRT_PLUGIN_ASSERT(inputs[0].nbDims == inputs[2].nbDims);

        IXRT_PLUGIN_ASSERT(outputIndex == 0 || outputIndex == 1);

        if (outputIndex == 0) {
            DimsExprs ret;
            ret.nbDims = 5;
            ret.d[0] = inputs[0].d[0];
            ret.d[1] = inputs[0].d[1];
            ret.d[2] = exprBuilder.constant(mHiddenSize);
            ret.d[3] = exprBuilder.constant(1);
            ret.d[4] = exprBuilder.constant(1);
            return ret;
        }

        DimsExprs ret;
        ret.nbDims = 2;
        ret.d[0] = inputs[0].d[BDIM];
        ret.d[1] = inputs[0].d[SDIM];
        return ret;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return DimsExprs{};
}

bool EmbLayerNormPluginDynamic::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                                          int32_t nbOutputs) noexcept {
    // 3 inputs of size BxS
    IXRT_PLUGIN_ASSERT(nbInputs == 3);
    IXRT_PLUGIN_ASSERT(nbOutputs == 2);

    PluginTensorDesc const& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR) {
        return false;
    }
    if (pos == 0) {
        return desc.type == DataType::kINT32;
    }

    PluginTensorDesc const& prev = inOut[pos - 1];
    if (pos == 1 || pos == 2) {
        return desc.type == DataType::kINT32 && desc.format == prev.format;
    }

    // embedded sequence
    if (pos == 3) {
        return desc.type == mMhaType && desc.format == prev.format;
    }
    // mask
    return desc.type == ((mMhaType == DataType::kHALF) ? DataType::kINT32 : mMhaType);
}

void EmbLayerNormPluginDynamic::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                                DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept {
    gLogInfo << "EmbLayerNormPluginDynamic configurePlugin." << endl;

    // Validate input arguments
    IXRT_PLUGIN_ASSERT(nbOutputs == 2);
    IXRT_PLUGIN_ASSERT(nbInputs == 3);

    IXRT_PLUGIN_ASSERT(inputs[0].desc.dims.nbDims == 2);
    int32_t const S = inputs[0].desc.dims.d[SDIM];
    mSeqLen = S;
    int32_t const B = inputs[0].desc.dims.d[BDIM];
    TRT_UNUSED B;
    IXRT_PLUGIN_ASSERT(mSeqLen == static_cast<size_t>(inputs[1].desc.dims.d[SDIM]));
    IXRT_PLUGIN_ASSERT(B == inputs[1].desc.dims.d[BDIM]);
    IXRT_PLUGIN_ASSERT(mSeqLen == static_cast<size_t>(inputs[2].desc.dims.d[SDIM]));
    IXRT_PLUGIN_ASSERT(B == inputs[2].desc.dims.d[BDIM]);

    IXRT_PLUGIN_ASSERT(outputs[0].desc.dims.nbDims == 5);
    IXRT_PLUGIN_ASSERT(mSeqLen == outputs[0].desc.dims.d[SDIM])
    IXRT_PLUGIN_ASSERT(outputs[0].desc.dims.d[BDIM] == B);
    IXRT_PLUGIN_ASSERT(static_cast<size_t>(outputs[0].desc.dims.d[2]) == mHiddenSize);
    IXRT_PLUGIN_ASSERT(outputs[0].desc.dims.d[3] == 1);
    IXRT_PLUGIN_ASSERT(outputs[0].desc.dims.d[4] == 1);

    IXRT_PLUGIN_ASSERT(outputs[1].desc.dims.nbDims == 2);
    IXRT_PLUGIN_ASSERT(outputs[1].desc.dims.d[0] == B);
    IXRT_PLUGIN_ASSERT(outputs[1].desc.dims.d[1] == mSeqLen);
}

size_t EmbLayerNormPluginDynamic::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                                                   PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t EmbLayerNormPluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                           void const* const* inputs, void* const* outputs, void* workspace,
                                           cudaStream_t stream) noexcept {
    gLogInfo << "enqueue EmbLayerNormPluginDynamic.." << endl;
    try {
        int32_t const B = inputDesc->dims.d[BDIM];
        int32_t const S = inputDesc->dims.d[SDIM];
        int32_t status = STATUS_SUCCESS;
        int32_t fmha_S = S;
        int32_t batch_tokens = B * fmha_S;

        // Our plugin outputs only one tensor
        auto const inputIds = static_cast<int32_t const*>(inputs[0]);
        auto const segmentIds = static_cast<int32_t const*>(inputs[1]);

        half const* beta = mBetaDev.get();
        half const* gamma = mGammaDev.get();
        if (mMhaType == DataType::kFLOAT) {
            gLogError << "embLayerNormPlugin float type not supported!" << endl;
            return STATUS_NOT_SUPPORTED;
        } else if (mMhaType == DataType::kHALF) {
            auto output = static_cast<half*>(outputs[0]);
            auto mNewMask = static_cast<int32_t*>(outputs[1]);
            auto const wordEmb = static_cast<half const*>(mWordEmbDev.get());
            auto const tokEmb = static_cast<half const*>(mTokEmbDev.get());
            auto const posEmb = static_cast<half const*>(mPosEmbDev.get());

            status =
                embLayerNorm(stream, static_cast<int32_t>(mHiddenSize), B, S, inputIds, segmentIds, beta, gamma,
                                wordEmb, posEmb, tokEmb, mWordVocabSize, mTokVocabSize, output, mNewMask, mPadId);
            if (status != cudaSuccess) {
                return STATUS_FAILURE;
            }
        }
        else {
            gLogError << "Unsupported type error, expected [kHALF,kFLOAT], but received "
                      << static_cast<int32_t>(mMhaType) << endl;

            return STATUS_NOT_SUPPORTED;
        }

        return status;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return STATUS_FAILURE;
}
