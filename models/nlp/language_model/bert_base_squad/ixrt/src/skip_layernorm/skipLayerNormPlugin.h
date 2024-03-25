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
#include <string>
#include <vector>

#include "NvInferRuntime.h"
#include "bertCommon.h"

namespace nvinfer1::ixrt_plugin {
namespace bert {

template <typename T, bool has_bias>
int32_t computeSkipLayerNorm(cudaStream_t stream, int32_t E, int32_t volume, const T* input, const T* gamma, const T* beta, const T* bias, T* skip, T* output);

void IxinferResidualBiasLn(const half *input, const half *scale, const half *bias, const half *residual_bias,
                           half *output, half *residual, int batch_tokens, int hidden_size, cudaStream_t stream,
                           bool is_post_ln);

void IxinferResidualBiasLnPad(const half *input, const half *scale, const half *bias, const half *residual_bias,
                              half *output, half *residual, int batch_tokens, int hidden_size, cudaStream_t stream,
                              bool is_post_ln);
class SkipLayerNormPluginDynamic : public IPluginV2DynamicExt {
   public:
    SkipLayerNormPluginDynamic(const std::string name, const nvinfer1::DataType type, int32_t const ld,
        nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma, nvinfer1::Weights const& bias);
    SkipLayerNormPluginDynamic(const std::string &name, void const* data, size_t length);
    SkipLayerNormPluginDynamic() noexcept = delete;
    ~SkipLayerNormPluginDynamic() override = default;

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
    cuda_unique_ptr<void> mGammaDev;
    cuda_unique_ptr<void> mBetaDev;
    WeightsWithOwnership mGamma;
    WeightsWithOwnership mBeta;
    size_t mHiddenSize{};
    size_t mParamWordsize{};
    DataType mType;
    DataType mCfgType;
    // mCfgType is the dataType for beta, gamma bias weights, always fp16 or fp32
    // mType is the plugin IO datatype, can be int8
    
    bool mHasBias{};
    cuda_unique_ptr<void> mBiasDev;
    WeightsWithOwnership mBias;
};

class SkipLayerNormPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    SkipLayerNormPluginDynamicCreator();

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