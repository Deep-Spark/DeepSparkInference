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

#include <string>
#include <vector>
#include "NvInferRuntime.h"
#include "bertCommon.h"

namespace nvinfer1::ixrt_plugin {
namespace bert {


void skipLayerNormI8II8O(const int8_t *input, const  float *scale, const float *bias, const float *residual_bias, 
                       int8_t *output, float *residual, float* residual_out, int batch_tokens, int hidden_size, float dequant_scale,
                       float quant_scale, int max_thread_per_block, cudaStream_t stream,
                       bool is_post_ln);

void skipLayerNormI8IF32O(const int8_t *input, const  float *scale, const float *bias, const float *residual_bias,
                       float *output, float *residual, float* residual_out, int batch_tokens, int hidden_size, float dequant_scale,
                       float quant_scale, int max_thread_per_block, cudaStream_t stream,
                       bool is_post_ln);

class SkipLayerNormInt8PluginBase : public nvinfer1::IPluginV2DynamicExt
{
public:
    SkipLayerNormInt8PluginBase(
        std::string const& name, nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma, nvinfer1::Weights const& bias, bool output_fp32);

    SkipLayerNormInt8PluginBase(std::string const& name, void const* data, size_t length);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without
    // arguments, so we delete default constructor.
    SkipLayerNormInt8PluginBase() = delete;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt Methods
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

protected:
    std::string const& mLayerName;
    std::string mNamespace;

    bert::cuda_unique_ptr<void> mGammaDev;
    bert::cuda_unique_ptr<void> mBetaDev;
    size_t mLd{}; // leading dim
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mBeta;

    size_t mParamWordsize{};
    bool mParamsOnDevice{};
    bool mHasBias{};
    cuda_unique_ptr<void> mBiasDev;
    WeightsWithOwnership mBias;
    bool output_fp32{};
};

class SkipLayerNormInt8PluginHFace : public SkipLayerNormInt8PluginBase
{
public:
    SkipLayerNormInt8PluginHFace(
        std::string const& name, nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma, nvinfer1::Weights const& bias, bool output_fp32);

    SkipLayerNormInt8PluginHFace(std::string const& name, void const* data, size_t length);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without
    // arguments, so we delete default constructor.
    SkipLayerNormInt8PluginHFace() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 Methods
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    int32_t getNbOutputs() const noexcept override;
    char const* getPluginVersion() const noexcept override;
};

class SkipLayerNormInt8PluginBaseCreator : public nvinfer1::IPluginCreator
{
public:
    SkipLayerNormInt8PluginBaseCreator();

    char const* getPluginName() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class SkipLayerNormInt8PluginHFaceCreator : public SkipLayerNormInt8PluginBaseCreator
{
public:
    SkipLayerNormInt8PluginHFaceCreator();

    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
};

} // namespace bert
} // namespace nvinfer1::ixrt_plugin