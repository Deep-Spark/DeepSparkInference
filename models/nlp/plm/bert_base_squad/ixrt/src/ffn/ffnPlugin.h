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
#ifdef __ILUVATAR__
#include <ixinfer.h>
#endif

#include <memory>

#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "backend/cublas/cublas_helper.h"
#include "bertCommon.h"
#include <vector>

namespace nvinfer1::ixrt_plugin {
namespace bert {

class FFNPluginDynamic : public nvinfer1::IPluginV2DynamicExt {
   public:
    FFNPluginDynamic(std::string const name, nvinfer1::DataType const type, int32_t const outDim,
                     int32_t const out_type, nvinfer1::Weights const& W1, nvinfer1::Weights const& W2,
                     nvinfer1::Weights const& B1);

    FFNPluginDynamic(std::string const name, void const* data, size_t length);

    // It doesn't make sense to make FFNPluginDynamic without arguments, so we
    // delete default constructor.
    FFNPluginDynamic() = delete;

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
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
                         nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
                            nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;

   private:
    std::string const mLayerName;
    std::string mNamespace;

    nvinfer1::DataType mType;
    size_t mHiddenSize;
    size_t mActType;

    bert::WeightsWithOwnership mW1;
    bert::WeightsWithOwnership mB1;
    bert::WeightsWithOwnership mW2;
    bert::cuda_unique_ptr<void> mWdev1;
    bert::cuda_unique_ptr<void> mWdev2;
    bert::cuda_unique_ptr<void> mBdev1;

#ifdef __ILUVATAR__
    cuinferHandle_t cuinfer_handle;
#else
    cublasLtHandle_t blaslt_handle;
#endif
    cudaStream_t stream;
};

class FFNPluginDynamicCreator : public nvinfer1::IPluginCreator {
   public:
    FFNPluginDynamicCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(char const* name, void const* serialData,
                                           size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

   private:
    static nvinfer1::PluginFieldCollection mFFN;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class FFNInt8PluginDynamic : public nvinfer1::IPluginV2DynamicExt {
   public:
    FFNInt8PluginDynamic(std::string const name, nvinfer1::DataType const type, int32_t const outDim,
                         nvinfer1::Weights const& W, nvinfer1::Weights const& Bias, vector<float> const& scale);

    FFNInt8PluginDynamic(std::string const name, void const* data, size_t length);

    // It doesn't make sense to make FFNInt8PluginDynamic without arguments, so we
    // delete default constructor.
    FFNInt8PluginDynamic() = delete;

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
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
                         nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
                            nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;

   private:
    std::string const mLayerName;
    std::string mNamespace;

    nvinfer1::DataType mType;
    size_t mOutDim;  // leading dim
    size_t mNumParams;
    int32_t mNmax;
    int32_t mK;
    int32_t mNumBias;

    vector<float> mScale;

    bert::WeightsWithOwnership mW;
    bert::cuda_unique_ptr<void> mWdev;

    bert::WeightsWithOwnership mBias;
    bert::cuda_unique_ptr<void> mBiasdev;

#ifdef __ILUVATAR__
    cuinferHandle_t cuinfer_handle;
#else
    cublasLtHandle_t blaslt_handle;
#endif
    cudaStream_t stream;
};

class FFNInt8PluginDynamicCreator : public nvinfer1::IPluginCreator {
   public:
    FFNInt8PluginDynamicCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(char const* name, void const* serialData,
                                           size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

   private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

}  // namespace bert
}  // namespace nvinfer1::ixrt_plugin