
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
#include <memory>
#include <vector>

#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "bertCommon.h"
#include "driver_types.h"

#ifdef __ILUVATAR__
#include "backend/ixinfer/ixinfer_gemm_helper.h"
#else
#include "backend/cublas/cublas_helper.h"
#endif

namespace nvinfer1 {
namespace ixrt_plugin {
namespace bert {

void quantGemm(int32_t* input, int8_t* output, int batch_seq_len, int hidden_size, float dequant_scale,
               cudaStream_t stream);

void dequantGemmWithBias(int32_t* input, float* bias, int8_t* output, int batch_seq_len, int hidden_size,
                         float dequant_scale1, float dequant_scale2, float quant_scale, cudaStream_t stream);

void dequantGemmWithBias(int8_t* input, float* bias, int8_t* output, int batch_seq_len, int hidden_size,
                         float dequant_scale, float quant_scale, cudaStream_t stream);

void dequantGemmWithoutBias(int8_t* input, int8_t* output, int batch_seq_len, int hidden_size, float dequant_scale,
                            float quant_scale, cudaStream_t stream);

class FCPluginDynamic : public nvinfer1::IPluginV2DynamicExt {
   public:
    FCPluginDynamic(std::string const name, nvinfer1::DataType const type, int32_t const outDim,
                    nvinfer1::Weights const& W, nvinfer1::Weights const& B);

    FCPluginDynamic(std::string const name, void const* data, size_t length);

    // It doesn't make sense to make FCPluginDynamic without arguments, so we
    // delete default constructor.
    FCPluginDynamic() = delete;

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
    size_t mNumBias;

    bert::WeightsWithOwnership mW;
    bert::cuda_unique_ptr<void> mWdev;
    bert::WeightsWithOwnership mB;
    bert::cuda_unique_ptr<void> mBdev;

#ifdef __ILUVATAR__
    cuinferHandle_t cuinfer_handle;
#else
    cublasLtHandle_t blaslt_handle;
#endif
    cudaStream_t stream;
};

class FCPluginDynamicCreator : public nvinfer1::IPluginCreator {
   public:
    FCPluginDynamicCreator();

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

class FCInt8PluginDynamic : public nvinfer1::IPluginV2DynamicExt {
   public:
    FCInt8PluginDynamic(std::string const name, nvinfer1::DataType const type, int32_t const outDim,
                        nvinfer1::Weights const& W, nvinfer1::Weights const& Bias, vector<float> const& scale);

    FCInt8PluginDynamic(std::string const name, void const* data, size_t length);

    // It doesn't make sense to make FCInt8PluginDynamic without arguments, so we
    // delete default constructor.
    FCInt8PluginDynamic() = delete;

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

class FCInt8PluginDynamicCreator : public nvinfer1::IPluginCreator {
   public:
    FCInt8PluginDynamicCreator();

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
}  // namespace ixrt_plugin
}  // namespace nvinfer1
