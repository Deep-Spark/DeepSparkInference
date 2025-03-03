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

template <typename T>
void IxinferBiasGeluI8II8O(int batch_token_num, cudaStream_t stream, int8_t *input, int8_t *output, const T *bias,
                           int feature_dim, float dequant_scale, float quant_scale);

int32_t computeGelu(cudaStream_t stream, int32_t n, float const* input, float* output);

int32_t computeGelu(cudaStream_t stream, int32_t n, half const* input, half* output);

int32_t computeGeluI8O8(cudaStream_t stream, int n, const int8_t* input, int8_t* output, float dequant_scale,
                        float quant_scale);

int32_t computeGeluBias(float* output, float const* input, float const* bias, int32_t const ld, int32_t const cols,
                        cudaStream_t stream);

int32_t computeGeluBias(half* output, half const* input, half const* bias, int32_t const ld, int32_t const cols,
                        cudaStream_t stream);

int32_t computeGeluI8O8Bias(int8_t* output, const int8_t* input, const half* bias, const int ld, const int cols,
                            float dequant_scale, float quant_scale, cudaStream_t stream);

class GeluPluginDynamic : public nvinfer1::IPluginV2DynamicExt {
   public:
    GeluPluginDynamic(const std::string name, const nvinfer1::DataType type, nvinfer1::Weights const& bias,
                      const int ld);

    GeluPluginDynamic(const std::string name, void const* data, size_t length);

    // It doesn't make sense to make GeluPluginDynamic without arguments, so we delete
    // default constructor.
    GeluPluginDynamic() = delete;

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
    // Helper method for enqueue()
    template <typename TDataType>
    int32_t enqueueTyped(void const* input, void* output, int32_t const inputVolume, cudaStream_t stream) noexcept;
    int32_t enqueueInt8(void const* input_, void* output_, float dequant_scale, float quant_scale,
                        int32_t const inputVolume, cudaStream_t stream) noexcept;

    const std::string mLayerName;
    std::string mNamespace;

    nvinfer1::DataType mType;
    bert::WeightsWithOwnership mBias;
    bert::cuda_unique_ptr<void> mBiasDev;
    size_t mLd;
    size_t mNumBias;
};

class GeluPluginDynamicCreator : public nvinfer1::IPluginCreator {
   public:
    GeluPluginDynamicCreator();

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
}  // namespace nvinfer1::plugin
