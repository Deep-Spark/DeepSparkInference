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

#include <cuda_runtime_api.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"

namespace nvinfer1::plugin {

class YoloxDecoderPlugin : public IPluginV2DynamicExt {
   public:
    YoloxDecoderPlugin(int32_t num_class, int32_t stride, int32_t faster_impl);
    YoloxDecoderPlugin(void const *data, size_t length);
    YoloxDecoderPlugin() noexcept = delete;
    ~YoloxDecoderPlugin() override = default;

    // IPluginV2 methods
    char const *getPluginType() const noexcept override;
    char const *getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void *buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const *libNamespace) noexcept override;
    char const *getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    DataType getOutputDataType(int32_t index, DataType const *inputType, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt *clone() const noexcept override;
    DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs,
                                  IExprBuilder &exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out,
                         int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs, PluginTensorDesc const *outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs,
                    void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

   private:
    std::string mNameSpace{};
    // from attributes:
    int32_t nb_classes_;
    int32_t stride_;
    int32_t faster_impl_;
};

class YoloxDecodePluginCreator : public IPluginCreator {
   public:
    YoloxDecodePluginCreator();

    ~YoloxDecodePluginCreator() override = default;

    char const *getPluginName() const noexcept override;

    char const *getPluginVersion() const noexcept override;

    PluginFieldCollection const *getFieldNames() noexcept override;

    IPluginV2DynamicExt *createPlugin(char const *name, PluginFieldCollection const *fc) noexcept override;

    IPluginV2DynamicExt *deserializePlugin(char const *name, void const *serialData,
                                           size_t serialLength) noexcept override;

    void setPluginNamespace(char const *pluginNamespace) noexcept override;
    char const *getPluginNamespace() const noexcept override;

   private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

}  // namespace nvinfer1::plugin
