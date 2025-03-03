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
#include "yoloxDecoderPlugin.h"

#include "checkMacrosPlugin.h"
#include "plugin.h"
#include "serialize.h"
#include "yoloxDecoderKernel.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
char const *kYoloxDecoderPluginVersion{"1"};
char const *kYoloxDecoderPluginName{"YoloXDecoder"};
}  // namespace

PluginFieldCollection YoloxDecodePluginCreator::mFC{};
std::vector<PluginField> YoloxDecodePluginCreator::mPluginAttributes;

YoloxDecoderPlugin::YoloxDecoderPlugin(int32_t num_class, int32_t stride, int32_t faster_impl)
    : nb_classes_(num_class), stride_(stride), faster_impl_(faster_impl) {}

int32_t YoloxDecoderPlugin::getNbOutputs() const noexcept { return 1; }

int32_t YoloxDecoderPlugin::initialize() noexcept { return 0; }

void YoloxDecoderPlugin::terminate() noexcept {}

void YoloxDecoderPlugin::destroy() noexcept { delete this; }

size_t YoloxDecoderPlugin::getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs,
                                            PluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

bool YoloxDecoderPlugin::supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs,
                                                   int32_t nbOutputs) noexcept {
    IXRT_PLUGIN_ASSERT(inOut != nullptr);
    IXRT_PLUGIN_ASSERT(pos < 4);
    IXRT_PLUGIN_ASSERT(nbInputs == 3);
    IXRT_PLUGIN_ASSERT(nbOutputs == 1);

    bool condition = true;
    switch (pos) {
        case 0: {
            condition &= (inOut[pos].type == DataType::kINT8 || inOut[pos].type == DataType::kHALF);
            condition &= inOut[pos].format == TensorFormat::kLINEAR;
            break;
        }
        case 1: {
            condition &= (inOut[pos].type == DataType::kINT8 || inOut[pos].type == DataType::kHALF);
            condition &= inOut[pos].format == TensorFormat::kLINEAR;
            condition &= (inOut[0].type == inOut[1].type);
            break;
        }
        case 2: {
            condition &= (inOut[pos].type == DataType::kINT8 || inOut[pos].type == DataType::kHALF);
            condition &= inOut[pos].format == TensorFormat::kLINEAR;
            condition &= (inOut[0].type == inOut[2].type);
            break;
        }
        case 3: {
            condition &= inOut[pos].type == DataType::kHALF;
            condition &= inOut[pos].format == TensorFormat::kLINEAR;
            break;
        }
        default: {
            IXRT_PLUGIN_ASSERT(false);
        }
    }
    return condition;
}

char const *YoloxDecoderPlugin::getPluginType() const noexcept { return kYoloxDecoderPluginName; }

char const *YoloxDecoderPlugin::getPluginVersion() const noexcept { return kYoloxDecoderPluginVersion; }

IPluginV2DynamicExt *YoloxDecoderPlugin::clone() const noexcept {
    try {
        auto plugin = new YoloxDecoderPlugin(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    } catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

void YoloxDecoderPlugin::setPluginNamespace(char const *libNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(libNamespace != nullptr);
        mNameSpace = libNamespace;
    } catch (std::exception const &e) {
        caughtError(e);
    }
}

char const *YoloxDecoderPlugin::getPluginNamespace() const noexcept { return mNameSpace.c_str(); }

DimsExprs YoloxDecoderPlugin::getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs,
                                                  IExprBuilder &exprBuilder) noexcept {
    IXRT_PLUGIN_ASSERT(inputs != nullptr);
    IXRT_PLUGIN_ASSERT(nbInputs == 3);
    IXRT_PLUGIN_ASSERT(outputIndex == 0);  // there is only one output
    DimsExprs result;
    result.nbDims = 3;

    // n
    result.d[0] = inputs[0].d[0];
    // H*W*anchor_number
    result.d[1] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);
    // box info
    result.d[2] = exprBuilder.constant(6);
    return result;
}

int32_t YoloxDecoderPlugin::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc,
                                    void const *const *inputs, void *const *outputs, void *workspace,
                                    cudaStream_t stream) noexcept {
    IXRT_PLUGIN_ASSERT(inputDesc != nullptr);
    IXRT_PLUGIN_ASSERT(inputs != nullptr);
    IXRT_PLUGIN_ASSERT(outputs != nullptr);
    IXRT_PLUGIN_ASSERT(outputDesc != nullptr);

    auto type = inputDesc[0].type;
    float box_scale = 1.f, conf_scale = 1.f, class_scale = 1.f;
    if (type == DataType::kINT8) {
        box_scale = 1.f / inputDesc[0].scale;
        conf_scale = 1.f / inputDesc[1].scale;
        class_scale = 1.f / inputDesc[2].scale;
    }

    int N = inputDesc[0].dims.d[0], H = inputDesc[0].dims.d[1], W = inputDesc[0].dims.d[2],
        box_channel = inputDesc[0].dims.d[3], conf_channel = inputDesc[1].dims.d[3],
        class_channel = inputDesc[2].dims.d[3];
    return YoloxDecoderInference(stream, inputs[0], inputs[1], inputs[2], outputs[0], box_scale, conf_scale,
                                 class_scale, N, H, W, box_channel, conf_channel, class_channel, stride_, nb_classes_,
                                 faster_impl_, type);
}

size_t YoloxDecoderPlugin::getSerializationSize() const noexcept {
    // Note:serialize_value and deserialize_value save/load vector as: vector size
    // + vector data,
    //    remember to count the space of the size itself as well.
    return /*num_class*/ sizeof(int32_t) + /*stride*/ sizeof(int32_t) +
           /*faster_impl*/ sizeof(int32_t);
}

void YoloxDecoderPlugin::serialize(void *buffer) const noexcept {
    IXRT_PLUGIN_ASSERT(buffer != nullptr);
    serialize_value(&buffer, nb_classes_);
    serialize_value(&buffer, stride_);
    serialize_value(&buffer, faster_impl_);
}

YoloxDecoderPlugin::YoloxDecoderPlugin(void const *data, size_t length) {
    deserialize_value(&data, &length, &nb_classes_);
    deserialize_value(&data, &length, &stride_);
    deserialize_value(&data, &length, &faster_impl_);
}

DataType YoloxDecoderPlugin::getOutputDataType(int32_t index, DataType const *inputTypes,
                                               int32_t nbInputs) const noexcept {
    IXRT_PLUGIN_ASSERT(inputTypes != nullptr);
    IXRT_PLUGIN_ASSERT(nbInputs == 3);
    IXRT_PLUGIN_ASSERT(index == 0);
    return DataType::kHALF;
}

void YoloxDecoderPlugin::configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs,
                                         DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept {
    IXRT_PLUGIN_ASSERT(in != nullptr);
    IXRT_PLUGIN_ASSERT(out != nullptr);
}

YoloxDecodePluginCreator::YoloxDecodePluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("faster_impl", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const *YoloxDecodePluginCreator::getPluginName() const noexcept { return kYoloxDecoderPluginName; }

char const *YoloxDecodePluginCreator::getPluginVersion() const noexcept { return kYoloxDecoderPluginVersion; }

PluginFieldCollection const *YoloxDecodePluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2DynamicExt *YoloxDecodePluginCreator::createPlugin(char const *name,
                                                            PluginFieldCollection const *fc) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(fc != nullptr);
        IXRT_PLUGIN_ASSERT(fc->nbFields == 3);
        PluginField const *fields = fc->fields;

        int32_t num_class, stride, faster_impl;
        float *anchor = nullptr;
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            char const *attrName = fields[i].name;
            if (!strcmp(attrName, "num_class")) {
                IXRT_PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                num_class = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
            } else if (!strcmp(attrName, "stride")) {
                IXRT_PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                stride = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
            } else if (!strcmp(attrName, "faster_impl")) {
                IXRT_PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                faster_impl = *static_cast<int32_t *>(const_cast<void *>(fields[i].data));
            }
        }
        IPluginV2DynamicExt *plugin = new YoloxDecoderPlugin(num_class, stride, faster_impl);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt *YoloxDecodePluginCreator::deserializePlugin(char const *name, void const *data,
                                                                 size_t length) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(data != nullptr);
        return new YoloxDecoderPlugin(data, length);
    } catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

void YoloxDecodePluginCreator::setPluginNamespace(char const *libNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const &e) {
        caughtError(e);
    }
}

char const *YoloxDecodePluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

REGISTER_TENSORRT_PLUGIN(YoloxDecodePluginCreator);
