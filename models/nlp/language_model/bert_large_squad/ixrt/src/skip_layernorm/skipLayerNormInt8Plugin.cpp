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
#include "skipLayerNormInt8Plugin.h"

#include "NvInferRuntime.h"
#include "checkMacrosPlugin.h"
#include "driver_types.h"
#include "plugin.h"
#include "serialize.h"

using namespace nvinfer1;
using namespace nvinfer1::ixrt_plugin;
using namespace nvinfer1::ixrt_plugin::bert;

// Clip plugin specific constants
namespace {
char const* kSKIP_LAYER_NORM_INT8_VERSION_HFACE{"3"};
char const* kSKIP_LAYER_NORM_INT8_VERSION_MTRON{"4"};
char const* kSKIP_LAYER_NORM_INT8_NAME{"CustomSkipLayerNormPluginDynamic_IxRT"};
}  // namespace

// Static class fields initialization
PluginFieldCollection SkipLayerNormInt8PluginBaseCreator::mFC{};
std::vector<PluginField> SkipLayerNormInt8PluginBaseCreator::mPluginAttributes;

constexpr auto param_type = DataType::kFLOAT;

SkipLayerNormInt8PluginBaseCreator::SkipLayerNormInt8PluginBaseCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("beta"));
    mPluginAttributes.emplace_back(PluginField("gamma"));
    mPluginAttributes.emplace_back(PluginField("bias"));
    mPluginAttributes.emplace_back(PluginField("output_fp32"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

SkipLayerNormInt8PluginHFaceCreator::SkipLayerNormInt8PluginHFaceCreator() : SkipLayerNormInt8PluginBaseCreator() {}

char const* SkipLayerNormInt8PluginBaseCreator::getPluginName() const noexcept { return kSKIP_LAYER_NORM_INT8_NAME; }

PluginFieldCollection const* SkipLayerNormInt8PluginBaseCreator::getFieldNames() noexcept { return &mFC; }

void SkipLayerNormInt8PluginBaseCreator::setPluginNamespace(char const* libNamespace) noexcept {
    mNamespace = libNamespace;
}

char const* SkipLayerNormInt8PluginBaseCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

char const* SkipLayerNormInt8PluginHFaceCreator::getPluginVersion() const noexcept {
    return kSKIP_LAYER_NORM_INT8_VERSION_HFACE;
}

bool buildBetaAndGamma(PluginFieldCollection const* fc, Weights& beta, Weights& gamma, Weights& bias) {
    ixrt_plugin::validateRequiredAttributesExist({"beta", "gamma"}, fc);

    bool output_fp32 = false;

    for (int32_t i = 0; i < fc->nbFields; i++) {
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("beta") == 0) {
            gLogInfo << "Building beta..." << endl;
            beta.values = fc->fields[i].data;
            beta.count = fc->fields[i].length;
            beta.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("gamma") == 0) {
            gLogInfo << "Building gamma..." << endl;
            gamma.values = fc->fields[i].data;
            gamma.count = fc->fields[i].length;
            gamma.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("bias") == 0) {
            gLogInfo << "Building bias..." << endl;
            bias.values = fc->fields[i].data;
            bias.count = fc->fields[i].length;
            bias.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("output_fp32") == 0) {
            IXRT_PLUGIN_ASSERT(fc->fields[i].type == PluginFieldType::kINT32);
            output_fp32 = (static_cast<int32_t const*>(fc->fields[i].data)[0] == 1);
            gLogInfo << "Building output_fp32" << output_fp32 << endl;
        }
    }

    IXRT_PLUGIN_CHECK_VALUE(beta.values != nullptr, "SkipLayerNorm: invalid beta");
    IXRT_PLUGIN_CHECK_VALUE(beta.count > 0, "SkipLayerNorm: invalid beta");

    IXRT_PLUGIN_CHECK_VALUE(gamma.values != nullptr, "SkipLayerNorm: invalid gamma");
    IXRT_PLUGIN_CHECK_VALUE(gamma.count > 0, "SkipLayerNorm: invalid gamma");
    return output_fp32;
}

IPluginV2* SkipLayerNormInt8PluginHFaceCreator::createPlugin(char const* name,
                                                             PluginFieldCollection const* fc) noexcept {
    try {
        gLogInfo << "SkipLayerNormInt8PluginHFaceCreator createPlugin" << endl;

        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        Weights bias{DataType::kFLOAT, nullptr, 0};
        bool output_fp32 = buildBetaAndGamma(fc, beta, gamma, bias);
        return new SkipLayerNormInt8PluginHFace(name, beta, gamma, bias, output_fp32);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormInt8PluginHFaceCreator::deserializePlugin(char const* name, void const* serialData,
                                                                  size_t serialLength) noexcept {
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormInterleavedPlugin::destroy()
    try {
        gLogInfo << "SkipLayerNormInterleavedPluginHFaceCreator deserializePlugin" << endl;
        return new SkipLayerNormInt8PluginHFace(name, serialData, serialLength);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

// REGISTER_TENSORRT_PLUGIN(SkipLayerNormInt8PluginHFaceCreator);
//#########################################################################//
SkipLayerNormInt8PluginBase::SkipLayerNormInt8PluginBase(std::string const& name, Weights const& beta,
                                                         Weights const& gamma, Weights const& bias, bool output_fp32)
    : mLayerName(name),
      mGammaDev(nullptr),
      mBetaDev(nullptr),
      mBiasDev(nullptr),
      mLd(beta.count),
      mParamsOnDevice(false),
      output_fp32(output_fp32) {
    IXRT_PLUGIN_ASSERT(mLd > 0);
    IXRT_PLUGIN_ASSERT(beta.count == gamma.count);
    // dataType for beta, gamma weights is always fp16
    mParamWordsize = getElementSize(param_type);

    mBeta.convertAndCopy(beta, param_type);
    mGamma.convertAndCopy(gamma, param_type);

    mHasBias = (bias.values != nullptr);
    if (mHasBias) {
        mBias.convertAndCopy(bias, param_type);
    }

    copyToDevice(mGamma, getWeightsSize(mGamma, param_type), mGammaDev);
    copyToDevice(mBeta, getWeightsSize(mBeta, param_type), mBetaDev);
    if (mHasBias) {
        copyToDevice(mBias, getWeightsSize(mBias, param_type), mBiasDev);
    }
}

SkipLayerNormInt8PluginBase::SkipLayerNormInt8PluginBase(std::string const& name, void const* data, size_t length)
    : mLayerName(name), mGammaDev(nullptr), mBetaDev(nullptr), mParamsOnDevice(false) {
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mHasBias);
    deserialize_value(&data, &length, &output_fp32);

    mParamWordsize = getElementSize(param_type);

    char const* d = static_cast<char const*>(data);
    mBeta.convertAndCopy(d, mLd, param_type);
    mGamma.convertAndCopy(d, mLd, param_type);

    if (mHasBias) {
        mBias.convertAndCopy(d, mLd, param_type);
    }

    copyToDevice(mGamma, getWeightsSize(mGamma, param_type), mGammaDev);
    copyToDevice(mBeta, getWeightsSize(mBeta, param_type), mBetaDev);
    if (mHasBias) {
        copyToDevice(mBias, getWeightsSize(mBias, param_type), mBiasDev);
    }
}

SkipLayerNormInt8PluginHFace::SkipLayerNormInt8PluginHFace(std::string const& name, Weights const& beta,
                                                           Weights const& gamma, Weights const& bias, bool output_fp32)
    : SkipLayerNormInt8PluginBase(name, beta, gamma, bias, output_fp32) {}

SkipLayerNormInt8PluginHFace::SkipLayerNormInt8PluginHFace(std::string const& name, void const* data, size_t length)
    : SkipLayerNormInt8PluginBase(name, data, length) {
    gLogInfo << "SkipLayerNormInt8PluginHFace deserialize" << endl;
}

// IPluginV2 Methods
char const* SkipLayerNormInt8PluginBase::getPluginType() const noexcept { return kSKIP_LAYER_NORM_INT8_NAME; }

size_t SkipLayerNormInt8PluginBase::getSerializationSize() const noexcept {
    const size_t biasSize = mHasBias ? (mLd * mParamWordsize) : 0;
    return 2 * mParamWordsize * mLd + sizeof(mLd) + sizeof(mHasBias) + sizeof(output_fp32) + biasSize;
}

void SkipLayerNormInt8PluginBase::serialize(void* buffer) const noexcept {
    try {
        serialize_value(&buffer, mLd);
        serialize_value(&buffer, mHasBias);
        serialize_value(&buffer, output_fp32);

        char* d = static_cast<char*>(buffer);
        serFromDev(d, static_cast<char*>(mBetaDev.get()), mLd * mParamWordsize);
        serFromDev(d, static_cast<char*>(mGammaDev.get()), mLd * mParamWordsize);
        if (mHasBias) {
            serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * mParamWordsize);
        }
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

void SkipLayerNormInt8PluginBase::destroy() noexcept {
    try {
        // This gets called when the network containing plugin is destroyed
        mGammaDev.reset(nullptr);
        mBetaDev.reset(nullptr);
        if (mHasBias) {
            mBiasDev.reset(nullptr);
        }
        delete this;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

void SkipLayerNormInt8PluginBase::setPluginNamespace(char const* libNamespace) noexcept { mNamespace = libNamespace; }

char const* SkipLayerNormInt8PluginBase::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// HFace
int32_t SkipLayerNormInt8PluginHFace::initialize() noexcept {
    gLogInfo << "SkipLayerNormInterleavedPluginHFace initialize" << endl;
    return 0;
}

void SkipLayerNormInt8PluginHFace::terminate() noexcept {
    gLogInfo << "SkipLayerNormInterleavedPluginHFace terminate" << endl;
}

void SkipLayerNormInt8PluginHFace::destroy() noexcept {
    gLogInfo << "SkipLayerNormInterleavedPluginHFace destroy" << endl;
    SkipLayerNormInt8PluginBase::destroy();
}

char const* SkipLayerNormInt8PluginHFace::getPluginVersion() const noexcept {
    return kSKIP_LAYER_NORM_INT8_VERSION_HFACE;
}

int32_t SkipLayerNormInt8PluginHFace::getNbOutputs() const noexcept { return 2; }

// IPluginV2Ext Methods
DataType SkipLayerNormInt8PluginBase::getOutputDataType(int32_t index, DataType const* inputTypes,
                                                        int32_t nbInputs) const noexcept {
    try {
        IXRT_PLUGIN_ASSERT(inputTypes != nullptr);
        IXRT_PLUGIN_ASSERT(index >= 0 && index < getNbOutputs());
        IXRT_PLUGIN_ASSERT(nbInputs == 3);
        if (index == 0) {
            return output_fp32 ? DataType::kFLOAT : DataType::kINT8;
        }
        return DataType::kFLOAT;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return DataType{};
}

// IPluginV2DynamicExt Methods
DimsExprs SkipLayerNormInt8PluginBase::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs,
                                                           int32_t nbInputs, IExprBuilder& exprBuilder) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        IXRT_PLUGIN_ASSERT(nbInputs == 3);
        IXRT_PLUGIN_ASSERT(outputIndex >= 0 && outputIndex < getNbOutputs());
        IXRT_PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);
        IXRT_PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);
        return inputs[0];
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return DimsExprs{};
}

bool SkipLayerNormInt8PluginBase::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut,
                                                            int32_t nbInputs, int32_t nbOutputs) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(inOut != nullptr);
        IXRT_PLUGIN_ASSERT(nbInputs == 3);
        IXRT_PLUGIN_ASSERT(nbOutputs == getNbOutputs());
        IXRT_PLUGIN_ASSERT(pos >= 0 && pos < (nbInputs + nbOutputs));

        PluginTensorDesc const& desc = inOut[pos];
        if (pos == 2 || pos == 4 || (output_fp32 && pos == 3)) {
            return desc.type == DataType::kFLOAT && desc.format == TensorFormat::kLINEAR;
        }
        return desc.type == DataType::kINT8 && desc.format == TensorFormat::kLINEAR;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return false;
}

void SkipLayerNormInt8PluginBase::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                                  DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept {
    try {
        // Validate input arguments
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        IXRT_PLUGIN_ASSERT(outputs != nullptr);
        IXRT_PLUGIN_ASSERT(nbOutputs == getNbOutputs());
        IXRT_PLUGIN_ASSERT(nbInputs == 3);

        auto const& inDims0 = inputs[0].desc.dims;
        auto const& inDims1 = inputs[1].desc.dims;
        auto const& inDims2 = inputs[2].desc.dims;
        TRT_UNUSED inDims1;
        TRT_UNUSED inDims2;

        IXRT_PLUGIN_ASSERT(inDims0.nbDims == inDims1.nbDims);
        IXRT_PLUGIN_ASSERT(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));
        IXRT_PLUGIN_ASSERT(inDims0.nbDims == inDims2.nbDims);
        IXRT_PLUGIN_ASSERT(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims2.d));

        mParamWordsize = getElementSize(param_type);
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

size_t SkipLayerNormInt8PluginBase::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                                                     PluginTensorDesc const* outputs,
                                                     int32_t nbOutputs) const noexcept {
    return 0;
}

// HFace IPluginV2DynamicExt Methods
IPluginV2DynamicExt* SkipLayerNormInt8PluginHFace::clone() const noexcept {
    try {
        gLogInfo << "SkipLayerNormInterleavedPluginHFace clone" << endl;
        auto* p = new SkipLayerNormInt8PluginHFace(mLayerName, mBeta, mGamma, mBias, output_fp32);
        p->initialize();
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

int32_t SkipLayerNormInt8PluginHFace::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                              void const* const* inputs, void* const* outputs, void* workspace,
                                              cudaStream_t stream) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        IXRT_PLUGIN_ASSERT(outputs != nullptr);
        auto const iDesc = inputDesc[0];
        auto const oDesc = outputDesc[0];

        const int32_t B = iDesc.dims.d[0];
        const int32_t S = iDesc.dims.d[1];
        const int32_t E = iDesc.dims.d[2];
        int batch_token_num = B * S;
        float const dqScaleIn = iDesc.scale;
        IXRT_PLUGIN_ASSERT(dqScaleIn > 1e-9);
        float const qScale = oDesc.scale;
        int8_t const* input = static_cast<int8_t const*>(inputs[0]);
        int8_t const* skip = static_cast<int8_t const*>(inputs[1]);
        float* residual = (float*)inputs[2];
        float const* gamma = static_cast<float const*>(mGammaDev.get());
        float const* beta = static_cast<float const*>(mBetaDev.get());
        float const* bias = static_cast<float const*>(mBiasDev.get());
        float* residual_out = static_cast<float*>(outputs[1]);

        if (!output_fp32) {
            int8_t* output = static_cast<int8_t*>(outputs[0]);
            skipLayerNormI8II8O(input, gamma, beta, bias, output, residual, residual_out, batch_token_num, E,
                                dqScaleIn, 1.0 / qScale, 1024, stream, true);
        } else {
            float* output = static_cast<float*>(outputs[0]);
            skipLayerNormI8IF32O(input, gamma, beta, bias, output, residual, residual_out, batch_token_num, E,
                                 1.0 / dqScaleIn, 1.0 / qScale, 1024, stream, true);
        }
        return STATUS_SUCCESS;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return STATUS_FAILURE;
}
