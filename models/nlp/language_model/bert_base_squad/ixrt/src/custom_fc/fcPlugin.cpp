#include "fcPlugin.h"

#include "NvInferRuntimeCommon.h"
#ifdef __ILUVATAR__
#include "backend/ixinfer/ixinfer_gemm_helper.h"
#endif
#include "bertCommon.h"
#include "checkMacrosPlugin.h"
#include "plugin.h"
#include "serialize.h"

using namespace nvinfer1;
using namespace nvinfer1::ixrt_plugin;
using namespace nvinfer1::ixrt_plugin::bert;
using namespace nvinfer1::ixrt_plugin::backend;

namespace {
char const* const kFC_VERSION{"1"};
char const* const kFC_NAME{"CustomFCPluginDynamic_IxRT"};
}  // namespace

// Static class fields initialization
PluginFieldCollection FCPluginDynamicCreator::mFC{};
std::vector<PluginField> FCPluginDynamicCreator::mPluginAttributes;

FCPluginDynamicCreator::FCPluginDynamicCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("out_dims", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("W", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* FCPluginDynamicCreator::getPluginName() const noexcept { return kFC_NAME; }

char const* FCPluginDynamicCreator::getPluginVersion() const noexcept { return kFC_VERSION; }

PluginFieldCollection const* FCPluginDynamicCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* FCPluginDynamicCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept {
    try {
        gLogInfo << "Creating FCPluginDynamicCreator..." << endl;
        IXRT_PLUGIN_ASSERT(name != nullptr);
        IXRT_PLUGIN_ASSERT(fc != nullptr);

        int32_t outDims = 0;
        int32_t typeId = -1;
        Weights W{DataType::kFLOAT, nullptr, 0LL};
        Weights B{DataType::kFLOAT, nullptr, 0LL};
        ixrt_plugin::validateRequiredAttributesExist({"out_dims", "type_id", "W"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++) {
            std::string fieldName(fc->fields[i].name);
            if (fieldName.compare("out_dims") == 0) {
                outDims = static_cast<int32_t const*>(fc->fields[i].data)[0];
                gLogInfo << "Building outDims: " << outDims << endl;
            }

            if (fieldName.compare("type_id") == 0) {
                typeId = static_cast<int32_t const*>(fc->fields[i].data)[0];
                gLogInfo << "Building typeId: " << typeId << endl;
            }

            if (fieldName.compare("W") == 0) {
                gLogInfo << "Building W..." << endl;
                W.values = fc->fields[i].data;
                W.count = fc->fields[i].length;
                W.type = fieldTypeToDataType(fc->fields[i].type);
                gLogInfo << "Is W float32: " << (W.type == DataType::kFLOAT) << endl;
            }

            if (fieldName.compare("B") == 0) {
                gLogInfo << "Building B..." << endl;
                B.values = fc->fields[i].data;
                B.count = fc->fields[i].length;
                B.type = fieldTypeToDataType(fc->fields[i].type);
                gLogInfo << "Is B float32: " << (B.type == DataType::kFLOAT) << endl;
            }
        }

        if (outDims <= 0) {
            gLogInfo << "Invalid output dimension" << endl;
        }
        if (typeId < 0 || typeId > 1) {
            gLogInfo << "Invalid type id" << typeId << endl;
        }
        if (W.count == 0 || W.values == nullptr || W.count < outDims) {
            gLogInfo << "Invalid weights" << endl;
        }

        DataType type = typeId == 0 ? DataType::kFLOAT : DataType::kHALF;
        return new FCPluginDynamic(name, type, outDims, W, B);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* FCPluginDynamicCreator::deserializePlugin(char const* name, void const* serialData,
                                                     size_t serialLength) noexcept {
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    try {
        return new FCPluginDynamic(name, serialData, serialLength);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void FCPluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* FCPluginDynamicCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// REGISTER_TENSORRT_PLUGIN(FCPluginDynamicCreator);
//#########################################################################//
FCPluginDynamic::FCPluginDynamic(std::string const name, DataType const type, int32_t const outDim, Weights const& W,
                                 Weights const& B)
    : mLayerName(name),
      mType(type),
      mOutDim(outDim),
      mNumParams(W.count),
      mNumBias(B.count),
      mWdev(nullptr),
      mBdev(nullptr) {
    mW.convertAndCopy(W, mType);
    copyToDevice(mW, getWeightsSize(mW, mType), mWdev);
    if (mNumBias) {
        mB.convertAndCopy(B, mType);
        copyToDevice(mB, getWeightsSize(mB, mType), mBdev);
    }
}

FCPluginDynamic::FCPluginDynamic(std::string const name, void const* data, size_t length)
    : mLayerName(name), mWdev(nullptr) {
    gLogInfo << "FCPluginDynamic deserialize" << endl;

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mOutDim);
    deserialize_value(&data, &length, &mNumParams);
    deserialize_value(&data, &length, &mNumBias);

    char const* d = static_cast<char const*>(data);

    mW.convertAndCopy(d, mNumParams, mType);
    copyToDevice(mW, getWeightsSize(mW, mType), mWdev);
    if (mNumBias) {
        mB.convertAndCopy(d, mNumBias, mType);
        copyToDevice(mB, getWeightsSize(mB, mType), mBdev);
    }
}

// IPluginV2 Methods
char const* FCPluginDynamic::getPluginType() const noexcept { return kFC_NAME; }

char const* FCPluginDynamic::getPluginVersion() const noexcept { return kFC_VERSION; }

int32_t FCPluginDynamic::getNbOutputs() const noexcept { return 1; }

int32_t FCPluginDynamic::initialize() noexcept {
    gLogInfo << "FCPluginDynamic initialize" << endl;
    return 0;
}

void FCPluginDynamic::terminate() noexcept { gLogInfo << "FCPluginDynamic terminate" << endl; }

size_t FCPluginDynamic::getSerializationSize() const noexcept {
    size_t wordSize = getElementSize(mType);
    return wordSize * (mNumParams + mNumBias) + sizeof(mType) + sizeof(mOutDim) + sizeof(mNumParams) + sizeof(mNumBias);
}

void FCPluginDynamic::serialize(void* buffer) const noexcept {
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mOutDim);
    serialize_value(&buffer, mNumParams);
    serialize_value(&buffer, mNumBias);

    size_t wordSize = getElementSize(mType);
    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mWdev.get()), mNumParams * wordSize);
    if (mNumBias) {
        serFromDev(d, static_cast<char*>(mBdev.get()), mNumBias * wordSize);
    }
}

void FCPluginDynamic::destroy() noexcept {
    gLogInfo << "FCPluginDynamic destroy" << endl;
    mWdev.reset(nullptr);
    if (mNumBias) {
        mBdev.reset(nullptr);
    }
    delete this;
}

void FCPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* FCPluginDynamic::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// IPluginV2Ext Methods
DataType FCPluginDynamic::getOutputDataType(int32_t index, DataType const* inputTypes,
                                            int32_t nbInputs) const noexcept {
    IXRT_PLUGIN_ASSERT(index == 0);
    IXRT_PLUGIN_ASSERT(nbInputs == 1);
    IXRT_PLUGIN_ASSERT(inputTypes != nullptr);
    IXRT_PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* FCPluginDynamic::clone() const noexcept {
    try {
        gLogInfo << "FCPluginDynamic clone" << endl;

        auto* p = new FCPluginDynamic(mLayerName, mType, mOutDim, mW, mB);
        p->setPluginNamespace(mNamespace.c_str());

        return p;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs FCPluginDynamic::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs,
                                               IExprBuilder& exprBuilder) noexcept {
    try {
        IXRT_PLUGIN_ASSERT(nbInputs == 1);
        IXRT_PLUGIN_ASSERT(outputIndex == 0);
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        DimsExprs ret;
        ret.nbDims = 5;
        ret.d[0] = inputs[0].d[0];
        ret.d[1] = inputs[0].d[1];
        ret.d[2] = exprBuilder.constant(mOutDim);
        ret.d[3] = exprBuilder.constant(1);
        ret.d[4] = exprBuilder.constant(1);
        return ret;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return DimsExprs{};
}

bool FCPluginDynamic::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                                int32_t nbOutputs) noexcept {
    IXRT_PLUGIN_ASSERT(nbInputs == 1);
    IXRT_PLUGIN_ASSERT(nbOutputs == 1);
    IXRT_PLUGIN_ASSERT(inOut != nullptr);

    PluginTensorDesc const& in = inOut[pos];
    if (pos == 0) {
        return (in.type == mType) && (in.format == TensorFormat::kLINEAR);
    }
    PluginTensorDesc const& prev = inOut[pos - 1];

    // output
    return in.type == prev.type && in.format == prev.format;
}

void FCPluginDynamic::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                      DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept {
    try {
        // Validate input arguments
        IXRT_PLUGIN_ASSERT(nbOutputs == 1);
        IXRT_PLUGIN_ASSERT(nbInputs == 1);
        IXRT_PLUGIN_ASSERT(inputs != nullptr);
        IXRT_PLUGIN_ASSERT(outputs != nullptr);
        IXRT_PLUGIN_ASSERT(mType == inputs[0].desc.type);
        auto const& inDims0 = inputs[0].desc.dims;

        IXRT_PLUGIN_ASSERT(inDims0.nbDims == 5);
        // IXRT_PLUGIN_ASSERT(hiddenSize * mOutDim == mNumParams);
        IXRT_PLUGIN_ASSERT(inDims0.d[3] == 1);
        IXRT_PLUGIN_ASSERT(inDims0.d[4] == 1);
#ifdef __ILUVATAR__
        CUINFER_CHECK(cuinferCreate(&cuinfer_handle));
#else
        CHECK_GPU_ERROR(cublasLtCreate(&blaslt_handle));
#endif
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

size_t FCPluginDynamic::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                                         PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t FCPluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                 void const* const* inputs, void* const* outputs, void* workSpace,
                                 cudaStream_t stream) noexcept {
    gLogInfo << "in FCPluginDynamic.." << endl;
    try {
#ifdef __ILUVATAR__
        CUINFER_CHECK(cuinferSetStream(cuinfer_handle, stream));
#endif
        int32_t const S = inputDesc->dims.d[SDIM];
        int32_t const B = inputDesc->dims.d[BDIM];
        int32_t const E = inputDesc->dims.d[HDIM];
        int32_t const n = S * B;
        IXRT_PLUGIN_ASSERT(n >= 0);

        if (mType == DataType::kHALF) {
            auto const* const input = static_cast<half const*>(inputs[0]);
            auto* output = static_cast<half*>(outputs[0]);
            auto weight = static_cast<half*>(mWdev.get());
            half* bias = nullptr;
            if (mNumBias) {
                bias = static_cast<half*>(mBdev.get());
            }

#ifdef __ILUVATAR__
            cuinfer_gemm(weight, input, bias, output, 1, mOutDim, n, E, 0, 0, 0, 1.0f, -1, stream, cuinfer_handle);
#else
            cublaslt_gemm(weight, input, output, 1, mOutDim, n, E, 0, 0, 0, 1.0f, blaslt_handle, stream);
#endif
        } else {
            gLogError << "Unsupported type error, expected [kHALF,kFLOAT], but received " << static_cast<int32_t>(mType)
                      << endl;
            return STATUS_FAILURE;
        }
        return STATUS_SUCCESS;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return STATUS_FAILURE;
}
