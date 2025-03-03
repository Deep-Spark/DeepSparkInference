#pragma once

#include <NvInfer.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "../macros.h"

using namespace nvinfer1;

#define PLUGIN_NAME "MaskRcnnInference"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {
int32_t maskRcnnInference(int32_t batchSize, const void *const *inputs, void *TRT_CONST_ENQUEUE *outputs,
                          int32_t detections_per_im, int32_t output_size, int32_t num_classes, cudaStream_t stream);
/*
    input1: indices{N, C, 1}
    input2: masks{N, C, NUM_CLASS, size, size}  format:XYXY
    output1: masks{N, C, 1, size, size} C->detections_per_img
    Description: implement index select
*/

class MaskRcnnInferencePlugin : public IPluginV2DynamicExt {
    int32_t _detections_per_im;
    int32_t _output_size;
    int32_t _num_classes = 1;
    int32_t _batchsize;

   protected:
    void deserialize(void const *data, size_t length) {
        const char *d = static_cast<const char *>(data);
        read(d, _detections_per_im);
        read(d, _output_size);
        read(d, _num_classes);
        read(d, _batchsize);
    }

    size_t getSerializationSize() const TRT_NOEXCEPT override {
        return sizeof(_detections_per_im) + sizeof(_output_size) + sizeof(_num_classes) + sizeof(_batchsize);
    }

    void serialize(void *buffer) const TRT_NOEXCEPT override {
        char *d = static_cast<char *>(buffer);
        write(d, _detections_per_im);
        write(d, _output_size);
        write(d, _num_classes);
        write(d, _batchsize);
    }

   public:
    MaskRcnnInferencePlugin(int32_t detections_per_im, int32_t output_size)
        : _detections_per_im(detections_per_im), _output_size(output_size), _batchsize(0) {
        assert(detections_per_im > 0);
        assert(output_size > 0);
    }

    MaskRcnnInferencePlugin(int32_t detections_per_im, int32_t output_size, int32_t num_classes)
        : _detections_per_im(detections_per_im), _output_size(output_size), _num_classes(num_classes), _batchsize(0) {
        assert(detections_per_im > 0);
        assert(output_size > 0);
        assert(num_classes > 0);
    }

    MaskRcnnInferencePlugin(void const *data, size_t length) { this->deserialize(data, length); }

    const char *getPluginType() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    int32_t getNbOutputs() const TRT_NOEXCEPT override { return 1; }

    DimsExprs getOutputDimensions(int32_t index, const DimsExprs *inputs, int nbInputDims,
                                  IExprBuilder &exprBuilder) TRT_NOEXCEPT override {
        assert(nbInputDims == 2);
        assert(index < this->getNbOutputs());

        DimsExprs ret = inputs[1];
        ret.d[2] = exprBuilder.constant(1);
        return ret;
    }

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs,
                                   int32_t nbOutputs) TRT_NOEXCEPT override {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    int initialize() TRT_NOEXCEPT override { return 0; }

    void terminate() TRT_NOEXCEPT override {}

    size_t getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs, PluginTensorDesc const *outputs,
                            int32_t nbOutputs) const TRT_NOEXCEPT override {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs,
                    void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override {
        _batchsize = inputDesc[0].dims.d[0];
        return maskRcnnInference(_batchsize, inputs, outputs, _detections_per_im, _output_size, _num_classes, stream);
    }

    void destroy() TRT_NOEXCEPT override { delete this; }

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}

    // IPluginV2Ext Methods
    DataType getOutputDataType(int32_t index, const DataType *inputTypes, int nbInputs) const TRT_NOEXCEPT override {
        assert(index < 1);
        return DataType::kFLOAT;
    }

    void configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out,
                         int32_t nbOutputs) TRT_NOEXCEPT override {
        assert(nbInputs == 2);
        assert(nbOutputs == 1);
        // assert(in[0].desc.type == nvinfer1::DataType::kFLOAT);
        // assert(in[0].desc.format == nvinfer1::PluginFormat::kLINEAR);
        _num_classes = in[1].desc.dims.d[2];
        _batchsize = in[0].desc.dims.d[0];
    }

    IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override {
        return new MaskRcnnInferencePlugin(_detections_per_im, _output_size, _num_classes);
    }

    using IPluginV2::enqueue;
    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2Ext::configurePlugin;

   private:
    template <typename T>
    void write(char *&buffer, const T &val) const {
        *reinterpret_cast<T *>(buffer) = val;
        buffer += sizeof(T);
    }
    template <typename T>
    void read(const char *&buffer, T &val) {
        val = *reinterpret_cast<const T *>(buffer);
        buffer += sizeof(T);
    }
};

class MaskRcnnInferencePluginCreator : public IPluginCreator {
   public:
    MaskRcnnInferencePluginCreator() {}

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }

    const char *getPluginName() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT override {
        return new MaskRcnnInferencePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}
    const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return nullptr; }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override {
        try {
            PluginField const *fields = fc->fields;

            int32_t detections_per_im, output_size;

            for (int32_t i = 0; i < fc->nbFields; ++i) {
                char const *attrName = fields[i].name;
                if (!strcmp(attrName, "detections_per_im")) {
                    detections_per_im = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "output_size")) {
                    output_size = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                }
            }
            IPluginV2DynamicExt *plugin = new MaskRcnnInferencePlugin(detections_per_im, output_size);
            return plugin;
        } catch (std::exception const &e) {
            std::cout << "build MaskRcnnInferencePlugin failed" << std::endl;
            return nullptr;
        }
        return nullptr;
    }
};

REGISTER_TENSORRT_PLUGIN(MaskRcnnInferencePluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
