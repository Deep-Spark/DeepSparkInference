#pragma once

#include <NvInfer.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "../macros.h"

using namespace nvinfer1;

#define PLUGIN_NAME "PredictorDecode"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {

int predictorDecode(int32_t batchSize, const void *const *inputs, void *TRT_CONST_ENQUEUE *outputs, uint32_t num_boxes,
                    uint32_t num_classes, uint32_t image_height, uint32_t image_width,
                    const std::vector<float> &bbox_reg_weights, void *workspace, size_t workspace_size,
                    cudaStream_t stream);

/*
    input1: scores{N,C,1,1} N->nums C->num of classes
    input2: boxes{N,C*4,1,1} N->nums C->num of classes
    input3: proposals{N,4} N->nums
    output1: scores{N, 1} N->nums
    output2: boxes{N, 4} N->nums format:XYXY
    output3: classes{N, 1} N->nums
    Description: implement fast rcnn decode
*/
class PredictorDecodePlugin : public IPluginV2DynamicExt {
    uint32_t _num_boxes;
    uint32_t _num_classes;
    uint32_t _image_height;
    uint32_t _image_width;
    std::vector<float> _bbox_reg_weights;
    mutable int size = -1;
    uint32_t _batchsize;

   protected:
    void deserialize(void const *data, size_t length) {
        const char *d = static_cast<const char *>(data);
        read(d, _batchsize);
        read(d, _num_boxes);
        read(d, _num_classes);
        read(d, _image_height);
        read(d, _image_width);
        size_t bbox_reg_weights_size;
        read(d, bbox_reg_weights_size);
        while (bbox_reg_weights_size--) {
            float val;
            read(d, val);
            _bbox_reg_weights.push_back(val);
        }
    }

    size_t getSerializationSize() const TRT_NOEXCEPT override {
        return sizeof(_batchsize) + sizeof(_num_boxes) + sizeof(_num_classes) + sizeof(_image_height) +
               sizeof(_image_width) + sizeof(size_t) + sizeof(float) * _bbox_reg_weights.size();
    }

    void serialize(void *buffer) const TRT_NOEXCEPT override {
        char *d = static_cast<char *>(buffer);
        write(d, _batchsize);
        write(d, _num_boxes);
        write(d, _num_classes);
        write(d, _image_height);
        write(d, _image_width);
        write(d, _bbox_reg_weights.size());
        for (auto &val : _bbox_reg_weights) {
            write(d, val);
        }
    }

   public:
    PredictorDecodePlugin(uint32_t num_boxes, uint32_t image_height, uint32_t image_width,
                          std::vector<float> const &bbox_reg_weights)
        : _num_boxes(num_boxes),
          _image_height(image_height),
          _batchsize(0),
          _image_width(image_width),
          _bbox_reg_weights(bbox_reg_weights) {}

    PredictorDecodePlugin(uint32_t num_boxes, uint32_t num_classes, uint32_t image_height, uint32_t image_width,
                          std::vector<float> const &bbox_reg_weights)
        : _num_boxes(num_boxes),
          _num_classes(num_classes),
          _batchsize(0),
          _image_height(image_height),
          _image_width(image_width),
          _bbox_reg_weights(bbox_reg_weights) {}

    PredictorDecodePlugin(void const *data, size_t length) { this->deserialize(data, length); }

    const char *getPluginType() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    int32_t getNbOutputs() const TRT_NOEXCEPT override { return 3; }

    DimsExprs getOutputDimensions(int32_t index, const DimsExprs *inputs, int nbInputDims,
                                  IExprBuilder &exprBuilder) TRT_NOEXCEPT override {
        assert(nbInputDims == 3);
        assert(index < this->getNbOutputs());

        DimsExprs ret;
        ret.nbDims = 3;
        switch (index) {
            case 0:
                ret.d[0] = inputs[0].d[0];
                ret.d[1] = inputs[0].d[1];
                ret.d[2] = exprBuilder.constant(1);
                return ret;
            case 1:
                ret.d[0] = inputs[0].d[0];
                ret.d[1] = inputs[0].d[1];
                ret.d[2] = exprBuilder.constant(4);
                return ret;
            case 2:
                ret.d[0] = inputs[0].d[0];
                ret.d[1] = inputs[0].d[1];
                ret.d[2] = exprBuilder.constant(1);
                return ret;
            default:
                return ret;
        }
    }

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs,
                                   int32_t nbOutputs) TRT_NOEXCEPT override {
        assert(0 <= pos && pos < 6);
        switch (pos) {
            case 0:
                return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
            case 1:
                return inOut[1].type == DataType::kFLOAT && inOut[1].format == TensorFormat::kLINEAR;
            case 2:
                return inOut[2].type == DataType::kFLOAT && inOut[2].format == TensorFormat::kLINEAR;
            case 3:
                return inOut[3].type == DataType::kFLOAT && inOut[3].format == TensorFormat::kLINEAR;
            case 4:
                return inOut[4].type == DataType::kFLOAT && inOut[4].format == TensorFormat::kLINEAR;
            case 5:
                return inOut[5].type == DataType::kFLOAT && inOut[5].format == TensorFormat::kLINEAR;
            default:
                return false;
        }
    }

    int initialize() TRT_NOEXCEPT override { return 0; }

    void terminate() TRT_NOEXCEPT override {}

    size_t getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs, PluginTensorDesc const *outputs,
                            int32_t nbOutputs) const TRT_NOEXCEPT override {
        if (size < 0) {
            size = predictorDecode(_batchsize, nullptr, nullptr, _num_boxes, _num_classes, _image_height, _image_width,
                                   _bbox_reg_weights, nullptr, 0, nullptr);
        }
        return size;
    }

    int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs,
                    void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override {
        _batchsize = inputDesc[0].dims.d[0];
        size_t workspacesize = getWorkspaceSize(inputDesc, 3, outputDesc, 3);
        return predictorDecode(_batchsize, inputs, outputs, _num_boxes, _num_classes, _image_height, _image_width,
                               _bbox_reg_weights, workspace, workspacesize, stream);
    }

    void destroy() TRT_NOEXCEPT override { delete this; };

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}

    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const TRT_NOEXCEPT override {
        assert(index < this->getNbOutputs());
        return DataType::kFLOAT;
    }

    void configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out,
                         int32_t nbOutputs) TRT_NOEXCEPT override {
        assert(nbInputs == 3);
        assert(nbOutputs == 3);
        // assert(in[0].desc.type == nvinfer1::DataType::kFLOAT);
        // assert(in[0].desc.format == nvinfer1::PluginFormat::kLINEAR);
        _num_classes = in[0].desc.dims.d[2];
        _batchsize = in[0].desc.dims.d[0];
    }

    IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override {
        return new PredictorDecodePlugin(_num_boxes, _num_classes, _image_height, _image_width, _bbox_reg_weights);
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

class PredictorDecodePluginCreator : public IPluginCreator {
   public:
    PredictorDecodePluginCreator() {}

    const char *getPluginName() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT override {
        return new PredictorDecodePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}
    const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return nullptr; }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override {
        try {
            PluginField const *fields = fc->fields;
            uint32_t num_boxes, image_height, image_width;
            std::vector<float> bbox_reg_weights_;
            float *bbox_reg_weights;

            for (int32_t i = 0; i < fc->nbFields; ++i) {
                char const *attrName = fields[i].name;
                if (!strcmp(attrName, "num_boxes")) {
                    num_boxes = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "image_height")) {
                    image_height = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "image_width")) {
                    image_width = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "bbox_reg_weights")) {
                    bbox_reg_weights = static_cast<float *>(const_cast<void *>((fields[i].data)));
                }
            }
            for (auto i = 0; i < 4; i++) {
                bbox_reg_weights_.emplace_back(bbox_reg_weights[i]);
            }

            IPluginV2DynamicExt *plugin =
                new PredictorDecodePlugin(num_boxes, image_height, image_width, bbox_reg_weights_);
            return plugin;
        } catch (std::exception const &e) {
            std::cout << "build PredictorDecodePlugin failed" << std::endl;
            return nullptr;
        }
        return nullptr;
    }
};

REGISTER_TENSORRT_PLUGIN(PredictorDecodePluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
