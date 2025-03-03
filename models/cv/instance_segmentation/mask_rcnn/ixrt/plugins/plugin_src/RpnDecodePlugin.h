#pragma once

#include <NvInfer.h>

#include <cassert>
#include <vector>

#include "../macros.h"

using namespace nvinfer1;

#define PLUGIN_NAME "RpnDecode"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {

int rpnDecode(int32_t batchSize, const void *const *inputs, void *TRT_CONST_ENQUEUE *outputs, size_t height,
              size_t width, size_t image_height, size_t image_width, float stride, const std::vector<float> &anchors,
              int top_n, void *workspace, size_t workspace_size, cudaStream_t stream);

/*
    input1: scores{N, C, H, W} C->anchors
    input2: boxes{N, C, H, W} C->4*anchors
    output1: scores{N, C, 1} C->topk
    output2: boxes{N, C, 4} C->topk format:XYXY
    Description: implement anchor decode
*/
class RpnDecodePlugin : public IPluginV2DynamicExt {
    int32_t _top_n;
    std::vector<float> _anchors;
    float _stride;
    int32_t _batchsize;
    size_t _height;
    size_t _width;
    size_t _image_height;  // for cliping the boxes by limiting y coordinates to the range [0, height]
    size_t _image_width;   // for cliping the boxes by limiting x coordinates to the range [0, width]
    mutable int size = -1;

   protected:
    void deserialize(void const *data, size_t length) {
        const char *d = static_cast<const char *>(data);
        read(d, _top_n);
        size_t anchors_size;
        read(d, anchors_size);
        while (anchors_size--) {
            float val;
            read(d, val);
            _anchors.push_back(val);
        }
        read(d, _stride);
        read(d, _height);
        read(d, _width);
        read(d, _image_height);
        read(d, _image_width);
        read(d, _batchsize);
    }

    size_t getSerializationSize() const TRT_NOEXCEPT override {
        return sizeof(_top_n) + sizeof(_batchsize) + sizeof(size_t) + sizeof(float) * _anchors.size() +
               sizeof(_stride) + sizeof(_height) + sizeof(_width) + sizeof(_image_height) + sizeof(_image_width);
    }

    void serialize(void *buffer) const TRT_NOEXCEPT override {
        char *d = static_cast<char *>(buffer);
        write(d, _top_n);
        write(d, _anchors.size());
        for (auto &val : _anchors) {
            write(d, val);
        }
        write(d, _stride);
        write(d, _height);
        write(d, _width);
        write(d, _image_height);
        write(d, _image_width);
        write(d, _batchsize);
    }

   public:
    RpnDecodePlugin(int top_n, std::vector<float> const &anchors, float stride, size_t image_height, size_t image_width)
        : _top_n(top_n),
          _anchors(anchors),
          _stride(stride),
          _image_height(image_height),
          _image_width(image_width),
          _batchsize(0),
          _height(0),
          _width(0) {}

    RpnDecodePlugin(int top_n, std::vector<float> const &anchors, float stride, size_t height, size_t width,
                    size_t image_height, size_t image_width)
        : _top_n(top_n),
          _anchors(anchors),
          _stride(stride),
          _batchsize(0),
          _height(height),
          _width(width),
          _image_height(image_height),
          _image_width(image_width) {}

    RpnDecodePlugin(void const *data, size_t length) { this->deserialize(data, length); }

    const char *getPluginType() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    int getNbOutputs() const TRT_NOEXCEPT override { return 2; }

    DimsExprs getOutputDimensions(int32_t index, const DimsExprs *inputs, int nbInputDims,
                                  IExprBuilder &exprBuilder) TRT_NOEXCEPT override {
        assert(nbInputDims == 2);
        assert(index < this->getNbOutputs());
        DimsExprs result;
        result.nbDims = 3;

        result.d[0] = inputs[0].d[0];
        result.d[1] = exprBuilder.constant(_top_n);
        result.d[2] = exprBuilder.constant(index == 1 ? 4 : 1);
        return result;
    }

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs,
                                   int32_t nbOutputs) TRT_NOEXCEPT override {
        bool ret = (inOut[pos].type == DataType::kFLOAT && inOut[pos].format == PluginFormat::kLINEAR);
        return ret;
    }

    int initialize() TRT_NOEXCEPT override { return 0; }

    void terminate() TRT_NOEXCEPT override {}

    size_t getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs, PluginTensorDesc const *outputs,
                            int32_t nbOutputs) const TRT_NOEXCEPT override {
        if (size < 0) {
            size = rpnDecode(_batchsize, nullptr, nullptr, _height, _width, _image_height, _image_width, _stride,
                             _anchors, _top_n, nullptr, 0, nullptr);
        }
        return size;
    }

    int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs,
                    void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override {
        _batchsize = inputDesc->dims.d[0];
        _height = inputDesc->dims.d[2];
        _width = inputDesc->dims.d[3];
        size_t workspacesize = getWorkspaceSize(inputDesc, 2, outputDesc, 2);

        return rpnDecode(_batchsize, inputs, outputs, _height, _width, _image_height, _image_width, _stride, _anchors,
                         _top_n, workspace, workspacesize, stream);
    }

    void destroy() TRT_NOEXCEPT override { delete this; };

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}

    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const TRT_NOEXCEPT override {
        assert(index < 3);
        return DataType::kFLOAT;
    }

    void configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out,
                         int32_t nbOutputs) TRT_NOEXCEPT override {
        assert(nbInputs == 2);
        assert(nbOutputs == 2);
        // assert(in->desc.type == nvinfer1::DataType::kFLOAT);
        // assert(in->desc.format == nvinfer1::PluginFormat::kLINEAR);
        _batchsize = in->desc.dims.d[0];
        _height = in->desc.dims.d[2];
        _width = in->desc.dims.d[3];
    }

    IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override {
        return new RpnDecodePlugin(_top_n, _anchors, _stride, _height, _width, _image_height, _image_width);
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

class RpnDecodePluginCreator : public IPluginCreator {
   public:
    RpnDecodePluginCreator() {}

    const char *getPluginName() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT override {
        return new RpnDecodePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}
    const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return nullptr; }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override {
        try {
            PluginField const *fields = fc->fields;
            float stride;
            int32_t top_n;
            size_t image_height, image_width;
            float *anchors;

            for (int32_t i = 0; i < fc->nbFields; ++i) {
                char const *attrName = fields[i].name;
                if (!strcmp(attrName, "top_n")) {
                    top_n = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "stride")) {
                    stride = *static_cast<float *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "image_height")) {
                    image_height = (size_t) * static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "image_width")) {
                    image_width = (size_t) * static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "anchors")) {
                    anchors = static_cast<float *>(const_cast<void *>((fields[i].data)));
                }
            }
            std::vector<float> _anchors;
            for (auto i = 0; i < 60; i++) {
                _anchors.emplace_back(anchors[i]);
            }

            IPluginV2DynamicExt *plugin = new RpnDecodePlugin(top_n, _anchors, stride, image_height, image_width);
            return plugin;
        } catch (std::exception const &e) {
            std::cout << "build RpnNmsPlugin failed" << std::endl;
            return nullptr;
        }
        return nullptr;
    }
};

REGISTER_TENSORRT_PLUGIN(RpnDecodePluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
