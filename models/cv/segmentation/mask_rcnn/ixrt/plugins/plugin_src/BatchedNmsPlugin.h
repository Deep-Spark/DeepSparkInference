#pragma once

#include <NvInfer.h>

#include <cassert>
#include <vector>

#include "../macros.h"

using namespace nvinfer1;

#define PLUGIN_NAME "BatchedNms"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {
int32_t batchedNms(int32_t nms_method, int32_t batchSize, const void *const *inputs, void *TRT_CONST_ENQUEUE *outputs,
                   size_t count, int32_t detections_per_im, float nms_thresh, void *workspace, size_t workspace_size,
                   cudaStream_t stream);

/*
    input1: scores{N, C, 1} C->topk
    input2: boxes{N, C, 4} C->topk format:XYXY
    input3: classes{N, C, 1} C->topk
    output1: scores{N, C2, 1} C->detections_per_img
    output2: boxes{N, C2, 4} C->detections_per_img format:XYXY
    output3: classes{N, C2, 1} C->detections_per_img
    Description: implement batched nms
*/
class BatchedNmsPlugin : public IPluginV2DynamicExt {
    int32_t _nms_method;
    float _nms_thresh;
    int32_t _detections_per_im;
    int32_t _batchsize;

    size_t _count = 1;

   protected:
    void deserialize(void const *data, size_t length) {
        const char *d = static_cast<const char *>(data);
        read(d, _nms_method);
        read(d, _nms_thresh);
        read(d, _detections_per_im);
        read(d, _count);
        read(d, _batchsize);
    }

    size_t getSerializationSize() const TRT_NOEXCEPT override {
        size_t ret = sizeof(_nms_method) + sizeof(_nms_thresh) + sizeof(_detections_per_im) + sizeof(_count) +
                     sizeof(_batchsize);
        return ret;
    }

    void serialize(void *buffer) const TRT_NOEXCEPT override {
        char *d = static_cast<char *>(buffer);
        write(d, _nms_method);
        write(d, _nms_thresh);
        write(d, _detections_per_im);
        write(d, _count);
        write(d, _batchsize);
    }

   public:
    BatchedNmsPlugin(int nms_method, float nms_thresh, int detections_per_im)
        : _nms_method(nms_method), _nms_thresh(nms_thresh), _batchsize(0), _detections_per_im(detections_per_im) {
        assert(nms_method >= 0);
        assert(nms_thresh > 0);
        assert(detections_per_im > 0);
    }

    BatchedNmsPlugin(int nms_method, float nms_thresh, int detections_per_im, size_t count)
        : _nms_method(nms_method),
          _nms_thresh(nms_thresh),
          _batchsize(0),
          _detections_per_im(detections_per_im),
          _count(count) {
        assert(nms_method >= 0);
        assert(nms_thresh > 0);
        assert(detections_per_im > 0);
        assert(count > 0);
    }

    BatchedNmsPlugin(void const *data, size_t length) { this->deserialize(data, length); }

    const char *getPluginType() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    int getNbOutputs() const TRT_NOEXCEPT override { return 3; }

    DimsExprs getOutputDimensions(int32_t index, const DimsExprs *inputs, int nbInputDims,
                                  IExprBuilder &exprBuilder) TRT_NOEXCEPT override {
        assert(nbInputDims == 3);
        assert(index < this->getNbOutputs());

        DimsExprs ret = inputs[index];
        switch (index) {
            case 0:
                ret.d[1] = exprBuilder.constant(_detections_per_im);
                return ret;
            case 1:
                ret.d[1] = exprBuilder.constant(_detections_per_im);
                return ret;
            case 2:
                ret.d[1] = exprBuilder.constant(_detections_per_im);
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
        static int size = -1;
        if (size < 0) {
            size = batchedNms(_nms_method, _batchsize, nullptr, nullptr, _count, _detections_per_im, _nms_thresh,
                              nullptr, 0, nullptr);
        }
        return size;
    }

    int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs,
                    void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override {
        _batchsize = inputDesc[0].dims.d[0];
        size_t workspacesize = getWorkspaceSize(inputDesc, 3, outputDesc, 3);
        return batchedNms(_nms_method, _batchsize, inputs, outputs, _count, _detections_per_im, _nms_thresh, workspace,
                          workspacesize, stream);
    }

    void destroy() TRT_NOEXCEPT override { delete this; }

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
        _batchsize = in[0].desc.dims.d[0];
        _count = in[0].desc.dims.d[1];
    }

    IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override {
        return new BatchedNmsPlugin(_nms_method, _nms_thresh, _detections_per_im, _count);
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

class BatchedNmsPluginCreator : public IPluginCreator {
   public:
    BatchedNmsPluginCreator() {}

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }
    const char *getPluginName() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT override {
        return new BatchedNmsPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}
    const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return nullptr; }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override {
        try {
            PluginField const *fields = fc->fields;

            int32_t nms_method, detections_per_im;
            float nms_thresh;
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                char const *attrName = fields[i].name;
                if (!strcmp(attrName, "nms_method")) {
                    nms_method = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "nms_thresh")) {
                    nms_thresh = *static_cast<float *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "detections_per_im")) {
                    detections_per_im = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                }
            }
            IPluginV2DynamicExt *plugin = new BatchedNmsPlugin(nms_method, nms_thresh, detections_per_im);
            return plugin;
        } catch (std::exception const &e) {
            std::cout << "build BatchedNMSPlugin failed" << std::endl;
            return nullptr;
        }
        return nullptr;
    }
};

REGISTER_TENSORRT_PLUGIN(BatchedNmsPluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
