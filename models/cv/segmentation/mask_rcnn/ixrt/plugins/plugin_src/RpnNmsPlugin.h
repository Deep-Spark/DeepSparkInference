#pragma once

#include <NvInfer.h>

#include <cassert>
#include <vector>

#include "../macros.h"

using namespace nvinfer1;

#define PLUGIN_NAME "RpnNms"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {

int rpnNms(int32_t batchSize, const void *const *inputs, void *TRT_CONST_ENQUEUE *outputs, size_t pre_nms_topk,
           int32_t post_nms_topk, float nms_thresh, void *workspace, size_t workspace_size, cudaStream_t stream);

/*
    input1: scores{N, C1, 1} C1->pre_nms_topk
    input2: boxes{N, C1, 4} C1->pre_nms_topk format:XYXY
    output1: boxes{N, C2, 4} C2->post_nms_topk format:XYXY
    Description: implement rpn nms
*/
class RpnNmsPlugin : public IPluginV2DynamicExt {
    float _nms_thresh;
    int32_t _post_nms_topk;
    int32_t _batchsize;
    size_t _pre_nms_topk = 1;
    mutable int size = -1;

   protected:
    void deserialize(void const *data, size_t length) {
        const char *d = static_cast<const char *>(data);
        read(d, _nms_thresh);
        read(d, _post_nms_topk);
        read(d, _pre_nms_topk);
        read(d, _batchsize);
    }

    size_t getSerializationSize() const TRT_NOEXCEPT override {
        return sizeof(_nms_thresh) + sizeof(_post_nms_topk) + sizeof(_pre_nms_topk) + sizeof(_batchsize);
    }

    void serialize(void *buffer) const TRT_NOEXCEPT override {
        char *d = static_cast<char *>(buffer);
        write(d, _nms_thresh);
        write(d, _post_nms_topk);
        write(d, _pre_nms_topk);
        write(d, _batchsize);
    }

   public:
    RpnNmsPlugin(float nms_thresh, int post_nms_topk)
        : _nms_thresh(nms_thresh), _post_nms_topk(post_nms_topk), _batchsize(0) {
        assert(nms_thresh > 0);
        assert(post_nms_topk > 0);
    }

    RpnNmsPlugin(float nms_thresh, int post_nms_topk, size_t pre_nms_topk)
        : _nms_thresh(nms_thresh), _post_nms_topk(post_nms_topk), _pre_nms_topk(pre_nms_topk), _batchsize(0) {
        assert(nms_thresh > 0);
        assert(post_nms_topk > 0);
        assert(pre_nms_topk > 0);
    }

    RpnNmsPlugin(void const *data, size_t length) { this->deserialize(data, length); }

    const char *getPluginType() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

    DimsExprs getOutputDimensions(int32_t index, const DimsExprs *inputs, int nbInputDims,
                                  IExprBuilder &exprBuilder) TRT_NOEXCEPT override {
        assert(nbInputDims == 2);
        assert(index < this->getNbOutputs());

        DimsExprs result;
        result.nbDims = 3;
        result.d[0] = inputs[0].d[0];
        result.d[1] = exprBuilder.constant(_post_nms_topk);
        result.d[2] = exprBuilder.constant(4);
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
            size =
                rpnNms(_batchsize, nullptr, nullptr, _pre_nms_topk, _post_nms_topk, _nms_thresh, nullptr, 0, nullptr);
        }
        return size;
    }

    int enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs,
                void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override {
        _batchsize = inputDesc->dims.d[0];
        size_t workspacesize = getWorkspaceSize(inputDesc, 2, outputDesc, 2);
        return rpnNms(_batchsize, inputs, outputs, _pre_nms_topk, _post_nms_topk, _nms_thresh, workspace, workspacesize,
                      stream);
    }

    void destroy() TRT_NOEXCEPT override { delete this; }

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}

    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const TRT_NOEXCEPT override {
        assert(index < 1);
        return DataType::kFLOAT;
    }

    void configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out,
                         int32_t nbOutputs) TRT_NOEXCEPT override {
        assert(nbInputs == 2);
        assert(nbOutputs == 1);
        // assert(in->desc.type == nvinfer1::DataType::kFLOAT);
        // assert(in->desc.format == nvinfer1::PluginFormat::kLINEAR);
        _batchsize = in->desc.dims.d[0];
        _pre_nms_topk = in->desc.dims.d[1];
    }

    IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override {
        return new RpnNmsPlugin(_nms_thresh, _post_nms_topk, _pre_nms_topk);
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

class RpnNmsPluginCreator : public IPluginCreator {
   public:
    RpnNmsPluginCreator() {}

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }
    const char *getPluginName() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT override {
        return new RpnNmsPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}
    const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return nullptr; }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override {
        try {
            PluginField const *fields = fc->fields;
            float nms_thresh;
            int32_t post_nms_topk;

            for (int32_t i = 0; i < fc->nbFields; ++i) {
                char const *attrName = fields[i].name;
                if (!strcmp(attrName, "nms_thresh")) {
                    nms_thresh = *static_cast<float *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "post_nms_topk")) {
                    post_nms_topk = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                }
            }
            IPluginV2DynamicExt *plugin = new RpnNmsPlugin(nms_thresh, post_nms_topk);
            return plugin;
        } catch (std::exception const &e) {
            std::cout << "build RpnNmsPlugin failed" << std::endl;
            return nullptr;
        }
        return nullptr;
    }
};

REGISTER_TENSORRT_PLUGIN(RpnNmsPluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
