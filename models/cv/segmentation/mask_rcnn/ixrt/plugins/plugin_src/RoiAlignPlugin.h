#pragma once

#include <NvInfer.h>

#include <cassert>
#include <vector>

#include "../macros.h"

using namespace nvinfer1;

#define PLUGIN_NAME "RoiAlign"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {
int32_t roiAlign(int32_t batchSize, const void *const *inputs, void *TRT_CONST_ENQUEUE *outputs,
                 int32_t pooler_resolution, float spatial_scale, int32_t sampling_ratio, int32_t num_proposals,
                 int32_t out_channels, int32_t feature_h, int32_t feature_w, cudaStream_t stream);

/*
    input1: boxes{B, N,4} N->post_nms_topk
    input2: features{B, C,H,W} C->num of feature map channels
    output1: features{B, N, C, H, W} N:nums of proposals C:output out_channels H,W:roialign size
    Description: roialign
*/
class RoiAlignPlugin : public IPluginV2DynamicExt {
    int32_t _pooler_resolution;
    float _spatial_scale;
    int32_t _sampling_ratio;
    int32_t _num_proposals;
    int32_t _out_channels;

   protected:
    void deserialize(void const *data, size_t length) {
        const char *d = static_cast<const char *>(data);
        read(d, _pooler_resolution);
        read(d, _spatial_scale);
        read(d, _sampling_ratio);
        read(d, _num_proposals);
        read(d, _out_channels);
    }

    size_t getSerializationSize() const TRT_NOEXCEPT override {
        return sizeof(_pooler_resolution) + sizeof(_spatial_scale) + sizeof(_sampling_ratio) + sizeof(_num_proposals) +
               sizeof(_out_channels);
    }

    void serialize(void *buffer) const TRT_NOEXCEPT override {
        char *d = static_cast<char *>(buffer);
        write(d, _pooler_resolution);
        write(d, _spatial_scale);
        write(d, _sampling_ratio);
        write(d, _num_proposals);
        write(d, _out_channels);
    }

   public:
    RoiAlignPlugin(int pooler_resolution, float spatial_scale, int sampling_ratio, int num_proposals, int out_channels)
        : _pooler_resolution(pooler_resolution),
          _spatial_scale(spatial_scale),
          _sampling_ratio(sampling_ratio),
          _num_proposals(num_proposals),
          _out_channels(out_channels) {}

    RoiAlignPlugin(void const *data, size_t length) { this->deserialize(data, length); }

    const char *getPluginType() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

    DimsExprs getOutputDimensions(int32_t index, const DimsExprs *inputs, int32_t nbInputs,
                                  IExprBuilder &exprBuilder) TRT_NOEXCEPT override {
        assert(index < this->getNbOutputs());
        assert(nbInputs == 2);
        DimsExprs output_dims;
        output_dims.nbDims = 5;
        output_dims.d[0] = inputs[0].d[0];
        output_dims.d[1] = exprBuilder.constant(_num_proposals);
        output_dims.d[2] = exprBuilder.constant(_out_channels);
        output_dims.d[3] = exprBuilder.constant(_pooler_resolution);
        output_dims.d[4] = exprBuilder.constant(_pooler_resolution);
        return output_dims;
    }

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs,
                                   int32_t nbOutputs) TRT_NOEXCEPT override {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == PluginFormat::kLINEAR;
    }

    int initialize() TRT_NOEXCEPT override { return 0; }

    void terminate() TRT_NOEXCEPT override {}

    size_t getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs, PluginTensorDesc const *outputs,
                            int32_t nbOutputs) const TRT_NOEXCEPT override {
        return 0;
    }

    int enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, const void *const *inputs,
                void *TRT_CONST_ENQUEUE *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override {
        int32_t batch_size = inputDesc[0].dims.d[0];
        int32_t feature_h = inputDesc[1].dims.d[2];
        int32_t feature_w = inputDesc[1].dims.d[3];
        return roiAlign(batch_size, inputs, outputs, _pooler_resolution, _spatial_scale, _sampling_ratio,
                        _num_proposals, _out_channels, feature_h, feature_w, stream);
    }

    void destroy() TRT_NOEXCEPT override { delete this; };

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}

    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const TRT_NOEXCEPT override {
        assert(index < this->getNbOutputs());
        return DataType::kFLOAT;
    }

    void configurePlugin(const DynamicPluginTensorDesc *inputDims, int nbInputs,
                         const DynamicPluginTensorDesc *outputDims, int nbOutputs) TRT_NOEXCEPT override {
        assert(nbInputs == 2);
        assert(nbOutputs == 1);
        auto const &boxes_dims = inputDims[0].desc.dims;
        auto const &feature_dims = inputDims[1].desc.dims;
        assert(_num_proposals == boxes_dims.d[1]);
        assert(_out_channels == feature_dims.d[1]);
        // _feature_h = feature_dims.d[2];
        // _feature_w = feature_dims.d[3];
    }

    IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override {
        return new RoiAlignPlugin(_pooler_resolution, _spatial_scale, _sampling_ratio, _num_proposals, _out_channels);
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

class RoiAlignPluginCreator : public IPluginCreator {
   public:
    RoiAlignPluginCreator() {}

    const char *getPluginName() const TRT_NOEXCEPT override { return PLUGIN_NAME; }

    const char *getPluginVersion() const TRT_NOEXCEPT override { return PLUGIN_VERSION; }

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return PLUGIN_NAMESPACE; }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT override {
        return new RoiAlignPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}
    const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return nullptr; }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override {
        try {
            // PLUGIN_ASSERT(fc != nullptr);
            // PLUGIN_ASSERT(fc->nbFields == 5);
            PluginField const *fields = fc->fields;

            int32_t pooler_resolution, sampling_ratio, num_proposals, out_channels;
            float spatial_scale;
            float *anchor = nullptr;
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                char const *attrName = fields[i].name;
                if (!strcmp(attrName, "pooler_resolution")) {
                    pooler_resolution = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "sampling_ratio")) {
                    sampling_ratio = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "num_proposals")) {
                    num_proposals = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "out_channels")) {
                    out_channels = *static_cast<int32_t *>(const_cast<void *>((fields[i].data)));
                } else if (!strcmp(attrName, "spatial_scale")) {
                    spatial_scale = *static_cast<float *>(const_cast<void *>(fields[i].data));
                }
            }
            IPluginV2DynamicExt *plugin =
                new RoiAlignPlugin(pooler_resolution, spatial_scale, sampling_ratio, num_proposals, out_channels);
            return plugin;
        } catch (std::exception const &e) {
            std::cout << "build RoiAlignPlugin failed" << std::endl;
            return nullptr;
        }
        return nullptr;
    }
};

REGISTER_TENSORRT_PLUGIN(RoiAlignPluginCreator);
}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
