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
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "checkMacrosPlugin.h"
#include "common_def.cuh"
#include "cuda_fp16.h"
#include "yoloxDecoderKernel.h"
#include "yolox_decoder.cuh"

namespace nvinfer1::plugin {
int32_t YoloxDecoderInference(cudaStream_t stream, void const *box_data, void const *conf_data, void const *class_data,
                              void *output_data, const float box_quant_factor, const float conf_quant_factor,
                              const float class_quant_factor, const int batch_size, const int input_h,
                              const int input_w, const int input_channel_0, const int input_channel_1,
                              const int input_channel_2, const int stride, const int num_class, const int faster_impl,
                              nvinfer1::DataType type) {
    uint32_t kThreadPerBlock = 1024;
    int32_t total_boxes = batch_size * input_h * input_w;
    int32_t grid_ = (total_boxes + kThreadPerBlock - 1) / kThreadPerBlock;
    int32_t block_ = kThreadPerBlock;

    switch (type) {
        case DataType::kHALF: {
            YOLOX_Decode_NHWC_FP16<<<grid_, block_, 0, stream>>>(
                input_channel_0, input_channel_1, input_channel_2, batch_size, input_h, input_w, stride, num_class,
                reinterpret_cast<const __half *>(box_data), reinterpret_cast<const __half *>(conf_data),
                reinterpret_cast<const __half *>(class_data), reinterpret_cast<__half *>(output_data));
            break;
        }
        case DataType::kINT8: {
            YOLOX_Decode_NHWC_INT8<<<grid_, block_, 0, stream>>>(
                box_quant_factor, conf_quant_factor, class_quant_factor, input_channel_0, input_channel_1,
                input_channel_2, batch_size, input_h, input_w, stride, num_class,
                reinterpret_cast<const int8_t *>(box_data), reinterpret_cast<const int8_t *>(conf_data),
                reinterpret_cast<const int8_t *>(class_data), reinterpret_cast<__half *>(output_data));
            break;
        }
        default:
            IXRT_PLUGIN_FAIL("YoloxDecoderPlugin Unsupported datatype");
    }

    return 0;
}
}  // namespace nvinfer1::plugin
