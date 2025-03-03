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
#pragma once
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "common_def.cuh"
#include "kernels/cuda_helper.cuh"

namespace nvinfer1::plugin {
__global__ void YOLOX_Decode_NHWC_INT8(const float inp_scale1,  // 4
                                       const float inp_scale2,  // 1
                                       const float inp_scale3,  // 80
                                       const int in_channel1, const int in_channel2, const int in_channel3,
                                       const int N,  // batch size
                                       const int H, const int W, const int stride, const int nb_classes,
                                       const int8_t *inp_data1, const int8_t *inp_data2, const int8_t *inp_data3,
                                       __half *oup) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * H * W) return;

    const int h_idx = (tid % (H * W)) / W;  // y
    const int w_idx = (tid % (H * W)) % W;  // x

    // pointer to a featuremp
    float xywh0 = inp_data1[tid * in_channel1 + 0];
    float xywh1 = inp_data1[tid * in_channel1 + 1];
    float xywh2 = inp_data1[tid * in_channel1 + 2];
    float xywh3 = inp_data1[tid * in_channel1 + 3];
    float conf0 = inp_data2[tid * in_channel2 + 0];

    const float cx = (dequantize(xywh0, inp_scale1) + w_idx) * stride;
    const float cy = (dequantize(xywh1, inp_scale1) + h_idx) * stride;
    const float w = exp(dequantize(xywh2, inp_scale1)) * stride;
    const float h = exp(dequantize(xywh3, inp_scale1)) * stride;
    const float conf = sigmoid(dequantize(conf0, inp_scale2));

    float max_prob = sigmoid(dequantize(inp_data3[tid * in_channel3], inp_scale3));
    int class_id = 1;
    // #pragma unroll
    for (int i = 1; i < nb_classes; ++i) {
        float tmp_prob = sigmoid(dequantize(inp_data3[tid * in_channel3 + i], inp_scale3));
        if (tmp_prob > max_prob) {
            max_prob = tmp_prob;
            class_id = i + 1;
        }
    }
    float x1 = cx - 0.5f * w;
    float y1 = cy - 0.5f * h;

    oup[tid * 6 + 0] = __float2half(x1);
    oup[tid * 6 + 1] = __float2half(y1);
    oup[tid * 6 + 2] = __float2half(x1 + w);
    oup[tid * 6 + 3] = __float2half(y1 + h);
    oup[tid * 6 + 4] = __float2half(class_id);
    oup[tid * 6 + 5] = __float2half(max_prob * conf);
}

__global__ void YOLOX_Decode_NHWC_FP16(const int in_channel1, const int in_channel2, const int in_channel3, const int N,
                                       const int H, const int W, const int stride, const int nb_classes,
                                       const __half *inp_data1, const __half *inp_data2, const __half *inp_data3,
                                       __half *oup) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * H * W) return;

    const int h_idx = (tid % (H * W)) / W;  // y
    const int w_idx = (tid % (H * W)) % W;  // x

    // pointer to a featuremp
    float xywh0 = __half2float(inp_data1[tid * in_channel1 + 0]);
    float xywh1 = __half2float(inp_data1[tid * in_channel1 + 1]);
    float xywh2 = __half2float(inp_data1[tid * in_channel1 + 2]);
    float xywh3 = __half2float(inp_data1[tid * in_channel1 + 3]);
    float conf0 = __half2float(inp_data2[tid * in_channel2 + 0]);

    const float cx = (xywh0 + w_idx) * stride;
    const float cy = (xywh1 + h_idx) * stride;
    const float w = exp(xywh2) * stride;
    const float h = exp(xywh3) * stride;
    const float conf = sigmoid(conf0);
    float max_prob = sigmoid(__half2float(inp_data3[tid * nb_classes]));
    int class_id = 1;
    // #pragma unroll
    for (int i = 1; i < nb_classes; ++i) {
        float tmp_prob = sigmoid(__half2float(inp_data3[tid * nb_classes + i]));
        if (tmp_prob > max_prob) {
            max_prob = tmp_prob;
            class_id = i + 1;
        }
    }
    float x1 = cx - 0.5f * w;
    float y1 = cy - 0.5f * h;

    oup[tid * 6 + 0] = __float2half(x1);
    oup[tid * 6 + 1] = __float2half(y1);
    oup[tid * 6 + 2] = __float2half(x1 + w);
    oup[tid * 6 + 3] = __float2half(y1 + h);
    oup[tid * 6 + 4] = __float2half(class_id);
    oup[tid * 6 + 5] = __float2half(max_prob * conf);
}
}  // namespace nvinfer1::plugin
