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
#include <cuda_runtime.h>
#include <stdint.h>

#include "NvInfer.h"

namespace nvinfer1::plugin {
int32_t YoloxDecoderInference(cudaStream_t stream, void const *box_data, void const *conf_data, void const *class_data,
                              void *output_data, const float box_quant_factor, const float conf_quant_factor,
                              const float class_quant_factor, const int batch_size, const int input_h,
                              const int input_w, const int input_channel_0, const int input_channel_1,
                              const int input_channel_2, const int stride, const int num_class, const int faster_impl,
                              nvinfer1::DataType type);
}
