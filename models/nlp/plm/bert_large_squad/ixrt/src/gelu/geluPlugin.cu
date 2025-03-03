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
*
* SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "backend/bert/bert_helper.h"
#include "geluPlugin.h"

namespace nvinfer1::ixrt_plugin {
using namespace backend;
namespace bert {
// constants for approximating the normal cdf
constexpr float A = 0.5f;
constexpr float B = 0.7978845608028654f;    // sqrt(2.0/M_PI)
constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0/M_PI)


template <typename T>
__global__ void IxinferBiasGeluI8II8OKernel(int8_t *input, int8_t *output, const T *bias, int feature_dim,
                                            float dequant_scale, float quant_scale) {
    int block_start = blockIdx.x * feature_dim;
    int start = block_start + threadIdx.x;
    int end = block_start + feature_dim;
    for (int i = start; i < end; i += blockDim.x) {
        int input_index = i;

        float fout = gelu<float>(float(input[input_index]) * dequant_scale + __ldg(&bias[i - block_start]));

        int output_index = i;
        output[output_index] = float2int8(fout, quant_scale);
    }
}

template <>
__global__ void IxinferBiasGeluI8II8OKernel<__half>(int8_t *input, int8_t *output, const __half *bias, int feature_dim,
                                                    float dequant_scale, float quant_scale) {
    //  #pragma unroll
    for (int block_index = 0; block_index < 2; block_index++) {
        int block_start = (blockIdx.x * 2 + block_index) * feature_dim;
        int start = block_start + threadIdx.x * 4;
        int input_index = start;
        char4 *p_input = (char4 *)(input + input_index);
        half2 *p_bias = (half2 *)(bias + input_index - block_start);
        float fout1 = gelu<float>(float(p_input[0].x) * dequant_scale + __half2float(p_bias[0].x));
        float fout2 = gelu<float>(float(p_input[0].y) * dequant_scale + __half2float(p_bias[0].y));
        float fout3 = gelu<float>(float(p_input[0].z) * dequant_scale + __half2float(p_bias[1].x));
        float fout4 = gelu<float>(float(p_input[0].w) * dequant_scale + __half2float(p_bias[1].y));

        int output_index = start;
        char4 out;
        out.x = float2int8(fout1, quant_scale);
        out.y = float2int8(fout2, quant_scale);
        out.z = float2int8(fout3, quant_scale);
        out.w = float2int8(fout4, quant_scale);
        char4 *p_output = (char4 *)(output + output_index);

        p_output[0] = out;
    }
}

template <typename T>
void IxinferBiasGeluI8II8O(int batch_token_num, cudaStream_t stream, int8_t *input, int8_t *output, const T *bias,
                           int feature_dim, float dequant_scale, float quant_scale) {
    IxinferBiasGeluI8II8OKernel<T>
        <<<batch_token_num, 1024, 0, stream>>>(input, output, bias, feature_dim, dequant_scale, quant_scale);
}

template void IxinferBiasGeluI8II8O<half>(int, cudaStream_t, int8_t*, int8_t *, const half *, int, float, float);

template <unsigned TPB>
__global__ void geluKernel(const half a, const half b, const half c, int n, const half* input, half* output) {
    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n) {
        const half in = input[idx];
        const half cdf = a + a * __float2half(tanh(__half2float(in * (c * in * in + b))));
        output[idx] = in * cdf;
    }
}

template <unsigned TPB>
__global__ void geluKernel(const float a, const float b, const float c, int n, const float* input, float* output) {
    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n) {
        const float in = input[idx];
        const float cdf = a + a * tanh(in * (c * in * in + b));
        output[idx] = in * cdf;
    }
}

template <unsigned TPB>
__global__ void geluKernel(const float a, const float b, const float c, int n, const int8_t* input, int8_t* output,
                           float dequant_scale, float quant_scale) {
    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n) {
        const float in = float(input[idx]) * dequant_scale;
        const float cdf = a + a * tanh(in * (c * in * in + b));
        float i8_f = in * cdf * quant_scale;
        int32_t i8 = floorf(i8_f + 0.5);
        i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
        output[idx] = int8_t(i8);
    }
}

int computeGelu(cudaStream_t stream, int n, const float* input, float* output) {
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    geluKernel<blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, n, input, output);

    return 0;
}

int computeGelu(cudaStream_t stream, int n, const half* input, half* output) {
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    geluKernel<blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, n, input, output);

    return 0;
}

int32_t computeGeluI8O8(cudaStream_t stream, int n, const int8_t* input, int8_t* output, float dequant_scale,
                        float quant_scale) {
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    geluKernel<blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, n, input, output, dequant_scale, quant_scale);

    return 0;
}

template <int TPB>
__global__ void geluBiasKernel(const half a, const half b, const half c, half* output, const half* input,
                               const half* bias, const int ld) {
    const int offset = blockIdx.x * ld;

    for (int it = threadIdx.x; it < ld; it += TPB) {
        const int idx = it + offset;
        const half in = input[idx] + bias[it];
        const half cdf = a + a * __float2half(tanh(__half2float(in * (c * in * in + b))));
        output[idx] = in * cdf;
    }
}

template <int TPB>
__global__ void geluBiasKernel(const float a, const float b, const float c, float* output, const float* input,
                               const float* bias, const int ld) {
    const int offset = blockIdx.x * ld;

    for (int it = threadIdx.x; it < ld; it += TPB) {
        const int idx = it + offset;
        const float in = input[idx] + bias[it];
        const float cdf = a + a * tanh(in * (c * in * in + b));
        output[idx] = in * cdf;
    }
}

template <int TPB>
__global__ void geluBiasKernel(const float a, const float b, const float c, int8_t* output, const int8_t* input,
                               const half* bias, float dequant_scale, float quant_scale, const int ld) {
    const int offset = blockIdx.x * ld;

    for (int it = threadIdx.x; it < ld; it += TPB) {
        const int idx = it + offset;
        const float in = float(input[idx]) * dequant_scale + __half2float(bias[it]);
        const float cdf = a + a * tanh(in * (c * in * in + b));
        float i8_f = in * cdf * quant_scale;
        int32_t i8 = floorf(i8_f + 0.5);
        i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
        output[idx] = int8_t(i8);
    }
}

int computeGeluBias(float* output, const float* input, const float* bias, const int ld, const int cols,
                    cudaStream_t stream) {
    geluBiasKernel<256><<<cols, 256, 0, stream>>>(A, B, C, output, input, bias, ld);
    return cudaPeekAtLastError();
}

int computeGeluBias(half* output, const half* input, const half* bias, const int ld, const int cols,
                    cudaStream_t stream) {
    geluBiasKernel<256><<<cols, 256, 0, stream>>>(A, B, C, output, input, bias, ld);
    return cudaPeekAtLastError();
}

int32_t computeGeluI8O8Bias(int8_t* output, const int8_t* input, const half* bias, const int ld, const int cols,
                            float dequant_scale, float quant_scale, cudaStream_t stream) {
    geluBiasKernel<256><<<cols, 256, 0, stream>>>(A, B, C, output, input, bias, dequant_scale, quant_scale, ld);
    return cudaPeekAtLastError();
}

}  // namespace bert
}  // namespace nvinfer1::plugin
