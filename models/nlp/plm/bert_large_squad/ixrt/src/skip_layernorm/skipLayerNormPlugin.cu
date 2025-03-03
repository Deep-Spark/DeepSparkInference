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
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cassert>

#include "backend/bert/bert_helper.h"
#include "skipLayerNormPlugin.h"
// #include "backend/transformer/transformer_add_norm.h"

using namespace nvinfer1::ixrt_plugin::backend;

namespace nvinfer1::ixrt_plugin {
namespace bert {

template <int THREAD_DATA_LEN>
__global__ void IxinferResidualBiasLnPad(const half *input, const half *scale, const half *bias,
                                         const half *residual_bias, half *output, half *residual, int hidden_size,
                                         bool is_post_ln) {
    float2 vals[THREAD_DATA_LEN];
    int block_start = blockIdx.x * hidden_size / 2;
    half2 *p_input = (half2 *)input;
    half2 *p_output = (half2 *)output;
    half2 *p_residual = (half2 *)residual;
    half2 *p_scale = (half2 *)scale;
    half2 *p_bias = (half2 *)bias;
    half2 *p_residual_bias = (half2 *)residual_bias;
    // one line start
    p_input += block_start;
    p_output += block_start;
    p_residual += block_start;

    float thread_m2 = 0;
    float thread_mean = 0;
    float thread_count = 0;

#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * C10_WARP_SIZE;
        if (element_index < hidden_size / 2) {
            half2 value1 = p_input[element_index];
            half2 value2 = p_residual[element_index];

            vals[it].x = __half2float(value1.x) + __half2float(value2.x);
            vals[it].y = __half2float(value1.y) + __half2float(value2.y);

            half2 res_bias_val_1;
            if (residual_bias == nullptr) {
                res_bias_val_1.x = __float2half(0.0f);
                res_bias_val_1.y = __float2half(0.0f);
            } else {
                res_bias_val_1 = p_residual_bias[element_index];
            }
            vals[it].x = vals[it].x + __half2float(res_bias_val_1.x);
            vals[it].y = vals[it].y + __half2float(res_bias_val_1.y);

            WelfordCombine(vals[it].x, &thread_mean, &thread_m2, &thread_count);
            WelfordCombine(vals[it].y, &thread_mean, &thread_m2, &thread_count);
        }
    }

    float mean = 0;
    float m2 = 0;
    float count = 0;
    WelfordWarpReduce(thread_mean, thread_m2, thread_count, &mean, &m2, &count);
    mean = __shfl_sync(0xffffffff, mean, 0, C10_WARP_SIZE);
    m2 = __shfl_sync(0xffffffff, m2, 0, C10_WARP_SIZE);
    count = __shfl_sync(0xffffffff, count, 0, C10_WARP_SIZE);

#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * C10_WARP_SIZE;
        if (element_index < hidden_size / 2) {
            float2 norm_value;
            half2 scale_1 = p_scale[element_index];
            half2 bias_1 = p_bias[element_index];
            norm_value.x = (vals[it].x - mean) * rsqrtf(m2 / hidden_size + epsilon) * __half2float(scale_1.x) +
                           __half2float(bias_1.x);
            norm_value.y = (vals[it].y - mean) * rsqrtf(m2 / hidden_size + epsilon) * __half2float(scale_1.y) +
                           __half2float(bias_1.y);

            half2 res;
            res.x = __float2half(norm_value.x);
            res.y = __float2half(norm_value.y);

            p_output[element_index] = res;

            half2 r1;
            if (is_post_ln) {
                r1 = res;
            } else {
                r1.x = __float2half(vals[it].x);
                r1.y = __float2half(vals[it].y);
            }
            p_residual[element_index] = r1;
        }
    }
}

void IxinferResidualBiasLnPad(const half *input, const half *scale, const half *bias, const half *residual_bias,
                              half *output, half *residual, int batch_tokens, int hidden_size, cudaStream_t stream,
                              bool is_post_ln) {
    if (hidden_size > 2048) {
        throw std::runtime_error("hidden_size should <= 1024");
    }
    if (hidden_size % 2 != 0) {
        throw std::runtime_error("hidden_size % 2 != 0");
    }

    dim3 gridSize(batch_tokens);
    dim3 blockSize(C10_WARP_SIZE);

    int neareast_hidden_size = hidden_size;
    if (neareast_hidden_size % (C10_WARP_SIZE * 2) != 0) {
        neareast_hidden_size = neareast_hidden_size + C10_WARP_SIZE * 2 - neareast_hidden_size % (C10_WARP_SIZE * 2);
    }

    int num_warp = neareast_hidden_size / C10_WARP_SIZE / 2;

    switch (num_warp) {
        case 1:
            IxinferResidualBiasLnPad<1><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                            residual, hidden_size, is_post_ln);
            break;
        case 2:
            IxinferResidualBiasLnPad<2><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                            residual, hidden_size, is_post_ln);
            break;
        case 3:
            IxinferResidualBiasLnPad<3><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                            residual, hidden_size, is_post_ln);
            break;
        case 4:
            IxinferResidualBiasLnPad<4><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                            residual, hidden_size, is_post_ln);
            break;
        case 5:
            IxinferResidualBiasLnPad<5><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                            residual, hidden_size, is_post_ln);
            break;
        case 6:
            IxinferResidualBiasLnPad<6><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                            residual, hidden_size, is_post_ln);
            break;
        case 7:
            IxinferResidualBiasLnPad<7><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                            residual, hidden_size, is_post_ln);
            break;
        case 8:
            IxinferResidualBiasLnPad<8><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                            residual, hidden_size, is_post_ln);
            break;
        case 9:
            IxinferResidualBiasLnPad<9><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                            residual, hidden_size, is_post_ln);
            break;
        case 10:
            IxinferResidualBiasLnPad<10><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
            break;
        case 11:
            IxinferResidualBiasLnPad<11><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
            break;
        case 12:
            IxinferResidualBiasLnPad<12><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
            break;
        case 13:
            IxinferResidualBiasLnPad<13><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
            break;
        case 14:
            IxinferResidualBiasLnPad<14><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
            break;
        case 15:
            IxinferResidualBiasLnPad<15><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
            break;
        case 16:
            IxinferResidualBiasLnPad<16><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
            break;
        default:
            std::cout << "hidden size: " << hidden_size << std::endl;
            throw std::runtime_error("IxinferResidualBiasLnPad not supported!");
            break;
    }
}

template <int THREAD_DATA_LEN>
__global__ void IxinferResidualBiasLn(const half *input, const half *scale, const half *bias, const half *residual_bias,
                                      half *output, half *residual, int hidden_size, bool is_post_ln) {
    float2 vals[THREAD_DATA_LEN];
    int block_start = blockIdx.x * hidden_size / 2;
    half2 *p_input = (half2 *)input;
    half2 *p_output = (half2 *)output;
    half2 *p_residual = (half2 *)residual;
    half2 *p_scale = (half2 *)scale;
    half2 *p_bias = (half2 *)bias;
    half2 *p_residual_bias = (half2 *)residual_bias;

    p_input += block_start;
    p_output += block_start;
    p_residual += block_start;

    float thread_m2 = 0;
    float thread_mean = 0;
    float thread_count = 0;

#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * C10_WARP_SIZE;
        half2 value1 = p_input[element_index];
        half2 value2 = p_residual[element_index];

        vals[it].x = __half2float(value1.x) + __half2float(value2.x);
        vals[it].y = __half2float(value1.y) + __half2float(value2.y);

        half2 res_bias_val_1;
        if (residual_bias == nullptr) {
            res_bias_val_1.x = __float2half(0.0f);
            res_bias_val_1.y = __float2half(0.0f);
        } else {
            res_bias_val_1 = p_residual_bias[element_index];
        }
        vals[it].x = vals[it].x + __half2float(res_bias_val_1.x);
        vals[it].y = vals[it].y + __half2float(res_bias_val_1.y);

        WelfordCombine(vals[it].x, &thread_mean, &thread_m2, &thread_count);
        WelfordCombine(vals[it].y, &thread_mean, &thread_m2, &thread_count);
    }

    float mean = 0;
    float m2 = 0;
    float count = 0;
    WelfordWarpReduce(thread_mean, thread_m2, thread_count, &mean, &m2, &count);
    mean = __shfl_sync(0xffffffff, mean, 0, C10_WARP_SIZE);
    m2 = __shfl_sync(0xffffffff, m2, 0, C10_WARP_SIZE);
    count = __shfl_sync(0xffffffff, count, 0, C10_WARP_SIZE);

#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * C10_WARP_SIZE;
        float2 norm_value;
        half2 scale_1 = p_scale[element_index];
        half2 bias_1 = p_bias[element_index];
        norm_value.x =
            (vals[it].x - mean) * rsqrtf(m2 / hidden_size + epsilon) * __half2float(scale_1.x) + __half2float(bias_1.x);
        norm_value.y =
            (vals[it].y - mean) * rsqrtf(m2 / hidden_size + epsilon) * __half2float(scale_1.y) + __half2float(bias_1.y);

        half2 res;
        res.x = __float2half(norm_value.x);
        res.y = __float2half(norm_value.y);

        p_output[element_index] = res;

        half2 r1;
        if (is_post_ln) {
            r1 = res;
        } else {
            r1.x = __float2half(vals[it].x);
            r1.y = __float2half(vals[it].y);
        }
        p_residual[element_index] = r1;
    }
}

void IxinferResidualBiasLn(const half *input, const half *scale, const half *bias, const half *residual_bias,
                           half *output, half *residual, int batch_tokens, int hidden_size, cudaStream_t stream,
                           bool is_post_ln) {
    if (hidden_size > 2048) {
        throw std::runtime_error("hidden_size should <= 1024");
    }
    if ((hidden_size % 2 == 0) && (hidden_size % (C10_WARP_SIZE * 2) != 0)) {
        IxinferResidualBiasLnPad(input, scale, bias, residual_bias, output, residual, batch_tokens, hidden_size, stream,
                                 is_post_ln);
    } else {
        if (hidden_size % (C10_WARP_SIZE * 2) != 0) {
            throw std::runtime_error("hidden_size // (C10_WARP_SIZE*2) != 0");
        }
        dim3 gridSize(batch_tokens);
        dim3 blockSize(C10_WARP_SIZE);

        int num_warp = hidden_size / C10_WARP_SIZE / 2;

        switch (num_warp) {
            case 1:
                IxinferResidualBiasLn<1><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
                break;
            case 2:
                IxinferResidualBiasLn<2><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
                break;
            case 3:
                IxinferResidualBiasLn<3><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
                break;
            case 4:
                IxinferResidualBiasLn<4><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
                break;
            case 5:
                IxinferResidualBiasLn<5><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
                break;
            case 6:
                IxinferResidualBiasLn<6><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
                break;
            case 7:
                IxinferResidualBiasLn<7><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
                break;
            case 8:
                IxinferResidualBiasLn<8><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
                break;
            case 9:
                IxinferResidualBiasLn<9><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                             residual, hidden_size, is_post_ln);
                break;
            case 10:
                IxinferResidualBiasLn<10><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                              residual, hidden_size, is_post_ln);
                break;
            case 11:
                IxinferResidualBiasLn<11><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                              residual, hidden_size, is_post_ln);
                break;
            case 12:
                IxinferResidualBiasLn<12><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                              residual, hidden_size, is_post_ln);
                break;
            case 13:
                IxinferResidualBiasLn<13><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                              residual, hidden_size, is_post_ln);
                break;
            case 14:
                IxinferResidualBiasLn<14><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                              residual, hidden_size, is_post_ln);
                break;
            case 15:
                IxinferResidualBiasLn<15><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                              residual, hidden_size, is_post_ln);
                break;
            case 16:
                IxinferResidualBiasLn<16><<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output,
                                                                              residual, hidden_size, is_post_ln);
                break;
            default:
                throw std::runtime_error("IxinferResidualBiasLn");
                break;
        }
    }
}

template <typename T, bool has_bias>
int32_t computeSkipLayerNorm(cudaStream_t stream, int32_t E, int32_t volume, const T* input, const T* gamma, const T* beta, const T* bias, T* skip, T* output)
{
    assert(volume % E == 0);
    int32_t batch_tokens = volume / E;
    IxinferResidualBiasLn(input, gamma, beta, bias, output, skip, batch_tokens, E, stream, true);
    return 0;
}

template int32_t computeSkipLayerNorm<half, true>(cudaStream_t, int32_t, int32_t, const half*, const half*, const half*, const half*, half*, half*);
template int32_t computeSkipLayerNorm<half, false>(cudaStream_t, int32_t, int32_t, const half*, const half*, const half*, const half*, half*, half*);
} // namespace bert
} // namespace nvinfer1::ixrt_plugin