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
#include "backend/bert/bert_helper.h"
#include "skipLayerNormInt8Plugin.h"
using namespace nvinfer1::ixrt_plugin::backend;

namespace nvinfer1::ixrt_plugin {
namespace bert {

template <int THREAD_DATA_LEN>
__global__ void skipLayernormI8II8OKernel(const int8_t *input, const float *scale, const float *bias,
                                        const float *residual_bias, int8_t *output, float *residual, float* residual_out, 
                                        int hidden_size, float dequant_scale, float quant_scale,
                                        bool is_post_ln) {
    // register
    // process 2 data
    float4 vals[THREAD_DATA_LEN];
    int block_start = blockIdx.x *  hidden_size / 4;
    char4 *p_input = (char4 *)input;
    char4 *p_output = (char4 *)output;
    float4 *p_residual = (float4 *)residual;
    float4 *p_residual_out = (float4 *)residual_out;
    float4 *p_scale = (float4 *)scale;
    float4 *p_bias = (float4 *)bias;
    float4 *p_residual_bias = (float4 *)residual_bias;
    // one line start
    p_input += block_start;
    p_output += block_start;
    p_residual += block_start; 
    p_residual_out += block_start;

    float thread_m2 = 0;
    float thread_mean = 0;
    float thread_count = 0;

    // load data from global memory
#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * C10_WARP_SIZE;
         // vals = dequant(input) + residual + bias
        p_residual_out[element_index].x = p_residual[element_index].x + p_residual_bias[element_index].x;
        p_residual_out[element_index].y = p_residual[element_index].y + p_residual_bias[element_index].y;
        p_residual_out[element_index].z = p_residual[element_index].z + p_residual_bias[element_index].z;
        p_residual_out[element_index].w = p_residual[element_index].w + p_residual_bias[element_index].w;
        vals[it] = char4addfloat4_dequant(p_input[element_index], p_residual_out[element_index], dequant_scale);
        WelfordCombine(vals[it].x, &thread_mean, &thread_m2, &thread_count);
        WelfordCombine(vals[it].y, &thread_mean, &thread_m2, &thread_count);
        WelfordCombine(vals[it].z, &thread_mean, &thread_m2, &thread_count);
        WelfordCombine(vals[it].w, &thread_mean, &thread_m2, &thread_count);
    }

    // mean var
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
        float4 norm_value = compute_float4_norm_value(vals[it], mean, m2, hidden_size, epsilon,
                                                      p_scale[element_index], p_bias[element_index]);

        p_residual_out[element_index].x = norm_value.x;
        p_residual_out[element_index].y = norm_value.y;
        p_residual_out[element_index].z = norm_value.z;
        p_residual_out[element_index].w = norm_value.w;

        char4 res = float42char4(norm_value, quant_scale);
        p_output[element_index] = res;
    }
}

template <int THREAD_DATA_LEN>
__global__ void skipLayernormI8IF32OKernel(const int8_t *input, const float *scale, const float *bias,
                                        const float *residual_bias, float *output, float *residual, float* residual_out, 
                                        int hidden_size, float dequant_scale, float quant_scale,
                                        bool is_post_ln) {
    // register
    // process 2 data
    float4 vals[THREAD_DATA_LEN];
    int block_start = blockIdx.x * hidden_size / 4;
    char4 *p_input = (char4 *)input;
    float4 *p_output = (float4 *)output;
    float4 *p_residual = (float4 *)residual;
    float4 *p_residual_out = (float4 *)residual_out;
    float4 *p_scale = (float4 *)scale;
    float4 *p_bias = (float4 *)bias;
    float4 *p_residual_bias = (float4 *)residual_bias;
    // one line start
    p_input += block_start;
    p_output += block_start;
    p_residual += block_start;
    p_residual_out += block_start;

    float thread_m2 = 0;
    float thread_mean = 0;
    float thread_count = 0;

    // load data from global memory
#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * C10_WARP_SIZE;
         // vals = dequant(input) + residual + bias
        p_residual_out[element_index].x = p_residual[element_index].x + p_residual_bias[element_index].x;
        p_residual_out[element_index].y = p_residual[element_index].y + p_residual_bias[element_index].y;
        p_residual_out[element_index].z = p_residual[element_index].z + p_residual_bias[element_index].z;
        p_residual_out[element_index].w = p_residual[element_index].w + p_residual_bias[element_index].w;
        vals[it] = char4addfloat4_dequant(p_input[element_index], p_residual_out[element_index], dequant_scale);
        WelfordCombine(vals[it].x, &thread_mean, &thread_m2, &thread_count);
        WelfordCombine(vals[it].y, &thread_mean, &thread_m2, &thread_count);
        WelfordCombine(vals[it].z, &thread_mean, &thread_m2, &thread_count);
        WelfordCombine(vals[it].w, &thread_mean, &thread_m2, &thread_count);
    }

    // mean var
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
        float4 norm_value = compute_float4_norm_value(vals[it], mean, m2, hidden_size, epsilon,
                                                      p_scale[element_index], p_bias[element_index]);
        
        p_output[element_index].x = norm_value.x;
        p_output[element_index].y = norm_value.y;
        p_output[element_index].z = norm_value.z;
        p_output[element_index].w = norm_value.w;
    }
}


void skipLayerNormI8II8O(const int8_t *input, const  float *scale, const float *bias, const float *residual_bias, 
                       int8_t *output, float *residual, float* residual_out, int batch_tokens, int hidden_size, float dequant_scale,
                       float quant_scale, int max_thread_per_block, cudaStream_t stream,
                       bool is_post_ln) {

    if (hidden_size > 1024) {
        throw std::runtime_error("hidden_size should <= 1024");
    }
    if (hidden_size % C10_WARP_SIZE != 0) {
        throw std::runtime_error("hidden_size // C10_WARP_SIZE != 0");
    }
    dim3 gridSize(batch_tokens);
    dim3 blockSize(C10_WARP_SIZE);

    int num_warp = hidden_size / C10_WARP_SIZE / 4;

    switch (num_warp) {
        case 1:
            skipLayernormI8II8OKernel<1>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 2:
            skipLayernormI8II8OKernel<2>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 3:
            skipLayernormI8II8OKernel<3>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 4:
            skipLayernormI8II8OKernel<4>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 5:
            skipLayernormI8II8OKernel<5>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 6:
            skipLayernormI8II8OKernel<6>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 7:
            skipLayernormI8II8OKernel<7>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 8:
            skipLayernormI8II8OKernel<8>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 9:
            skipLayernormI8II8OKernel<9>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 10:
            skipLayernormI8II8OKernel<10>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 11:
            skipLayernormI8II8OKernel<11>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 12:
            skipLayernormI8II8OKernel<12>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 13:
            skipLayernormI8II8OKernel<13>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 14:
            skipLayernormI8II8OKernel<14>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 15:
            skipLayernormI8II8OKernel<15>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 16:
            skipLayernormI8II8OKernel<16>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        default:
            throw std::runtime_error("skipLayernormI8II8OKernel");
            break;
    }
}

void skipLayerNormI8IF32O(const int8_t *input, const  float *scale, const float *bias, const float *residual_bias,
                       float *output, float *residual, float* residual_out, int batch_tokens, int hidden_size, float dequant_scale,
                       float quant_scale, int max_thread_per_block, cudaStream_t stream,
                       bool is_post_ln) {
    if (hidden_size > 1024) {
        throw std::runtime_error("hidden_size should <= 1024");
    }
    if (hidden_size % C10_WARP_SIZE != 0) {
        throw std::runtime_error("hidden_size // C10_WARP_SIZE != 0");
    }
    dim3 gridSize(batch_tokens);
    dim3 blockSize(C10_WARP_SIZE);

    int num_warp = hidden_size / C10_WARP_SIZE / 4;

    switch (num_warp) {
        case 1:
            skipLayernormI8IF32OKernel<1>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 2:
            skipLayernormI8IF32OKernel<2>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 3:
            skipLayernormI8IF32OKernel<3>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 4:
            skipLayernormI8IF32OKernel<4>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 5:
            skipLayernormI8IF32OKernel<5>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 6:
            skipLayernormI8IF32OKernel<6>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 7:
            skipLayernormI8IF32OKernel<7>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 8:
            skipLayernormI8IF32OKernel<8>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 9:
            skipLayernormI8IF32OKernel<9>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 10:
            skipLayernormI8IF32OKernel<10>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 11:
            skipLayernormI8IF32OKernel<11>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 12:
            skipLayernormI8IF32OKernel<12>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 13:
            skipLayernormI8IF32OKernel<13>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 14:
            skipLayernormI8IF32OKernel<14>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 15:
            skipLayernormI8IF32OKernel<15>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        case 16:
            skipLayernormI8IF32OKernel<16>
                <<<gridSize, blockSize, 0, stream>>>(input, scale, bias, residual_bias, output, residual, residual_out, hidden_size,
                                                     dequant_scale, quant_scale, is_post_ln);
            break;
        default:
            throw std::runtime_error("skipLayernormI8II8OKernel");
            break;    
    }             
}

} // namespace bert
} // namespace nvinfer1::ixrt_plugin 