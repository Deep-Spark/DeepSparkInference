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
#include "fcPlugin.h"
using namespace nvinfer1::ixrt_plugin::backend;
namespace nvinfer1 {
namespace ixrt_plugin {
namespace bert {

template <int THREAD_DATA_LEN>
__global__ void dequant_gemm_without_bias(const int8_t* input, int8_t* output, int hidden_size, float dequant_scale,
                                          float quant_scale, int num_per_tca) {
    float4 val[THREAD_DATA_LEN];

    int block_start = blockIdx.x * hidden_size;
    input += block_start;
    output += block_start;

    char4* p_input = (char4*)input;
    char4* p_output = (char4*)output;

#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * num_per_tca;

        val[it].x = __int2float_rn(p_input[element_index].x) * dequant_scale;
        val[it].y = __int2float_rn(p_input[element_index].y) * dequant_scale;
        val[it].z = __int2float_rn(p_input[element_index].z) * dequant_scale;
        val[it].w = __int2float_rn(p_input[element_index].w) * dequant_scale;

        char4 res = float42char4(val[it], quant_scale);
        p_output[element_index] = res;
    }
}

template <int THREAD_DATA_LEN>
__global__ void dequant_gemm_with_bias(const int8_t* input, const float* bias, int8_t* output, int hidden_size,
                                       float dequant_scale, float quant_scale, int num_per_tca) {
    float4 val[THREAD_DATA_LEN];

    int block_start = blockIdx.x * hidden_size;
    input += block_start;
    output += block_start;

    char4* p_input = (char4*)input;
    float4* p_bias = (float4*)bias;
    char4* p_output = (char4*)output;

    float4 bias_val;
#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * num_per_tca;
        bias_val.x = p_bias[element_index].x;
        bias_val.y = p_bias[element_index].y;
        bias_val.z = p_bias[element_index].z;
        bias_val.w = p_bias[element_index].w;

        val[it].x = __int2float_rn(p_input[element_index].x) * dequant_scale + bias_val.x;
        val[it].y = __int2float_rn(p_input[element_index].y) * dequant_scale + bias_val.y;
        val[it].z = __int2float_rn(p_input[element_index].z) * dequant_scale + bias_val.z;
        val[it].w = __int2float_rn(p_input[element_index].w) * dequant_scale + bias_val.w;

        char4 res = float42char4(val[it], quant_scale);
        p_output[element_index] = res;
    }
}

template <int THREAD_DATA_LEN>
__global__ void dequant_gemm_with_bias(const int32_t* input, const float* bias, int8_t* output, int hidden_size,
                                       float quant_scale1, float dequant_scale, float quant_scale2, int num_per_tca) {
    float4 val[THREAD_DATA_LEN];

    int block_start = blockIdx.x * hidden_size;
    input += block_start;
    output += block_start;

    int4* p_input = (int4*)input;
    float4* p_bias = (float4*)bias;
    char4* p_output = (char4*)output;

    float4 bias_val;
#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * num_per_tca;
        bias_val.x = p_bias[element_index].x;
        bias_val.y = p_bias[element_index].y;
        bias_val.z = p_bias[element_index].z;
        bias_val.w = p_bias[element_index].w;

        char4 q_input;
        q_input.x = float2int8(p_input[element_index].x*1.0, quant_scale1);
        q_input.y = float2int8(p_input[element_index].y*1.0, quant_scale1);
        q_input.z = float2int8(p_input[element_index].z*1.0, quant_scale1);
        q_input.w = float2int8(p_input[element_index].w*1.0, quant_scale1);

        val[it].x = __int2float_rn(q_input.x) * dequant_scale + bias_val.x;
        val[it].y = __int2float_rn(q_input.y) * dequant_scale + bias_val.y;
        val[it].z = __int2float_rn(q_input.z) * dequant_scale + bias_val.z;
        val[it].w = __int2float_rn(q_input.w) * dequant_scale + bias_val.w;

        char4 res = float42char4(val[it], quant_scale2);
        p_output[element_index] = res;
    }
}

void dequantGemmWithoutBias(int8_t* input, int8_t* output, int batch_seq_len, int hidden_size, float dequant_scale,
                            float quant_scale, cudaStream_t stream) {
    if (hidden_size > 4096) {
        throw std::runtime_error("hidden_size should <= 4096");
    }
    if (hidden_size / 4 % C10_WARP_SIZE != 0) {
        throw std::runtime_error("hidden_size // C10_WARP_SIZE != 0");
    }
    int num_per_tca = 64;
    dim3 gridSize(batch_seq_len);
    dim3 blockSize(num_per_tca);

    int num_warp = hidden_size / num_per_tca / 4;

    switch (num_warp) {
        case 1:
            dequant_gemm_without_bias<1>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 2:
            dequant_gemm_without_bias<2>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 3:
            dequant_gemm_without_bias<3>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 4:
            dequant_gemm_without_bias<4>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 5:
            dequant_gemm_without_bias<5>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 6:
            dequant_gemm_without_bias<6>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 7:
            dequant_gemm_without_bias<7>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 8:
            dequant_gemm_without_bias<8>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 9:
            dequant_gemm_without_bias<9>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 10:
            dequant_gemm_without_bias<10>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 11:
            dequant_gemm_without_bias<11>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 12:
            dequant_gemm_without_bias<12>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 13:
            dequant_gemm_without_bias<13>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 14:
            dequant_gemm_without_bias<14>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 15:
            dequant_gemm_without_bias<15>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 16:
            dequant_gemm_without_bias<16>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        default:
            throw std::runtime_error("dequantGemmWithoutBias");
            break;
    }
}

void dequantGemmWithBias(int8_t* input, float* bias, int8_t* output, int batch_seq_len, int hidden_size,
                         float dequant_scale, float quant_scale, cudaStream_t stream) {
    if (hidden_size > 4096) {
        throw std::runtime_error("hidden_size should <= 4096");
    }
    if (hidden_size / 4 % C10_WARP_SIZE != 0) {
        throw std::runtime_error("hidden_size // C10_WARP_SIZE != 0");
    }
    int num_per_tca = 64;
    dim3 gridSize(batch_seq_len);
    dim3 blockSize(num_per_tca);

    int num_warp = hidden_size / num_per_tca / 4;

    switch (num_warp) {
        case 1:
            dequant_gemm_with_bias<1>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 2:
            dequant_gemm_with_bias<2>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 3:
            dequant_gemm_with_bias<3>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 4:
            dequant_gemm_with_bias<4>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 5:
            dequant_gemm_with_bias<5>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 6:
            dequant_gemm_with_bias<6>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 7:
            dequant_gemm_with_bias<7>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 8:
            dequant_gemm_with_bias<8>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 9:
            dequant_gemm_with_bias<9>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 10:
            dequant_gemm_with_bias<10>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 11:
            dequant_gemm_with_bias<11>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 12:
            dequant_gemm_with_bias<12>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 13:
            dequant_gemm_with_bias<13>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 14:
            dequant_gemm_with_bias<14>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 15:
            dequant_gemm_with_bias<15>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        case 16:
            dequant_gemm_with_bias<16>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, dequant_scale, quant_scale, num_per_tca);
            break;
        default:
            throw std::runtime_error("dequantGemmWithBias with int8_t input");
            break;
    }
}

void dequantGemmWithBias(int32_t* input, float* bias, int8_t* output, int batch_seq_len, int hidden_size,
                         float quant_scale1, float dequant_scale, float quant_scale2, cudaStream_t stream) {
    if (hidden_size > 4096) {
        throw std::runtime_error("hidden_size should <= 4096");
    }
    if (hidden_size / 4 % C10_WARP_SIZE != 0) {
        throw std::runtime_error("hidden_size // C10_WARP_SIZE != 0");
    }
    int num_per_tca = 64;
    dim3 gridSize(batch_seq_len);
    dim3 blockSize(num_per_tca);

    int num_warp = hidden_size / num_per_tca / 4;

    switch (num_warp) {
        case 1:
            dequant_gemm_with_bias<1>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 2:
            dequant_gemm_with_bias<2>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 3:
            dequant_gemm_with_bias<3>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 4:
            dequant_gemm_with_bias<4>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 5:
            dequant_gemm_with_bias<5>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 6:
            dequant_gemm_with_bias<6>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 7:
            dequant_gemm_with_bias<7>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 8:
            dequant_gemm_with_bias<8>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 9:
            dequant_gemm_with_bias<9>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 10:
            dequant_gemm_with_bias<10>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 11:
            dequant_gemm_with_bias<11>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 12:
            dequant_gemm_with_bias<12>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 13:
            dequant_gemm_with_bias<13>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 14:
            dequant_gemm_with_bias<14>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 15:
            dequant_gemm_with_bias<15>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        case 16:
            dequant_gemm_with_bias<16>
                <<<gridSize, blockSize, 0, stream>>>(input, bias, output, hidden_size, quant_scale1, dequant_scale, quant_scale2, num_per_tca);
            break;
        default:
            throw std::runtime_error("dequantGemmWithBias with int32_t input");
            break;
    }
}

template <int THREAD_DATA_LEN>
__global__ void quant_gemm(const int32_t* input, int8_t* output, int hidden_size, float quant_scale, int num_per_tca) {
    float4 val[THREAD_DATA_LEN];

    int block_start = blockIdx.x * hidden_size;
    input += block_start;
    output += block_start;

    int4* p_input = (int4*)input;
    char4* p_output = (char4*)output;

    float4 bias_val;
#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * num_per_tca;
        char4 q_input;
        q_input.x = float2int8(p_input[element_index].x*1.0, quant_scale);
        q_input.y = float2int8(p_input[element_index].y*1.0, quant_scale);
        q_input.z = float2int8(p_input[element_index].z*1.0, quant_scale);
        q_input.w = float2int8(p_input[element_index].w*1.0, quant_scale);

        p_output[element_index] = q_input;
    }
}

void quantGemm(int32_t* input, int8_t* output, int batch_seq_len, int hidden_size, float dequant_scale, cudaStream_t stream) {
    if (hidden_size > 4096) {
        throw std::runtime_error("hidden_size should <= 4096");
    }
    if (hidden_size / 4 % C10_WARP_SIZE != 0) {
        throw std::runtime_error("hidden_size // C10_WARP_SIZE != 0");
    }
    int num_per_tca = 64;
    dim3 gridSize(batch_seq_len);
    dim3 blockSize(num_per_tca);

    int num_warp = hidden_size / num_per_tca / 4;

    switch (num_warp) {
        case 1:
            quant_gemm<1>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 2:
            quant_gemm<2>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 3:
            quant_gemm<3>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 4:
            quant_gemm<4>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 5:
            quant_gemm<5>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 6:
            quant_gemm<6>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 7:
            quant_gemm<7>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 8:
            quant_gemm<8>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 9:
            quant_gemm<9>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 10:
            quant_gemm<10>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 11:
            quant_gemm<11>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 12:
            quant_gemm<12>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 13:
            quant_gemm<13>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 14:
            quant_gemm<14>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 15:
            quant_gemm<15>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 16:
            quant_gemm<16>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        default:
            throw std::runtime_error("quantGemm");
            break;
    }
}

}  // namespace bert
}  // namespace ixrt_plugin
}  // namespace nvinfer1
