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
#include "embLayerNormInt8Plugin.h"
#include "backend/bert/bert_helper.h"

namespace nvinfer1::ixrt_plugin {
using namespace backend;
namespace bert {

template <int THREAD_DATA_LEN>
__global__ void IxinferResidualI8O(const float *input, int8_t *output, int hidden_size, float quant_scale) {
    float4 vals[THREAD_DATA_LEN];
    int block_start = blockIdx.x * hidden_size;

    input += block_start;
    output += block_start;

    float4 *p_input = (float4 *)input;
    char4 *p_output = (char4 *)output;

#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * C10_WARP_SIZE;
        vals[it].x = p_input[element_index].x;
        vals[it].y = p_input[element_index].y;
        vals[it].z = p_input[element_index].z;
        vals[it].w = p_input[element_index].w;

        char4 res = float42char4(vals[it], quant_scale);
        p_output[element_index] = res;
    }
}

template <typename T>
void IxinferResidualI8OLauncher(const T *input, int8_t *output, int batch_tokens, int hidden_size, float quant_scale,
                                  cudaStream_t stream) {
    if (hidden_size > 4096) {
        throw std::runtime_error("hidden_size should <= 4096");
    }
    if (hidden_size / 4 % C10_WARP_SIZE != 0) {
        throw std::runtime_error("hidden_size // C10_WARP_SIZE != 0");
    }
    dim3 gridSize(batch_tokens);
    dim3 blockSize(C10_WARP_SIZE);

    int num_warp = hidden_size / C10_WARP_SIZE / 4;

    switch (num_warp) {
        case 1:
            IxinferResidualI8O<1><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 2:
            IxinferResidualI8O<2><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 3:
            IxinferResidualI8O<3><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 4:
            IxinferResidualI8O<4><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 5:
            IxinferResidualI8O<5><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 6:
            IxinferResidualI8O<6><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 7:
            IxinferResidualI8O<7><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 8:
            IxinferResidualI8O<8><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 9:
            IxinferResidualI8O<9><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 10:
            IxinferResidualI8O<10><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 11:
            IxinferResidualI8O<11><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 12:
            IxinferResidualI8O<12><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 13:
            IxinferResidualI8O<13><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 14:
            IxinferResidualI8O<14><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 15:
            IxinferResidualI8O<15><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        case 16:
            IxinferResidualI8O<16><<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, quant_scale);
            break;
        default:
            throw std::runtime_error("IxinferResidualI8OLauncher");
            break;
    }
}

template <int THREAD_DATA_LEN>
__global__ void IxinferBertEmbedLnKernel(const float *token_emb, const float *pos_emb, const float *type_emb, const int *tokens,
                                         float *output, int *pad_mask, int *type_ids, int pad_id, int batch_size,
                                         int seq_len, int hidden_dim, const float *scale, const float *bias) {
    float4 vals[THREAD_DATA_LEN];
    int block_start = blockIdx.x * hidden_dim;
    int batch_idx, seq_idx;
    batch_idx = blockIdx.x / seq_len;
    seq_idx = blockIdx.x % seq_len;

    int tokens_idx = blockIdx.x;
    int token = tokens[tokens_idx];
    int token_type = type_ids[tokens_idx];

    output += block_start;

    float4 *p_output = (float4 *)output;

    float4 *p_scale = (float4 *)scale;
    float4 *p_bias = (float4 *)bias;
    float4 *p_value = (float4 *)(token_emb + token * hidden_dim);
    float4 *p_pemb = (float4 *)(pos_emb + seq_idx * hidden_dim);
    float4 *p_temb = (float4 *)(type_emb + token_type * hidden_dim);

    float thread_m2 = 0;
    float thread_mean = 0;
    float thread_count = 0;

#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * C10_WARP_SIZE;

        if (token == pad_id) {
            if (element_index == 0) {
                pad_mask[tokens_idx] = 1;
            }
            vals[it] = make_float4(0.f, 0.f, 0.f, 0.f);

        } else {
            if (element_index == 0) {
                pad_mask[tokens_idx] = 0;
            }
        
            vals[it].x = p_value[element_index].x + p_pemb[element_index].x + p_temb[element_index].x;
            vals[it].y = p_value[element_index].y + p_pemb[element_index].y + p_temb[element_index].y;
            vals[it].z = p_value[element_index].z + p_pemb[element_index].z + p_temb[element_index].z;
            vals[it].w = p_value[element_index].w + p_pemb[element_index].w + p_temb[element_index].w;
            WelfordCombine(vals[it].x, &thread_mean, &thread_m2, &thread_count);
            WelfordCombine(vals[it].y, &thread_mean, &thread_m2, &thread_count);
            WelfordCombine(vals[it].z, &thread_mean, &thread_m2, &thread_count);
            WelfordCombine(vals[it].w, &thread_mean, &thread_m2, &thread_count);
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
        float4 scale_value = p_scale[element_index];
        float4 bias_value = p_bias[element_index];
        float4 norm_value = compute_float4_norm_value(vals[it], mean, m2, hidden_dim, epsilon,
                                                      scale_value, bias_value);
        int tokens_idx = blockIdx.x;

        int token = tokens[tokens_idx];
        if (token == pad_id) {
            p_output[element_index] = make_float4(0.f, 0.f, 0.f, 0.f);
        } else {
            p_output[element_index] = norm_value;
        }
    }
}


void IxinferBertEmbedLn(const float *token_emb, const float *pos_emb, const float *type_emb, const int *tokens, float *output,
                        int *pad_mask, int *type_ids, int pad_id, int batch_size, int seq_len, int hidden_size,
                        const float *scale, const float *bias, cudaStream_t stream) {
    if (hidden_size > 4096) {
        throw std::runtime_error("hidden_size should <= 4096");
    }
    if (hidden_size % C10_WARP_SIZE != 0) {
        throw std::runtime_error("hidden_size // C10_WARP_SIZE != 0");
    }
    int batch_tokens = batch_size * seq_len;
    dim3 gridSize(batch_tokens);
    dim3 blockSize(C10_WARP_SIZE);
    int num_warp = hidden_size / C10_WARP_SIZE / 4; 

    switch (num_warp) {
        case 1:
            IxinferBertEmbedLnKernel<1>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 2:
            IxinferBertEmbedLnKernel<2>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 3:
            IxinferBertEmbedLnKernel<3>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 4:
            IxinferBertEmbedLnKernel<4>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 5:
            IxinferBertEmbedLnKernel<5>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 6:
            IxinferBertEmbedLnKernel<6>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 7:
            IxinferBertEmbedLnKernel<7>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 8:
            IxinferBertEmbedLnKernel<8>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 9:
            IxinferBertEmbedLnKernel<9>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 10:
            IxinferBertEmbedLnKernel<10>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 11:
            IxinferBertEmbedLnKernel<11>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 12:
            IxinferBertEmbedLnKernel<12>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 13:
            IxinferBertEmbedLnKernel<13>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 14:
            IxinferBertEmbedLnKernel<14>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 15:
            IxinferBertEmbedLnKernel<15>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        case 16:
            IxinferBertEmbedLnKernel<16>
                <<<gridSize, blockSize, 0, stream>>>(token_emb, pos_emb, type_emb, tokens, output, pad_mask, type_ids,
                                                     pad_id, batch_size, seq_len, hidden_size, scale, bias);
            break;
        default:
            throw std::runtime_error("IxinferBertEmbedLn");
            break;
    }
}

cudaError_t embLayerNorm(cudaStream_t stream, int E, int B, int S, int32_t const* inputIds, int32_t const* segmentIds,
    float const* beta, float const* gamma, float const* wordEmb, float const* posEmb, float const* tokEmb, int32_t const wordSize,
    int32_t const tokSize, float* buffer, int8_t* output, int32_t* maskIdx, int32_t padId, float l0_qkv_in_amax)
{
    IxinferBertEmbedLn(wordEmb, posEmb, tokEmb, inputIds, buffer, maskIdx, (int*)segmentIds,
                                    padId, B, S, E, gamma, beta, stream);
                         
    IxinferResidualI8OLauncher<float>(buffer, output, B*S, E, 127.0 / l0_qkv_in_amax, stream);
    return cudaSuccess;
}

void __global__ IxinferMaskPadKernel(const int32_t* mask, int8_t* new_mask, int bsz,
                                    int ori_seq_len, int hsz, int fmha_seq_len) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;

    if (seq_idx < ori_seq_len) {
        if (threadIdx.x == 0) {
            new_mask[batch_idx * fmha_seq_len + seq_idx] = mask[batch_idx * ori_seq_len + seq_idx];
        }
    } else {
        new_mask[batch_idx * fmha_seq_len + seq_idx] = 1;
    }
} 

void IxinferMaskPad(int32_t* mask, int8_t* new_mask, int bsz, int ori_seq_len, int hsz,
                   int fmha_seq_len, int batch_tokens, cudaStream_t stream) {
    if (hsz / 2 > 4096) {
        throw std::runtime_error("hsz/2>4096");
    }
    if (hsz % 2 != 0) {
        throw std::runtime_error("hsz % 2 !=0");
    }
    if (ori_seq_len > fmha_seq_len) {
        throw std::runtime_error("ori_seq_len > fmha_seq_len");
    }
    if (bsz * ori_seq_len > batch_tokens) {
        throw std::runtime_error("bsz*ori_seq_len > batch_tokens");
    }
    dim3 blockSize(bsz, fmha_seq_len);
    IxinferMaskPadKernel<<<blockSize, hsz / 2, 0, stream>>>(mask, new_mask, bsz, ori_seq_len, hsz,
                                                           fmha_seq_len);
}

} // namespace bert
} // namespace nvinfer1::ixrt_plugin                  