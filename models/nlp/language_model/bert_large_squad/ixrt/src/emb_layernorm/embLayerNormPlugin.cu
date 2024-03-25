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
#include "embLayerNormPlugin.h"
#include "backend/bert/bert_helper.h"

namespace nvinfer1::ixrt_plugin {
using namespace backend;
namespace bert {

template <int THREAD_DATA_LEN>
__global__ void IxinferBertEmbedLnKernel(const __half *token_emb, const __half *pos_emb, const __half *type_emb,
                                         const int *tokens, __half *output, int *pad_mask, int *type_ids, int pad_id,
                                         int batch_size, int seq_len, int hidden_dim, const __half *scale,
                                         const __half *bias) {
    float2 vals[THREAD_DATA_LEN];
    int block_start = blockIdx.x * hidden_dim;
    output += block_start;

    __half2 *p_output = (__half2 *)output;
    __half2 *p_scale = (__half2 *)scale;
    __half2 *p_bias = (__half2 *)bias;

    float thread_m2 = 0;
    float thread_mean = 0;
    float thread_count = 0;

#pragma unroll
    for (int it = 0; it < THREAD_DATA_LEN; ++it) {
        int element_index = threadIdx.x + it * C10_WARP_SIZE;

        int batch_idx, seq_idx, dim_idx;
        batch_idx = blockIdx.x / seq_len;
        seq_idx = blockIdx.x % seq_len;
        dim_idx = element_index;
        int tokens_idx = blockIdx.x;
        int token = tokens[tokens_idx];
        int token_type = type_ids[tokens_idx];

        half2 value;

        if (token == pad_id) {
            if (dim_idx == 0) {
                pad_mask[tokens_idx] = 1;
            }
            value.x = __float2half(0.f);
            value.y = __float2half(0.f);

        } else {
            if (dim_idx == 0) {
                pad_mask[tokens_idx] = 0;
            }
            value = ((half2 *)(token_emb + token * hidden_dim + dim_idx * 2))[0];
            half2 pemb = ((half2 *)(pos_emb + seq_idx * hidden_dim + dim_idx * 2))[0];
            half2 temb = ((half2 *)(type_emb + token_type * hidden_dim + dim_idx * 2))[0];

            vals[it].x = __half2float(value.x) + __half2float(pemb.x) + __half2float(temb.x);
            vals[it].y = __half2float(value.y) + __half2float(pemb.y) + __half2float(temb.y);

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
        m2 = rsqrtf(m2 / hidden_dim + epsilon);

#pragma unroll
        for (int it = 0; it < THREAD_DATA_LEN; ++it) {
            int element_index = threadIdx.x + it * C10_WARP_SIZE;

            __half2 scale_value = p_scale[element_index];
            __half2 bias_value = p_bias[element_index];

            float2 norm_value;
            norm_value.x = (vals[it].x - mean) * m2 * __half2float(scale_value.x) + __half2float(bias_value.x);
            norm_value.y = (vals[it].y - mean) * m2 * __half2float(scale_value.y) + __half2float(bias_value.y);

            __half2 res;
            res.x = __float2half(norm_value.x);
            res.y = __float2half(norm_value.y);

            int token = tokens[tokens_idx];
            if (token == pad_id) {
                res.x = __float2half(0.f);
                res.y = __float2half(0.f);
                p_output[element_index] = res;
            } else {
                p_output[element_index] = res;
            }
        }
    }
}

void IxinferBertEmbedLn(const half *token_emb, const half *pos_emb, const half *type_emb,
                                const int *tokens, half *output, int *pad_mask, int *type_ids, int pad_id,
                                int batch_size, int seq_len, int hidden_size, const half *scale, const half *bias,
                                cudaStream_t stream) {
    if (hidden_size > 2048) {
        throw std::runtime_error("hidden_size should <= 2048");
    }
    if (hidden_size / 2 % C10_WARP_SIZE != 0) {
        throw std::runtime_error("hidden_size / 2 // C10_WARP_SIZE != 0");
    }
    int batch_tokens = batch_size * seq_len;
    dim3 gridSize(batch_tokens);
    dim3 blockSize(C10_WARP_SIZE);

    int num_warp = hidden_size / C10_WARP_SIZE / 2;

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
    half const* beta, half const* gamma, half const* wordEmb, half const* posEmb, half const* tokEmb, int32_t const wordSize,
    int32_t const tokSize, half* output, int32_t* maskIdx, int32_t padId)
{
    IxinferBertEmbedLn(wordEmb, posEmb, tokEmb, inputIds, output, maskIdx, (int*)segmentIds,
                                    padId, B, S, E, gamma, beta, stream);
    return cudaSuccess;
}

void __global__ IxinferMaskPadKernel(const int32_t* mask, int32_t* new_mask, int bsz,
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

void IxinferMaskPad(int32_t* mask, int32_t* new_mask, int bsz, int ori_seq_len, int hsz,
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