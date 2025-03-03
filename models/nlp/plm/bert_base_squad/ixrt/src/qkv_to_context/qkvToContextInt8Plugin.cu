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
#include "backend/cublas/cublas_helper.h"
#ifdef __ILUVATAR__
#include "backend/ixinfer/ixinfer_gemm_helper.h"
#endif
#include "qkvToContextInt8Plugin.h"

using namespace nvinfer1::ixrt_plugin::backend;

namespace nvinfer1::ixrt_plugin {
namespace bert {
const int _max_thread_per_block = 1024;
const float _quant_range = 127.0;

__global__ void IxinferArrangeEncselfQkvI8II8ONoBias(const int8_t *ori_qkv, int8_t *new_qkv, int max_batch_dim,
                                                     int batch_seq_len, int dim_per_head, int head_num) {
    int hidden_size = dim_per_head * head_num;
    int batch_id = blockIdx.x / batch_seq_len;
    int token_id = blockIdx.x % batch_seq_len;

    int i = threadIdx.x;  // 1个线程处理4个数据

    int head_id = (i * 4) / dim_per_head;
    int dim_id = (i * 4) % dim_per_head;
    int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num, batch_seq_len, dim_per_head);

#pragma unroll
    for (int qkv_idx = 0; qkv_idx < 3; qkv_idx++) {
        char4 *p_ori_qkv = (char4 *)(ori_qkv + (blockIdx.x * 3 + qkv_idx) * hidden_size);
        int qkv_offset = max_batch_dim * qkv_idx;
        char4 *p_new_qkv = (char4 *)(new_qkv + qkv_offset + target_id);
        p_new_qkv[0] = p_ori_qkv[i];
    }
}

template <int log2_elements, int WARP_BATCH>
__global__ void IxinferCorrelationSoftmaxEncselfI8II8OKernel(int8_t *correlation, const int8_t *src_padding_mask,
                                                             int batch_seq_len, float quant_scale,
                                                             float dequant_scale) {
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int SOFT_WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / SOFT_WARP_SIZE;
    int local_idx = threadIdx.x;

    for (int warp_idx = 0; warp_idx < WARP_BATCH; ++warp_idx) {
        int start_idx = (blockIdx.x * gridDim.y * WARP_BATCH * gridDim.z * batch_seq_len +
                         (blockIdx.y + gridDim.y * warp_idx) * gridDim.z * batch_seq_len + blockIdx.z * batch_seq_len);

        char4 *p_correlation = (char4 *)(correlation + start_idx);
        char4 *p_src_padding_mask = (char4 *)(src_padding_mask + blockIdx.x * batch_seq_len);

        // load data from global memory
        // float
        float4 elements[WARP_ITERATIONS];
#pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            int element_index = local_idx + it * SOFT_WARP_SIZE;
            if (element_index < batch_seq_len / 4) {
                char4 mask = p_src_padding_mask[element_index];
                char4 correlation_value = p_correlation[element_index];

                elements[it].x =
                    mask.x ? -INFINITY : (float)correlation_value.x * dequant_scale;
                elements[it].y =
                    mask.y ? -INFINITY : (float)correlation_value.y * dequant_scale;
                elements[it].z =
                    mask.z ? -INFINITY : (float)correlation_value.z * dequant_scale;
                elements[it].w =
                    mask.w ? -INFINITY : (float)correlation_value.w * dequant_scale;

            } else {
                elements[it].x = -INFINITY;
                elements[it].y = -INFINITY;
                elements[it].z = -INFINITY;
                elements[it].w = -INFINITY;
            }
        }

        // compute max_value
        float max_value = elements[0].x;
        max_value = (max_value > elements[0].y) ? max_value : elements[0].y;
        max_value = (max_value > elements[0].z) ? max_value : elements[0].z;
        max_value = (max_value > elements[0].w) ? max_value : elements[0].w;

#pragma unroll
        for (int it = 1; it < WARP_ITERATIONS; ++it) {
            max_value = (max_value > elements[it].x) ? max_value : elements[it].x;
            max_value = (max_value > elements[it].y) ? max_value : elements[it].y;
            max_value = (max_value > elements[it].z) ? max_value : elements[it].z;
            max_value = (max_value > elements[it].w) ? max_value : elements[it].w;
        }

        warp_reduce<float, SOFT_WARP_SIZE, Max>(&max_value);

        // exp sum
        float sum = 0.0f;
#pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            elements[it].x = __expf(elements[it].x - max_value);
            elements[it].y = __expf(elements[it].y - max_value);
            elements[it].z = __expf(elements[it].z - max_value);
            elements[it].w = __expf(elements[it].w - max_value);

            sum += (elements[it].x + elements[it].y + elements[it].z + elements[it].w);
        }

        warp_reduce<float, SOFT_WARP_SIZE, Add>(&sum);
        sum = 1.0f / sum;
        // store result
#pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            int element_index = local_idx + it * SOFT_WARP_SIZE;
            char4 correlation_value;
            if (element_index < batch_seq_len / 4) {
                correlation_value.x = float2int8(elements[it].x * sum, quant_scale);
                correlation_value.y = float2int8(elements[it].y * sum, quant_scale);
                correlation_value.z = float2int8(elements[it].z * sum, quant_scale);
                correlation_value.w = float2int8(elements[it].w * sum, quant_scale);

                p_correlation[element_index] = correlation_value;

            } else {
                break;
            }
        }
    }
}

void IxinferCorrelationSoftmaxEncselfI8II8O(int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
                                            int8_t *correlation, const int8_t *src_padding_mask, float quant_scale,
                                            float dequant_scale) {
    const int NUM_INT8_SOFTMAX_BATCH_WARP = 4;
    if (batch_seq_len > 512) {
        throw std::runtime_error("batch_seq_len should <= 512");
    }
    if (head_num % NUM_INT8_SOFTMAX_BATCH_WARP != 0) {
        throw std::runtime_error("head_num % NUM_INT8_SOFTMAX_BATCH_WARP !0");
    }
    if (batch_seq_len % 4 != 0) {
        throw std::runtime_error("batch_seq_len % 4 != 0");
    }

    int log2_elements = log2_ceil(batch_seq_len / 4);
    int next_power_of_two = 1 << log2_elements;
    int SOFT_WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    // dim3 blockSize(batch_size, head_num / NUM_INT8_SOFTMAX_BATCH_WARP,
    // batch_seq_len);
    //
    dim3 grid(batch_size, head_num / NUM_INT8_SOFTMAX_BATCH_WARP, batch_seq_len);

    dim3 block(SOFT_WARP_SIZE);

    switch (log2_elements) {
        case 0:
            IxinferCorrelationSoftmaxEncselfI8II8OKernel<0, NUM_INT8_SOFTMAX_BATCH_WARP>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len, quant_scale, dequant_scale);

            break;

        case 1:
            IxinferCorrelationSoftmaxEncselfI8II8OKernel<1, NUM_INT8_SOFTMAX_BATCH_WARP>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len, quant_scale, dequant_scale);
            break;

        case 2:
            IxinferCorrelationSoftmaxEncselfI8II8OKernel<2, NUM_INT8_SOFTMAX_BATCH_WARP>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len, quant_scale, dequant_scale);
            break;

        case 3:
            IxinferCorrelationSoftmaxEncselfI8II8OKernel<3, NUM_INT8_SOFTMAX_BATCH_WARP>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len, quant_scale, dequant_scale);
            break;

        case 4:
            IxinferCorrelationSoftmaxEncselfI8II8OKernel<4, NUM_INT8_SOFTMAX_BATCH_WARP>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len, quant_scale, dequant_scale);
            break;

        case 5:
            IxinferCorrelationSoftmaxEncselfI8II8OKernel<5, NUM_INT8_SOFTMAX_BATCH_WARP>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len, quant_scale, dequant_scale);
            break;

        case 6:
            IxinferCorrelationSoftmaxEncselfI8II8OKernel<6, NUM_INT8_SOFTMAX_BATCH_WARP>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len, quant_scale, dequant_scale);
            break;
        case 7:
            IxinferCorrelationSoftmaxEncselfI8II8OKernel<7, NUM_INT8_SOFTMAX_BATCH_WARP>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len, quant_scale, dequant_scale);
            break;
        case 8:
            IxinferCorrelationSoftmaxEncselfI8II8OKernel<8, NUM_INT8_SOFTMAX_BATCH_WARP>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len, quant_scale, dequant_scale);
            break;
        case 9:
            IxinferCorrelationSoftmaxEncselfI8II8OKernel<9, NUM_INT8_SOFTMAX_BATCH_WARP>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len, quant_scale, dequant_scale);
            break;
        default:
            throw std::runtime_error(
                "ker_correlation_softmax_encself_i8I_i8O_ix_ "
                "NotImplementedError");
            break;
    }
}


__global__ void IxinferArrangeAttenOutputI8II8OKernel(const int8_t *ori_q, int8_t *new_q, int beam_size,
                                                      int dim_per_head, int head_num, float quant_scale,
                                                      float dequant_scale) {
    int hidden_size = dim_per_head * head_num;

#pragma unroll
    for (int blockin = 0; blockin < 4; blockin++) {
        int batch_id = (blockIdx.x * 4 + blockin) / beam_size;
        // note, for encoder, beam_id is token_id; for decoder, beam_id is beam_id
        int beam_id = (blockIdx.x * 4 + blockin) % beam_size;
        int i = threadIdx.x;
        int out_index = (blockIdx.x * 4 + blockin) * hidden_size + i;
        int head_id = i / dim_per_head;
        int dim_id = i % dim_per_head;

        char4 *p_ori_q = (char4 *)ori_q;
        char4 *p_new_q = (char4 *)new_q;
        char4 value;

        value = p_ori_q[targetid_4dim(batch_id, head_id, beam_id, dim_id, head_num, beam_size, dim_per_head)];
        value.x = float2int8(value.x * dequant_scale, quant_scale);
        value.y = float2int8(value.y * dequant_scale, quant_scale);
        value.z = float2int8(value.z * dequant_scale, quant_scale);
        value.w = float2int8(value.w * dequant_scale, quant_scale);
        p_new_q[out_index] = value;
    }
}

void IxinferArrangeAttenOutputI8II8O(int batch_token_num, int hidden_size, cudaStream_t stream, const int8_t *ori_q,
                                     int8_t *new_q, int beam_size, int dim_per_head, int head_num,
                                     int max_thread_per_block, float quant_scale, float dequant_scale) {
    int qual_hidden_size = hidden_size >> 2;
    int qual_dim_per_head = dim_per_head >> 2;
    IxinferArrangeAttenOutputI8II8OKernel<<<batch_token_num / 4, qual_hidden_size, 0, stream>>>(
        ori_q, new_q, beam_size, qual_dim_per_head, head_num, quant_scale, dequant_scale);
}

#ifdef __ILUVATAR__
cudaError_t fused_multihead_attetion_int8(int8_t* qkv_buffer, int8_t* mask, int8_t* q_buffer, int8_t* k_buffer,
                                          int8_t* v_buffer, int8_t* qkv_out, int8_t* qk_buffer,
                                          int batch_size, int batch_seq_len, int head_dim, int head_num,
                                          int hidden_size, float arrange_qkv_amax, float softmax_in_amax,
                                          float softmax_out_amax, float linear_in_amax, cuinferHandle_t& cuinfer_handle,
                                          cudaStream_t& stream) {
    int batch_token_num = batch_size * batch_seq_len;
    int max_batch_dim = batch_token_num * hidden_size;

    float scaleCtx = linear_in_amax / _quant_range;
    float scaleArrange = arrange_qkv_amax / _quant_range;
    float scaleSoftin = softmax_in_amax / _quant_range;
    float scaleSoftout = softmax_out_amax / _quant_range;

    float scaleBmm1 = scaleArrange * scaleArrange / scaleSoftin * sqrt(1.f / head_dim);
    float scaleBmm2 = scaleSoftout * scaleArrange / scaleCtx;

    IxinferArrangeEncselfQkvI8II8ONoBias<<<batch_token_num, hidden_size / 4, 0, stream>>>(
        qkv_buffer, q_buffer, max_batch_dim, batch_seq_len, head_dim, head_num);

    switch (head_dim) {
        case 64:
        case 128:
        case 192:
        case 256: {
            cuinferFlashAttnConfigInfo flashAttnInfo;
            flashAttnInfo.scaling = sqrt(1.f / (head_dim * 1.0));
            flashAttnInfo.quantParam.q_amax = arrange_qkv_amax;
            flashAttnInfo.quantParam.k_amax = arrange_qkv_amax;
            flashAttnInfo.quantParam.v_amax = arrange_qkv_amax;
            flashAttnInfo.quantParam.p_amax = softmax_out_amax;
            flashAttnInfo.quantParam.o_amax = linear_in_amax;

            cuinferTensorDescriptor_t qDesc, kDesc, vDesc, maskDesc, oDesc;
            CUINFER_CHECK(cuinferCreateTensorDescriptor(&qDesc));
            CUINFER_CHECK(cuinferCreateTensorDescriptor(&kDesc));
            CUINFER_CHECK(cuinferCreateTensorDescriptor(&vDesc));
            CUINFER_CHECK(cuinferCreateTensorDescriptor(&maskDesc));
            CUINFER_CHECK(cuinferCreateTensorDescriptor(&oDesc));

            CUINFER_CHECK(cuinferSetTensor4dDescriptor(qDesc, cuinferTensorFormat_t::CUINFER_TENSOR_NCHW,
                                                       CUINFER_DATA_INT8, batch_size, head_num, batch_seq_len,
                                                       head_dim));
            CUINFER_CHECK(cuinferSetTensor4dDescriptor(kDesc, cuinferTensorFormat_t::CUINFER_TENSOR_NCHW,
                                                       CUINFER_DATA_INT8, batch_size, head_num, batch_seq_len,
                                                       head_dim));
            CUINFER_CHECK(cuinferSetTensor4dDescriptor(vDesc, cuinferTensorFormat_t::CUINFER_TENSOR_NCHW,
                                                       CUINFER_DATA_INT8, batch_size, head_num, batch_seq_len,
                                                       head_dim));
            CUINFER_CHECK(cuinferSetTensor4dDescriptor(maskDesc, cuinferTensorFormat_t::CUINFER_TENSOR_NCHW,
                                                       CUINFER_DATA_INT8, batch_size, 1, 1, batch_seq_len));
            CUINFER_CHECK(cuinferSetTensor4dDescriptor(oDesc, cuinferTensorFormat_t::CUINFER_TENSOR_NCHW,
                                                       CUINFER_DATA_INT8, batch_size, head_num, batch_seq_len,
                                                       head_dim));

            CUINFER_CHECK(cuinferFMHAForwardEx(cuinfer_handle, flashAttnInfo, qDesc, q_buffer, kDesc, k_buffer, vDesc,
                                               v_buffer, maskDesc, mask, oDesc, qk_buffer));
            break;
        }
        default: {
            cuinfer_i8_gemm(k_buffer, q_buffer, nullptr, qkv_buffer, batch_size * head_num, batch_seq_len,
                            batch_seq_len, head_dim, batch_seq_len * head_dim, batch_seq_len * head_dim,
                            batch_seq_len * batch_seq_len, scaleBmm1, 0.0, 0, cuinfer_handle, stream);

            IxinferCorrelationSoftmaxEncselfI8II8O(batch_size, batch_seq_len, head_num, stream, qkv_buffer, mask,
                                                   1.0 / scaleSoftout, scaleSoftin);

            cuinfer_nn_i8_gemm(v_buffer, qkv_buffer, qk_buffer, batch_size * head_num, head_dim, batch_seq_len,
                               batch_seq_len, batch_seq_len * head_dim, batch_seq_len * batch_seq_len,
                               batch_seq_len * head_dim, scaleBmm2, cuinfer_handle, stream);
            break;
        }
    }

    IxinferArrangeAttenOutputI8II8O(batch_token_num, hidden_size, stream, qk_buffer, qkv_out, batch_seq_len, head_dim,
                                    head_num, _max_thread_per_block, 1.f, 1.f);
    return cudaSuccess;
}
#else
template <int THREAD_DATA_LEN>
__global__ void quant_qkv_gemm(const int32_t* input, int8_t* output, int hidden_size, float quant_scale, int num_per_tca) {
    float4 val[THREAD_DATA_LEN];

    int block_id = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    int block_start = block_id * hidden_size;
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

void quantQKVGemm(int32_t* input, int8_t* output, int batch_size, int head_num, int batch_seq_len, int hidden_size, float dequant_scale, cudaStream_t stream) {
    if (hidden_size > 4096) {
        throw std::runtime_error("hidden_size should <= 4096");
    }
    int num_per_tca = min(hidden_size / 4, C10_WARP_SIZE); 
    dim3 gridSize(batch_size, head_num, batch_seq_len);
    dim3 blockSize(num_per_tca);

    int num_warp = hidden_size / num_per_tca / 4;
    switch (num_warp) {
        case 1:
            quant_qkv_gemm<1>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 2:
            quant_qkv_gemm<2>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 3:
            quant_qkv_gemm<3>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 4:
            quant_qkv_gemm<4>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 5:
            quant_qkv_gemm<5>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 6:
            quant_qkv_gemm<6>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 7:
            quant_qkv_gemm<7>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 8:
            quant_qkv_gemm<8>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 9:
            quant_qkv_gemm<9>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 10:
            quant_qkv_gemm<10>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 11:
            quant_qkv_gemm<11>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 12:
            quant_qkv_gemm<12>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 13:
            quant_qkv_gemm<13>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 14:
            quant_qkv_gemm<14>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 15:
            quant_qkv_gemm<15>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        case 16:
            quant_qkv_gemm<16>
                <<<gridSize, blockSize, 0, stream>>>(input, output, hidden_size, dequant_scale, num_per_tca);
            break;
        default:
            throw std::runtime_error("quantQKVGemm");
            break;
    }
}


cudaError_t fused_multihead_attetion_int8(int8_t *qkv_buffer, int8_t *mask, int8_t *q_buffer, int8_t *k_buffer,
                                          int8_t *v_buffer, int32_t *qk_out, int8_t *qkv_out, int8_t *qk_buffer, int batch_size,
                                          int batch_seq_len, int head_dim, int head_num, int hidden_size,
                                          float arrange_qkv_amax, float softmax_in_amax, float softmax_out_amax,
                                          float linear_in_amax, cublasLtHandle_t &cublas_lt_handle,
                                          cudaStream_t &stream) {
    int batch_token_num = batch_size * batch_seq_len;
    int max_batch_dim = batch_token_num * hidden_size;

    float scaleCtx = linear_in_amax / _quant_range;
    float scaleArrange = arrange_qkv_amax / _quant_range;
    float scaleSoftin = softmax_in_amax / _quant_range;
    float scaleSoftout = softmax_out_amax / _quant_range;

    float scaleBmm1 = scaleArrange * scaleArrange / scaleSoftin * sqrt(1.f / head_dim);
    float scaleBmm2 = scaleSoftout * scaleArrange / scaleCtx;

    IxinferArrangeEncselfQkvI8II8ONoBias<<<batch_token_num, hidden_size / 4, 0, stream>>>(
        qkv_buffer, q_buffer, max_batch_dim, batch_seq_len, head_dim, head_num);

    cublaslt_gemm(k_buffer, q_buffer, qk_out, batch_size * head_num, batch_seq_len, batch_seq_len, head_dim,
                  batch_seq_len * head_dim, batch_seq_len * head_dim, batch_seq_len * batch_seq_len, 1,
                  cublas_lt_handle, stream);
    quantQKVGemm(qk_out, qk_buffer, batch_size, head_num, batch_seq_len, batch_seq_len, scaleBmm1, stream);

    IxinferCorrelationSoftmaxEncselfI8II8O(batch_size, batch_seq_len, head_num, stream, qk_buffer, mask,
                                           1.0 / scaleSoftout, scaleSoftin);

    cublaslt_gemm_nn(v_buffer, qk_buffer, qk_out, batch_size * head_num, head_dim, batch_seq_len, batch_seq_len,
                     batch_seq_len * head_dim, batch_seq_len * batch_seq_len, batch_seq_len * head_dim, 1,
                     cublas_lt_handle, stream);
    quantQKVGemm(qk_out, q_buffer, batch_size, head_num, batch_seq_len, head_dim, scaleBmm2, stream);

    IxinferArrangeAttenOutputI8II8O(batch_token_num, hidden_size, stream, q_buffer, qkv_out, batch_seq_len, head_dim,
                                    head_num, _max_thread_per_block, 1.f, 1.f);
    return cudaSuccess;
}
#endif
}  // namespace bert
}  // namespace nvinfer1::ixrt_plugin
