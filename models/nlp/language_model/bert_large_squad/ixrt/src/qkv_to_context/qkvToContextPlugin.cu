#include "qkvToContextPlugin.h"
#include "backend/bert/bert_helper.h"
#ifdef __ILUVATAR__
#include "backend/ixinfer/ixinfer_gemm_helper.h"
#else
#include "backend/cublas/cublas_helper.h"
#endif

using namespace nvinfer1::ixrt_plugin::backend;

namespace nvinfer1::ixrt_plugin {
namespace bert {

void __global__ IxinferArrangeEncQkvKernel(half *ori_qkv, half *new_q, half *new_k, half *new_v,
                                           int head_dim, int head_num, int batch_seq_len, int fmha_seq_len) {
    int hidden_size = head_dim * head_num;
    int batch_id = blockIdx.x;
    int token_id = blockIdx.y;

    int i = threadIdx.x;  // 1个线程处理2个数据
    int head_id = (i * 2) / head_dim;
    int dim_id = (i * 2) % head_dim;

    half2 *p_ori_qkv = (half2 *)(ori_qkv + batch_id * batch_seq_len * hidden_size * 3 + token_id * hidden_size * 3);
    half2 *p_new_qkv;

    int target_id = batch_id * head_num * fmha_seq_len * head_dim + head_id * fmha_seq_len * head_dim +
                    token_id * head_dim + dim_id;
    /* q */
    p_new_qkv = (half2 *)(new_q + target_id);
    p_new_qkv[0] = p_ori_qkv[i];
    /* k */
    p_ori_qkv += hidden_size / 2;
    p_new_qkv = (half2 *)(new_k + target_id);
    p_new_qkv[0] = p_ori_qkv[i];
    /* v */
    p_ori_qkv += hidden_size / 2;
    p_new_qkv = (half2 *)(new_v + target_id);
    p_new_qkv[0] = p_ori_qkv[i];
}

void IxinferArrangeEncQkv(half *ori_qkv, half *new_q, half *new_k, half *new_v, int bsz,
                          int head_num, int head_dim, int ori_seq_len, int fmha_seq_len, cudaStream_t stream) {
    int hsz = head_num * head_dim;
    if (hsz / 2 > 4096) {
        throw std::runtime_error("hidden_size / 2 > 4096");
    }
    if (hsz % 2 != 0) {
        throw std::runtime_error("hsz % 2 != 0");
    }
    if (head_dim % 2 != 0) {
        throw std::runtime_error("head_dim %2 != 0");
    }
    dim3 blockSize(bsz, ori_seq_len);
    IxinferArrangeEncQkvKernel<<<blockSize, hsz / 2, 0, stream>>>(ori_qkv, new_q, new_k, new_v, head_dim,
                                                                  head_num, ori_seq_len, fmha_seq_len);
}

__global__ void IxinferEncAttnOutArrangeKernel(const half *ori_q, half *new_q, const int bsz, const int ori_seq_len,
                                               const int fmha_seq_len, const int head_num, const int head_dim) {
    half2 *p_ori_q = (half2 *)ori_q;
    half2 *p_new_q = (half2 *)new_q;

    int batch_token_num = ori_seq_len * head_dim * head_num;
    int hidden_size = head_dim * head_num;
    int date_length = bsz * ori_seq_len * head_num * head_dim;

    int elem_idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (elem_idx < date_length / 2) {
        int half_elem_idx = elem_idx * 2;

        int bsz_idx = half_elem_idx / batch_token_num;
        int seq_idx = half_elem_idx % batch_token_num / hidden_size;
        int head_idx = half_elem_idx % batch_token_num % hidden_size / head_dim;
        int dim_idx = half_elem_idx % batch_token_num % hidden_size % head_dim;

        int src_index = bsz_idx * head_num * fmha_seq_len * head_dim + head_idx * fmha_seq_len * head_dim +
                        seq_idx * head_dim + dim_idx;

        p_new_q[elem_idx] = p_ori_q[src_index / 2];

        elem_idx += gridDim.x * blockDim.x;
    }
}

void IxinferEncAttnOutArrange(half *ori_q, half *new_q, int bsz, int ori_seq_len, int fmha_seq_len, int head_num,
                              int head_dim, cudaStream_t stream) {
    if (bsz * ori_seq_len * head_num * head_dim % 2 != 0) {
        throw std::runtime_error("bsz * ori_seq_len * head_num * head_dim % 2 != 0");
    }
    int data_length = bsz * ori_seq_len * head_num * head_dim / 2;
    int num_threads = 512;
    int num_blocks = ((data_length - 1 + num_threads) / num_threads);
    num_blocks = std::min(num_blocks, 128);
    IxinferEncAttnOutArrangeKernel<<<num_blocks, num_threads, 0, stream>>>(ori_q, new_q, bsz, ori_seq_len, fmha_seq_len,
                                                                           head_num, head_dim);
}


template <int log2_elements>
__global__ void IxinferCorrelationSoftmaxEncselfKernel(__half *correlation, const int *src_padding_mask,
                                                       const int batch_seq_len) {
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int SOFT_WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / SOFT_WARP_SIZE;

    int head_num = blockDim.y;
    int seq_len = gridDim.y;
    int start_idx = (blockIdx.x * head_num * seq_len * batch_seq_len + threadIdx.y * seq_len * batch_seq_len +
                     blockIdx.y * batch_seq_len);

    half2 *p_correlation = (half2 *)(correlation + start_idx);
    int32_t *p_mask = (int32_t *)(src_padding_mask + blockIdx.x * batch_seq_len);

    int local_idx = threadIdx.x;

    float2 elements[WARP_ITERATIONS];
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
        int element_index = local_idx + it * SOFT_WARP_SIZE;
        if (element_index < batch_seq_len / 2) {
            half2 correlation_value = p_correlation[element_index];

            elements[it].x =
                p_mask[element_index * 2] ? -INFINITY : __half2float(correlation_value.x);
            elements[it].y = p_mask[element_index * 2 + 1] ? -INFINITY
                                                           : __half2float(correlation_value.y);

        } else {
            elements[it].x = -INFINITY;
            elements[it].y = -INFINITY;
        }
    }

    float max_value = elements[0].x;
    max_value = (max_value > elements[0].y) ? max_value : elements[0].y;

#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
        max_value = (max_value > elements[it].x) ? max_value : elements[it].x;
        max_value = (max_value > elements[it].y) ? max_value : elements[it].y;
    }

    warp_reduce<float, SOFT_WARP_SIZE, Max>(&max_value);

    float sum = 0.0f;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
        elements[it].x = __expf(elements[it].x - max_value);
        elements[it].y = __expf(elements[it].y - max_value);

        sum += (elements[it].x + elements[it].y);
    }

    warp_reduce<float, SOFT_WARP_SIZE, Add>(&sum);
    sum = 1.0f / sum;

#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
        int element_index = local_idx + it * SOFT_WARP_SIZE;
        half2 correlation_value;
        if (element_index < batch_seq_len / 2) {
            correlation_value.x = __float2half(elements[it].x * sum);
            correlation_value.y = __float2half(elements[it].y * sum);

            p_correlation[element_index] = correlation_value;

        } else {
            break;
        }
    }
}

void IxinferCorrelationSoftmaxEncself(int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
                                      __half *correlation, const int *src_padding_mask) {
    if (batch_seq_len > 4096) {
        throw std::runtime_error("batch_seq_len should <= 4096");
    }
    if (batch_seq_len % 2 != 0) {
        throw std::runtime_error("batch_seq_len % 2 != 0");
    }

    int log2_elements = log2_ceil(batch_seq_len / 2);
    int next_power_of_two = 1 << log2_elements;
    int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

    dim3 grid(batch_size, batch_seq_len);

    dim3 block(WARP_SIZE, head_num);

    switch (log2_elements) {
        case 0:
            IxinferCorrelationSoftmaxEncselfKernel<0>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;

        case 1:
            IxinferCorrelationSoftmaxEncselfKernel<1>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;

        case 2:
            IxinferCorrelationSoftmaxEncselfKernel<2>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;

        case 3:
            IxinferCorrelationSoftmaxEncselfKernel<3>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;

        case 4:
            IxinferCorrelationSoftmaxEncselfKernel<4>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;

        case 5:
            IxinferCorrelationSoftmaxEncselfKernel<5>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;

        case 6:
            IxinferCorrelationSoftmaxEncselfKernel<6>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;
        case 7:
            IxinferCorrelationSoftmaxEncselfKernel<7>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;
        case 8:
            IxinferCorrelationSoftmaxEncselfKernel<8>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;
        case 9:
            IxinferCorrelationSoftmaxEncselfKernel<9>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;
        case 10:
            IxinferCorrelationSoftmaxEncselfKernel<10>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;
        case 11:
            IxinferCorrelationSoftmaxEncselfKernel<11>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;
        case 12:
            IxinferCorrelationSoftmaxEncselfKernel<12>
                <<<grid, block, 0, stream>>>(correlation, src_padding_mask, batch_seq_len);
            break;
        default:
            throw std::runtime_error("IxinferCorrelationSoftmaxEncself NotImplementedError");
            break;
    }
}

#ifdef __ILUVATAR__
cudaError_t fused_multihead_attetion(half* qkv_buffer, int32_t* mask,
                              half* q_buffer, half* k_buffer, half* v_buffer, half* qkv_out,
                              int bsz, int head_dim, int head_num, int hsz, int ori_seq_len, int fmha_seq_len,
                              cuinferHandle_t &cuinfer_handle, cudaStream_t &stream) {
    /* qkv arrange*/
    // bsz,ori_seq_len,3*hsz -> 3*(bsz,head_num,fmha_seq_len,head_dim)
    IxinferArrangeEncQkv(qkv_buffer, q_buffer, k_buffer, v_buffer, bsz, head_num, head_dim, ori_seq_len,
                         fmha_seq_len, stream);

    cuinferTensorDescriptor_t qDesc, kDesc, vDesc, maskDesc, oDesc;
    cuinferDataType_t _cuinferCompType = cuinferDataType_t::CUINFER_DATA_FLOAT;
    cuinferDataType_t _cuinferDataType = cuinferDataType_t::CUINFER_DATA_HALF;
    cuinferDataType_t _cuinferMaskType = cuinferDataType_t::CUINFER_DATA_INT32;
    cuinferCreateTensorDescriptor(&qDesc);
    cuinferCreateTensorDescriptor(&kDesc);
    cuinferCreateTensorDescriptor(&vDesc);
    cuinferCreateTensorDescriptor(&maskDesc);
    cuinferCreateTensorDescriptor(&oDesc);

    cuinferSetTensor4dDescriptor(qDesc, cuinferTensorFormat_t::CUINFER_TENSOR_NCHW, _cuinferDataType, bsz, head_num,
                                    fmha_seq_len, head_dim);
    cuinferSetTensor4dDescriptor(kDesc, cuinferTensorFormat_t::CUINFER_TENSOR_NCHW, _cuinferDataType, bsz, head_num,
                                    fmha_seq_len, head_dim);
    cuinferSetTensor4dDescriptor(vDesc, cuinferTensorFormat_t::CUINFER_TENSOR_NCHW, _cuinferDataType, bsz, head_num,
                                    fmha_seq_len, head_dim);
    cuinferSetTensor4dDescriptor(maskDesc, cuinferTensorFormat_t::CUINFER_TENSOR_NCHW, _cuinferMaskType, bsz, 1, 1,
                                    fmha_seq_len);
    cuinferSetTensor4dDescriptor(oDesc, cuinferTensorFormat_t::CUINFER_TENSOR_NCHW, _cuinferDataType, bsz, head_num,
                                    fmha_seq_len, head_dim);

    cuinferFMHAParam fmha_param;
    cuinferFMHAForward(cuinfer_handle, fmha_param, _cuinferCompType, _cuinferDataType, _cuinferMaskType, qDesc,
                        q_buffer, kDesc, k_buffer, vDesc, v_buffer, maskDesc, mask, oDesc, q_buffer, true);
    
    IxinferEncAttnOutArrange(q_buffer, qkv_out, bsz, ori_seq_len, fmha_seq_len, head_num, head_dim, stream);
    return cudaSuccess;
}
#else
cudaError_t fused_multihead_attetion(half* qkv_buffer, int32_t* mask, 
                              half* q_buffer, half* k_buffer, half* v_buffer, half* qk_out, half* qkv_out,
                              int bsz, int head_dim, int head_num, int hsz, int ori_seq_len, int fmha_seq_len,
                              cublasLtHandle_t &blaslt_handle, cudaStream_t &stream) {
    /* qkv arrange*/
    // bsz,ori_seq_len,3*hsz -> 3*(bsz,head_num,fmha_seq_len,head_dim)
    IxinferArrangeEncQkv(qkv_buffer, q_buffer, k_buffer, v_buffer, bsz, head_num, head_dim, ori_seq_len,
                         fmha_seq_len, stream);

    cublaslt_gemm(k_buffer, q_buffer, qk_out, bsz * head_num, fmha_seq_len, fmha_seq_len, head_dim,
                    fmha_seq_len * head_dim, fmha_seq_len * head_dim, fmha_seq_len * fmha_seq_len, 1.0/sqrt(head_dim*1.0), blaslt_handle, stream);
 
    IxinferCorrelationSoftmaxEncself(bsz, fmha_seq_len, head_num, stream, qk_out, mask);
 
    cublaslt_gemm_nn(v_buffer, qk_out, q_buffer, bsz * head_num, head_dim, fmha_seq_len, fmha_seq_len,
                    fmha_seq_len * head_dim, fmha_seq_len * fmha_seq_len, fmha_seq_len * head_dim, 1.0f, blaslt_handle, stream);

    IxinferEncAttnOutArrange(q_buffer, qkv_out, bsz, ori_seq_len, fmha_seq_len, head_num, head_dim, stream);
    return cudaSuccess;                            
}
#endif
} // namespace bert
} // namespace nvinfer1::ixrt_plugin