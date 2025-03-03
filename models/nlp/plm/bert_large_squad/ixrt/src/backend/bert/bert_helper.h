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
#include <cuda.h>
#include <cuda_fp16.h>

#include <stdexcept>

#ifndef C10_WARP_SIZE

#ifdef __ILUVATAR__
#define C10_WARP_SIZE 64
#else
#define C10_WARP_SIZE 32
#endif

#endif

namespace nvinfer1 {
namespace ixrt_plugin {
namespace backend {

const float epsilon = 0.000000000001;
const unsigned int WARP_REDUCE_MASK = 0xffffffff;
const float CUDA_FLOAT_INF_NEG = -100000000.f;  // FIXME later
const float CUDA_FLOAT_INF_POS = 100000000.f;   // FIXME later
const int CUDA_INT_INF = 2147483647;
const int MAX_THREADS = 1024;

__forceinline__ __device__ int8_t float2int8(float x, float quant_scale) {
    float i8_f = x * quant_scale;
    int32_t i8 = floorf(i8_f + 0.5);
    i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
    return int8_t(i8);
}

inline __device__ void WelfordCombine(float val, float *mean, float *m2, float *count) {
    // Use Welford Online algorithem to compute mean and variance
    // For more details you can refer to:
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    *count += 1;
    float delta1 = val - *mean;
    *mean += delta1 / *count;
    float delta2 = val - *mean;
    *m2 += delta1 * delta2;
}

inline __device__ void WelfordCombine(float b_mean, float b_m2, float b_count, float *mean, float *m2, float *count) {
    if (b_count == 0) {
        return;
    }
    float new_count = *count + b_count;
    float nb_over_n = b_count / new_count;
    float delta = b_mean - *mean;
    *mean += delta * nb_over_n;
    *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
    *count = new_count;
}

__inline__ __device__ void WelfordWarpReduce(float thread_mean, float thread_m2, float thread_count, float *mean,
                                             float *m2, float *count) {
    *mean = thread_mean;
    *m2 = thread_m2;
    *count = thread_count;
    for (int mask = C10_WARP_SIZE / 2; mask > 0; mask /= 2) {
        float b_mean = __shfl_down_sync(0xffffffff, *mean, mask);
        float b_m2 = __shfl_down_sync(0xffffffff, *m2, mask);
        float b_count = __shfl_down_sync(0xffffffff, *count, mask);
        WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
    }
}
// addd by pxl
// block内所有数据完成reduce
//  template <int >
__inline__ __device__ void WelfordBlockAllReduce(float thread_mean, float thread_m2, float thread_count,
                                                 float *result_mean, float *result_m2, float *result_count) {
    __shared__ float mean_shared[C10_WARP_SIZE];
    __shared__ float m2_shared[C10_WARP_SIZE];
    __shared__ float count_shared[C10_WARP_SIZE];
    __shared__ float mean_result_broadcast;
    __shared__ float m2_result_broadcast;
    __shared__ float count_result_broadcast;

    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    float warp_mean = 0;
    float warp_m2 = 0;
    float warp_count = 0;
    WelfordWarpReduce(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
    __syncthreads();

    if (lid == 0) {
        mean_shared[wid] = warp_mean;
        m2_shared[wid] = warp_m2;
        count_shared[wid] = warp_count;
    }
    __syncthreads();

    if (wid == 0) {
        if (threadIdx.x < blockDim.x / C10_WARP_SIZE) {
            warp_mean = mean_shared[lid];
            warp_m2 = m2_shared[lid];
            warp_count = count_shared[lid];

        } else {
            warp_mean = 0.f;
            warp_m2 = 0.f;
            warp_count = 0.f;
        }
        __syncwarp();

        float block_mean = 0;
        float block_m2 = 0;
        float block_count = 0;

        WelfordWarpReduce(warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);

        if (lid == 0) {
            mean_result_broadcast = block_mean;
            m2_result_broadcast = block_m2;
            count_result_broadcast = block_count;
        }
    }
    __syncthreads();
    *result_mean = mean_result_broadcast;
    *result_m2 = m2_result_broadcast;
    *result_count = count_result_broadcast;
}
__forceinline__ __device__ char4 float42char4(float4 vals, float quant_scale) {
    char4 res;
    res.x = float2int8(vals.x, quant_scale);
    res.y = float2int8(vals.y, quant_scale);
    res.z = float2int8(vals.z, quant_scale);
    res.w = float2int8(vals.w, quant_scale);
    return res;
}

// load 两个 half2, 保存到 float4
__forceinline__ __device__ void load_float4_from_half(float4 &vals, __half2 *input, int index) {
    __half2 i1 = input[index * 2];
    __half2 i2 = input[index * 2 + 1];

    vals.x = __half2float(i1.x);
    vals.y = __half2float(i1.y);
    vals.z = __half2float(i2.x);
    vals.w = __half2float(i2.y);
}

/* Convert vector index to 3-dim tensor index */
__forceinline__ __host__ __device__ void decompose_3dim(int src, int dim1, int dim2, int *id0, int *id1, int *id2) {
    *id2 = src % dim2;
    src /= dim2;

    *id1 = src % dim1;
    *id0 = src / dim1;
}

__forceinline__ __device__ float4 compute_float4_norm_value(float4 vals, float mean, float m2, int hidden_size,
                                                            float epsilon, float4 scale, float4 bias) {
    float4 norm_value;
    norm_value.x =
        (vals.x - mean) * rsqrtf(m2 / hidden_size + epsilon) * scale.x + bias.x;
    norm_value.y =
        (vals.y - mean) * rsqrtf(m2 / hidden_size + epsilon) * scale.y + bias.y;
    norm_value.z =
        (vals.z - mean) * rsqrtf(m2 / hidden_size + epsilon) * scale.z + bias.z;
    norm_value.w =
        (vals.w - mean) * rsqrtf(m2 / hidden_size + epsilon) * scale.w + bias.w;
    return norm_value;
}

// for layer norm
__forceinline__ __device__ float4 compute_float4_norm_value(float4 vals, float mean, float m2, int hidden_size,
                                                            float epsilon, half2 scale_1, half2 scale_2, half2 bias_1,
                                                            half2 bias_2) {
    float4 norm_value;
    norm_value.x =
        (vals.x - mean) * rsqrtf(m2 / hidden_size + epsilon) * __half2float(scale_1.x) + __half2float(bias_1.x);
    norm_value.y =
        (vals.y - mean) * rsqrtf(m2 / hidden_size + epsilon) * __half2float(scale_1.y) + __half2float(bias_1.y);
    norm_value.z =
        (vals.z - mean) * rsqrtf(m2 / hidden_size + epsilon) * __half2float(scale_2.x) + __half2float(bias_2.x);
    norm_value.w =
        (vals.w - mean) * rsqrtf(m2 / hidden_size + epsilon) * __half2float(scale_2.y) + __half2float(bias_2.y);
    return norm_value;
}
/* Convert half2 into float2, mask inf and -inf */
__forceinline__ __host__ __device__ float safe_half_to_float(half hval) {
    return fmax(fmin(100000.f, __half2float(hval)), -100000.f);
}
__forceinline__ __device__ float4 char4addfloat4_dequant(char4 input_4, float4 residual,
                                                        float dequant_scale) {
    float4 res;
    res.x = __int2float_rn(input_4.x) * dequant_scale + residual.x;
    res.y = __int2float_rn(input_4.y) * dequant_scale + residual.y;
    res.z = __int2float_rn(input_4.z) * dequant_scale + residual.z;
    res.w = __int2float_rn(input_4.w) * dequant_scale + residual.w;
    return res;
}

__forceinline__ __device__ float4 char4addhalf2_dequant(char4 input_4, half2 residual_1, half2 residual_2,
                                                        float dequant_scale) {
    float4 res;
    res.x = __int2float_rn(input_4.x) * dequant_scale + safe_half_to_float(residual_1.x);
    res.y = __int2float_rn(input_4.y) * dequant_scale + safe_half_to_float(residual_1.y);
    res.z = __int2float_rn(input_4.z) * dequant_scale + safe_half_to_float(residual_2.x);
    res.w = __int2float_rn(input_4.w) * dequant_scale + safe_half_to_float(residual_2.y);
    return res;
}

// gelu
//  IxinferBiasGeluI8II8OKernel
template <typename T>
__forceinline__ __device__ T tanhf_exp(T x) {
    // float e1 = __expf(x);
    // float e2 = 1.0f / e1;
    // return (e1 - e2) / (e1 + e2);

    return (2.f / (1.f + __expf(-2.f * x)) - 1.f);
}

template <typename T>
__forceinline__ __device__ T gelu(T x) {
    float cdf = 0.5f * (1.0f + tanhf_exp((0.7978845608028654f * (x + 0.044715f * x * x * x))));
    return x * cdf;
}

// softmax
__forceinline__ __host__ __device__ int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}
template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width, unsigned int mask = 0xffffffff) {
#if !(defined(__HIP_PLATFORM_HCC__) || defined(__ILUVATAR__))
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
struct Add {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct Max {
    __device__ __forceinline__ T operator()(T a, T b) const { return a < b ? b : a; }
};
template <typename acc_t, int REDUCE_WARP_SIZE, template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t *sum) {
    ReduceOp<acc_t> r;
#pragma unroll
    for (int offset = REDUCE_WARP_SIZE / 2; offset > 0; offset /= 2) {
        acc_t b = WARP_SHFL_XOR(*sum, offset, REDUCE_WARP_SIZE);
        *sum = r(*sum, b);
    }
}
/* Convert 3-dim tensor index into vector index */
__forceinline__ __host__ __device__ int targetid_3dim(int id1, int id2, int id3, int dim2, int dim3) {
    return id1 * dim2 * dim3 + id2 * dim3 + id3;
}

/* Convert 4-dim tensor index into vector index */
__forceinline__ __host__ __device__ int targetid_4dim(int id1, int id2, int id3, int id4, int dim2, int dim3,
                                                      int dim4) {
    // return id1*(dim2*dim3*dim4) + id2*(dim3*dim4) + id3*dim4 + id4;
    int res = id4;

    int ld = dim4;
    res += id3 * ld;

    ld *= dim3;
    res += id2 * ld;

    ld *= dim2;
    res += id1 * ld;

    return res;
}

}  // namespace backend
}  // namespace ixrt_plugin
}  // namespace nvinfer1
