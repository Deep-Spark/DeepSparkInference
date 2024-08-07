#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

namespace iluvatar {
namespace inferrt {
namespace transformer {

__forceinline__ int nearest_4(int x) {
  if (x % 4 == 0) {
    return x;
  } else {
    int padding = 4 - x % 4;
    return x + padding;
  }
}

__forceinline__ int nearest_2(int x) {
  if (x % 2 == 0) {
    return x;
  } else {
    int padding = 2 - x % 2;
    return x + padding;
  }
}

__forceinline__ int nearest_num(int x, int value) {
  if (x % value == 0) {
    return x;
  } else {
    int padding = value - x % value;
    return x + padding;
  }
}

__device__ int8_t float2int8(float x, float quant_scale) {
  float i8_f = x * quant_scale;
  int32_t i8 = floorf(i8_f + 0.5);
  i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
  return int8_t(i8);
}

__device__ void WelfordCombine(float val, float *mean, float *m2,
                               float *count) {
  // Use Welford Online algorithem to compute mean and variance
  // For more details you can refer to:
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  *count += 1;
  float delta1 = val - *mean;
  *mean += delta1 / *count;
  float delta2 = val - *mean;
  *m2 += delta1 * delta2;
}

__device__ void WelfordCombine(float b_mean, float b_m2, float b_count,
                               float *mean, float *m2, float *count) {
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

__device__ void WelfordWarpReduce(float thread_mean, float thread_m2,
                                  float thread_count, float *mean, float *m2,
                                  float *count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  for (int mask = warpSize / 2; mask > 0; mask /= 2) {
    float b_mean = __shfl_down_sync(0xffffffff, *mean, mask);
    float b_m2 = __shfl_down_sync(0xffffffff, *m2, mask);
    float b_count = __shfl_down_sync(0xffffffff, *count, mask);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

// load 两个 half2, 保存到 float4
__device__ void load_float4_from_half(float4 &vals, __half2 *input, int index) {
  __half2 i1 = input[index * 2];
  __half2 i2 = input[index * 2 + 1];

  vals.x = __half2float(i1.x);
  vals.y = __half2float(i1.y);
  vals.z = __half2float(i2.x);
  vals.w = __half2float(i2.y);
}

__device__ char4 float42char4(float4 vals, float quant_scale) {
  char4 res;
  res.x = float2int8(vals.x, quant_scale);
  res.y = float2int8(vals.y, quant_scale);
  res.z = float2int8(vals.z, quant_scale);
  res.w = float2int8(vals.w, quant_scale);
  return res;
}

__device__ float4 char4addhalf2_dequant(char4 input_4, half2 residual_1,
                                        half2 residual_2, float dequant_scale) {
  float4 res;
  res.x =
      __int2float_rn(input_4.x) * dequant_scale + __half2float(residual_1.x);
  res.y =
      __int2float_rn(input_4.y) * dequant_scale + __half2float(residual_1.y);
  res.z =
      __int2float_rn(input_4.z) * dequant_scale + __half2float(residual_2.x);
  res.w =
      __int2float_rn(input_4.w) * dequant_scale + __half2float(residual_2.y);
  return res;
}

__device__ float4 compute_float4_norm_value(float4 vals, float mean, float m2,
                                            int hidden_size, float epsilon,
                                            half2 scale_1, half2 scale_2,
                                            half2 bias_1, half2 bias_2) {
  float4 norm_value;
  norm_value.x = (vals.x - mean) * rsqrtf(m2 / hidden_size + epsilon) *
                     __half2float(scale_1.x) +
                 __half2float(bias_1.x);
  norm_value.y = (vals.y - mean) * rsqrtf(m2 / hidden_size + epsilon) *
                     __half2float(scale_1.y) +
                 __half2float(bias_1.y);
  norm_value.z = (vals.z - mean) * rsqrtf(m2 / hidden_size + epsilon) *
                     __half2float(scale_2.x) +
                 __half2float(bias_2.x);
  norm_value.w = (vals.w - mean) * rsqrtf(m2 / hidden_size + epsilon) *
                     __half2float(scale_2.y) +
                 __half2float(bias_2.y);
  return norm_value;
}

// softmax
__forceinline__ __host__ __device__ int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}
template <typename T>
__device__ T WARP_SHFL_XOR(T value, int laneMask, int width) {
  unsigned int mask = 0xffffffff;
#if !(defined(__HIP_PLATFORM_HCC__) || defined(__ILUVATAR__))
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
struct Add {
  __device__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct Max {
  __device__ T operator()(T a, T b) const { return a < b ? b : a; }
};
template <typename acc_t, int REDUCE_WARP_SIZE,
          template <typename> class ReduceOp>
__device__ void warp_reduce(acc_t *sum) {
  ReduceOp<acc_t> r;
#pragma unroll
  for (int offset = REDUCE_WARP_SIZE / 2; offset > 0; offset /= 2) {
    acc_t b = WARP_SHFL_XOR(*sum, offset, REDUCE_WARP_SIZE);
    *sum = r(*sum, b);
  }
}

__device__ void warp_argmax(float &value, int32_t &idx) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    float next_value = WARP_SHFL_XOR(value, offset, warpSize);
    float next_idx = WARP_SHFL_XOR(idx, offset, warpSize);
    if (next_value > value) {
      value = next_value;
      idx = next_idx;
    }
  }
}

// gelu
//  IxinferBiasGeluI8II8OKernel
template <typename T>
__device__ T tanhf_exp(T x) {
  // float e1 = __expf(x);
  // float e2 = 1.0f / e1;
  // return (e1 - e2) / (e1 + e2);

  return (2.f / (1.f + __expf(-2.f * x)) - 1.f);
}

template <typename T>
__device__ T gelu(T x) {
  float cdf =
      0.5f *
      (1.0f + tanhf_exp((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

/* fp16 gelu */
template <>
__forceinline__ __device__ __half2 gelu<__half2>(__half2 val) {
  __half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp = __half22float2(val);

  tmp.x =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

/* Convert vector index to 3-dim tensor index */
__forceinline__ __host__ __device__ void decompose_3dim(int src, int dim1,
                                                        int dim2, int *id0,
                                                        int *id1, int *id2) {
  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T *val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = warpSize / 2; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, warpSize);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T *val) {
  static __shared__ T shared[NUM][warpSize + 1];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = lane < (blockDim.x / warpSize);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

__inline__ __device__ void warpReduceSum2Number(float *x, float *y) {
#pragma unroll
  for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
    *x += __shfl_xor_sync(0xffffffff, *x, mask, warpSize);
    *y += __shfl_xor_sync(0xffffffff, *y, mask, warpSize);
  }
}

__inline__ __device__ void blockReduceSum2Number(float *x, float *y) {
  static __shared__ float shared[2][warpSize + 1];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  warpReduceSum2Number(x, y);
  if (lane == 0) {
    shared[0][wid] = *x;
    shared[1][wid] = *y;
  }
  __syncthreads();
  bool is_mask = lane < (blockDim.x / warpSize);
  *x = is_mask ? shared[0][lane] : 0.0f;
  *y = is_mask ? shared[0][lane] : 0.0f;

  warpReduceSum2Number(x, y);
}

}  // namespace transformer

}  // namespace inferrt
}  // namespace iluvatar
