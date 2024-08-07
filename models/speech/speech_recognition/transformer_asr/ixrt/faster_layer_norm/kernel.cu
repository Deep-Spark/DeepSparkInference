#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <stdexcept>

#include "transformer_helper.cuh"

namespace iluvatar::inferrt::transformer {

template <int UNROLL_FACTOR>
__global__ void LnOpt2Kernel(half* input, half* ln_weight, half* ln_bias,
                             half* output, int hidden_size,
                             float layernorm_eps) {
  input += blockIdx.x * hidden_size;
  output += blockIdx.x * hidden_size;

  half2* p_in = reinterpret_cast<half2*>(input);
  half2* p_out = reinterpret_cast<half2*>(output);
  half2* p_wei = reinterpret_cast<half2*>(ln_weight);
  half2* p_bias = reinterpret_cast<half2*>(ln_bias);
  int half_hidden_size = hidden_size / 2;

  extern __shared__ half2 shmem[];

  float s_mean;
  float s_variance;
  float x_sum = 0.0f;
  float x2_sum = 0.0f;
#pragma unroll UNROLL_FACTOR
  for (int i = 0; i < UNROLL_FACTOR; ++i) {
    int index = i * blockDim.x + threadIdx.x;
    if (index < half_hidden_size) {
      half2 value = p_in[index];
      shmem[index] = value;
      float val_1 = __half2float(value.x);
      float val_2 = __half2float(value.y);
      x_sum += val_1 + val_2;
      x2_sum += val_1 * val_1 + val_2 * val_2;
    }
  }
  float sums[2];  // 和，平方和
  sums[0] = x_sum;
  sums[1] = x2_sum;
  blockReduceSumV2<float, 2>(sums);

  s_mean = sums[0] / hidden_size;
  s_variance = rsqrtf(sums[1] / hidden_size - s_mean * s_mean + layernorm_eps);

#pragma unroll UNROLL_FACTOR
  for (int i = 0; i < UNROLL_FACTOR; ++i) {
    int index = i * blockDim.x + threadIdx.x;
    if (index < half_hidden_size) {
      half2 wei_value = p_wei[index];
      half2 bias_value = p_bias[index];
      half2 vals_value = shmem[index];

      float2 norm_value;
      norm_value.x = (__half2float(vals_value.x) - s_mean) * s_variance *
                         __half2float(wei_value.x) +
                     __half2float(bias_value.x);
      norm_value.y = (__half2float(vals_value.y) - s_mean) * s_variance *
                         __half2float(wei_value.y) +
                     __half2float(bias_value.y);

      __half2 res;
      res.x = __float2half(norm_value.x);
      res.y = __float2half(norm_value.y);

      p_out[index] = res;
    }
  }
}

// FasterTransformer/src/fastertransformer/kernels/layernorm_kernels.cu
void IxinferLnLauncherOpt2(__half* input, __half* ln_weight, __half* ln_bias,
                           __half* output, int batch_tokens, int hidden_size,
                           cudaStream_t stream) {
  const float layernorm_eps = 1e-5;
  if (hidden_size % 2 != 0) {
    throw std::runtime_error("layer norm error: hidden_size % 2 != 0");
  }
  dim3 grid(batch_tokens);
  int half_n = hidden_size / 2;
  int half_n_warp = (half_n + warpSize - 1) / warpSize * warpSize;
  dim3 block(std::min(half_n_warp, 1024));
  int rolls_per_thread = (half_n + block.x - 1) / block.x;
  switch (rolls_per_thread) {
    case 1:
      LnOpt2Kernel<1><<<grid, block, hidden_size * sizeof(half), stream>>>(
          input, ln_weight, ln_bias, output, hidden_size, layernorm_eps);
      break;
    case 2:
      LnOpt2Kernel<2><<<grid, block, hidden_size * sizeof(half), stream>>>(
          input, ln_weight, ln_bias, output, hidden_size, layernorm_eps);
      break;
    case 3:
      LnOpt2Kernel<3><<<grid, block, hidden_size * sizeof(half), stream>>>(
          input, ln_weight, ln_bias, output, hidden_size, layernorm_eps);
      break;
    case 4:
      LnOpt2Kernel<4><<<grid, block, hidden_size * sizeof(half), stream>>>(
          input, ln_weight, ln_bias, output, hidden_size, layernorm_eps);
      break;
    case 5:
      LnOpt2Kernel<5><<<grid, block, hidden_size * sizeof(half), stream>>>(
          input, ln_weight, ln_bias, output, hidden_size, layernorm_eps);
      break;
    case 6:
      LnOpt2Kernel<6><<<grid, block, hidden_size * sizeof(half), stream>>>(
          input, ln_weight, ln_bias, output, hidden_size, layernorm_eps);
      break;
    case 7:
      LnOpt2Kernel<7><<<grid, block, hidden_size * sizeof(half), stream>>>(
          input, ln_weight, ln_bias, output, hidden_size, layernorm_eps);
      break;
    case 8:
      LnOpt2Kernel<8><<<grid, block, hidden_size * sizeof(half), stream>>>(
          input, ln_weight, ln_bias, output, hidden_size, layernorm_eps);
      break;
    default:
      std::cout << "hidden_size: " << hidden_size << std::endl;
      throw std::runtime_error("layer norm error, unsupport hidden size! ");
      break;
  }
}
}  // namespace iluvatar::inferrt::transformer

std::vector<at::Tensor> one_test_opt(at::Tensor input, at::Tensor ln_weight,
                                     at::Tensor ln_bias) {
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.is_contiguous());

  TORCH_CHECK(ln_weight.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(ln_weight.is_cuda());
  TORCH_CHECK(ln_weight.is_contiguous());

  TORCH_CHECK(ln_bias.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(ln_bias.is_cuda());
  TORCH_CHECK(ln_bias.is_contiguous());

  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(ln_weight.dim() == 1);
  TORCH_CHECK(ln_bias.dim() == 1);

  int batch_tokens = input.size(0);
  int hidden_size = input.size(1);

  TORCH_CHECK(ln_weight.size(0) == hidden_size);
  TORCH_CHECK(ln_bias.size(0) == hidden_size);

  at::Tensor output = at::empty_like(input);

  half* p_in = (half*)input.data_ptr();
  half* p_wei = (half*)ln_weight.data_ptr();
  half* p_bias = (half*)ln_bias.data_ptr();
  half* p_out = (half*)output.data_ptr();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  iluvatar::inferrt::transformer::IxinferLnLauncherOpt2(
      p_in, p_wei, p_bias, p_out, batch_tokens, hidden_size, stream);
  return {output};
}
