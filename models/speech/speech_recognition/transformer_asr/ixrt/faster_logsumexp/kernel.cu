#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <stdexcept>

namespace iluvatar::inferrt::transformer {

__global__ void LogSumExpWith2(half* input, half* output, int H) {
  half2* h2_in1 = reinterpret_cast<half2*>(input + blockIdx.x * 2 * H);
  half2* h2_in2 = reinterpret_cast<half2*>(input + blockIdx.x * 2 * H + H);
  half2* h2_out = reinterpret_cast<half2*>(output + blockIdx.x * H);

  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i < H / 2) {
    float2 res;
    half2 value1 = h2_in1[i];
    half2 value2 = h2_in2[i];

    res.x = std::log(__expf(__half2float(value1.x)) +
                     __expf(__half2float(value2.x)));
    res.y = std::log(__expf(__half2float(value1.y)) +
                     __expf(__half2float(value2.y)));

    half2 res_h2;
    res_h2.x = __float2half(res.x);
    res_h2.y = __float2half(res.y);
    h2_out[i] = res_h2;
  }
}

void IxinferLogSumExpLauncher(half* input, half* output, int N, int C, int H,
                              cudaStream_t stream) {
  const float layernorm_eps = 1e-5;
  if (H % 2 != 0) {
    throw std::runtime_error("IxinferLogSumExpLauncher: size error!");
  }
  int num_threads = 1024;
  int half_h = H / 2;
  int num_roll = (half_h - 1 + num_threads) / num_threads;
  dim3 grid(N, num_roll);
  dim3 block(num_threads);
  switch (C) {
    case 2:
      LogSumExpWith2<<<grid, block, 0, stream>>>(input, output, H);
      break;
    default:
      throw std::runtime_error(
          "IxinferLogSumExpLauncher error, unsupport size! ");
      break;
  }
}

// https://zhuanlan.zhihu.com/p/153535799
__global__ void LogSumExpDim0(half* input, half* output, int N, int H) {
  half2* h2_out = reinterpret_cast<half2*>(output);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float2 res;
  res.x = 0.f;
  res.y = 0.f;

  float2 max_values;
  max_values.x = -1000.f;
  max_values.y = -1000.f;

  for (int batch_idx = 0; batch_idx < N; batch_idx++) {
    half2* h2_in = reinterpret_cast<half2*>(input + batch_idx * H);
    half2 value = h2_in[i];

    if (max_values.x < __half2float(value.x)) {
      max_values.x = __half2float(value.x);
    }
    if (max_values.y < __half2float(value.y)) {
      max_values.y = __half2float(value.y);
    }
  }

  for (int batch_idx = 0; batch_idx < N; batch_idx++) {
    half2* h2_in = reinterpret_cast<half2*>(input + batch_idx * H);
    half2 value = h2_in[i];

    res.x += __expf(__half2float(value.x) - max_values.x);
    res.y += __expf(__half2float(value.y) - max_values.y);
  }

  half2 res_h2;
  res_h2.x = __float2half(std::log(res.x) + max_values.x);
  res_h2.y = __float2half(std::log(res.y) + max_values.y);

  h2_out[i] = res_h2;
}

void IxinferLogSumExpLauncher(half* input, half* output, int N, int H,
                              cudaStream_t stream) {
  if (H % 2 != 0) {
    throw std::runtime_error("IxinferLogSumExpLauncher: size error!");
  }
  int num_threads = 1024;
  int half_h = H / 2;
  int num_roll = (half_h - 1 + num_threads) / num_threads;
  dim3 grid(num_roll);
  dim3 block(num_threads);
  LogSumExpDim0<<<grid, block, 0, stream>>>(input, output, N, H);
}

}  // namespace iluvatar::inferrt::transformer

std::vector<at::Tensor> one_test_opt(at::Tensor input) {
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.is_contiguous());

  TORCH_CHECK(input.dim() == 3);

  int N = input.size(0);
  int C = input.size(1);
  int H = input.size(2);

  at::Tensor output = input.new_empty({N, H});

  half* p_in = (half*)input.data_ptr();
  half* p_out = (half*)output.data_ptr();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  iluvatar::inferrt::transformer::IxinferLogSumExpLauncher(p_in, p_out, N, C, H,
                                                           stream);
  return {output};
}

std::vector<at::Tensor> one_test_dim0(at::Tensor input) {
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.is_contiguous());

  TORCH_CHECK(input.dim() == 2);

  int N = input.size(0);
  int H = input.size(1);

  at::Tensor output = input.new_empty({H});

  half* p_in = (half*)input.data_ptr();
  half* p_out = (half*)output.data_ptr();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  iluvatar::inferrt::transformer::IxinferLogSumExpLauncher(p_in, p_out, N, H,
                                                           stream);
  return {output};
}