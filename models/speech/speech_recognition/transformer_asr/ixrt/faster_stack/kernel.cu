#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <stdexcept>

namespace iluvatar::inferrt::transformer {

__global__ void Stack(half* a, half* b, half* c, half* d, half* output, int H) {
  half2* h2_a = reinterpret_cast<half2*>(a);
  half2* h2_b = reinterpret_cast<half2*>(b);
  half2* h2_c = reinterpret_cast<half2*>(c);
  half2* h2_d = reinterpret_cast<half2*>(d);

  half2* h2_out_a = reinterpret_cast<half2*>(output);
  half2* h2_out_b = reinterpret_cast<half2*>(output + H);
  half2* h2_out_c = reinterpret_cast<half2*>(output + H * 2);
  half2* h2_out_d = reinterpret_cast<half2*>(output + H * 3);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < H / 2) {
    h2_out_a[i] = h2_a[i];
    h2_out_b[i] = h2_b[i];
    h2_out_c[i] = h2_c[i];
    h2_out_d[i] = h2_d[i];
  }
}

void IxinferStackLauncher(half* a, half* b, half* c, half* d, half* output,
                          int H, cudaStream_t stream) {
  if (H % 2 != 0) {
    throw std::runtime_error("IxinferStackLauncher: size error!");
  }
  int num_threads = 1024;
  int half_h = H / 2;
  int num_roll = (half_h - 1 + num_threads) / num_threads;
  dim3 grid(num_roll);
  dim3 block(num_threads);
  Stack<<<grid, block, 0, stream>>>(a, b, c, d, output, H);
}

__global__ void Stack(half* a, half* b, half* output, int H) {
  half2* h2_a = reinterpret_cast<half2*>(a);
  half2* h2_b = reinterpret_cast<half2*>(b);

  half2* h2_out_a = reinterpret_cast<half2*>(output);
  half2* h2_out_b = reinterpret_cast<half2*>(output + H);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < H / 2) {
    h2_out_a[i] = h2_a[i];
    h2_out_b[i] = h2_b[i];
  }
}

void IxinferStackLauncher(half* a, half* b, half* output, int H,
                          cudaStream_t stream) {
  if (H % 2 != 0) {
    throw std::runtime_error("IxinferStackLauncher: size error!");
  }
  int num_threads = 1024;
  int half_h = H / 2;
  int num_roll = (half_h - 1 + num_threads) / num_threads;
  dim3 grid(num_roll);
  dim3 block(num_threads);
  Stack<<<grid, block, 0, stream>>>(a, b, output, H);
}

}  // namespace iluvatar::inferrt::transformer

std::vector<at::Tensor> one_test_opt(at::Tensor a, at::Tensor b, at::Tensor c,
                                     at::Tensor d) {
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.is_contiguous());

  TORCH_CHECK(b.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.is_contiguous());

  TORCH_CHECK(c.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.is_contiguous());

  TORCH_CHECK(d.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(d.is_cuda());
  TORCH_CHECK(d.is_contiguous());

  TORCH_CHECK(a.dim() == 1);
  TORCH_CHECK(b.dim() == 1);
  TORCH_CHECK(c.dim() == 1);
  TORCH_CHECK(d.dim() == 1);

  int N = a.size(0);

  TORCH_CHECK(b.size(0) == N);
  TORCH_CHECK(c.size(0) == N);
  TORCH_CHECK(d.size(0) == N);

  at::Tensor output = a.new_empty({N * 4});

  half* p_a = (half*)a.data_ptr();
  half* p_b = (half*)b.data_ptr();
  half* p_c = (half*)c.data_ptr();
  half* p_d = (half*)d.data_ptr();
  half* p_out = (half*)output.data_ptr();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  iluvatar::inferrt::transformer::IxinferStackLauncher(p_a, p_b, p_c, p_d,
                                                       p_out, N, stream);
  return {output};
}

std::vector<at::Tensor> one_test_opt_2(at::Tensor a, at::Tensor b) {
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.is_contiguous());

  TORCH_CHECK(b.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.is_contiguous());

  TORCH_CHECK(a.dim() == 1);
  TORCH_CHECK(b.dim() == 1);

  int N = a.size(0);

  TORCH_CHECK(b.size(0) == N);

  at::Tensor output = a.new_empty({N * 2});

  half* p_a = (half*)a.data_ptr();
  half* p_b = (half*)b.data_ptr();
  half* p_out = (half*)output.data_ptr();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  iluvatar::inferrt::transformer::IxinferStackLauncher(p_a, p_b, p_out, N,
                                                       stream);
  return {output};
}
