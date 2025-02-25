#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <stdexcept>

namespace iluvatar::inferrt::transformer {

__global__ void Cat(half* a, half* b, half* output, int m1, int m2, int k) {
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  // a
  if (blockIdx.x < m1) {
    half2* h2_a = reinterpret_cast<half2*>(a + blockIdx.x * k);
    half2* h2_out_a = reinterpret_cast<half2*>(output + blockIdx.x * k);
    if (i < k / 2) {
      h2_out_a[i] = h2_a[i];
    }
  }
  // b
  if (blockIdx.x < m2) {
    half2* h2_b = reinterpret_cast<half2*>(b + blockIdx.x * k);
    half2* h2_out_b =
        reinterpret_cast<half2*>(output + blockIdx.x * k + m1 * k);
    if (i < k / 2) {
      h2_out_b[i] = h2_b[i];
    }
  }
}

void IxinferCatLauncher(half* a, half* b, half* output, int m1, int m2, int k,
                        cudaStream_t stream) {
  if (k % 2 != 0) {
    throw std::runtime_error("IxinferStackLauncher: size error!");
  }
  int m = std::max(m1, m2);
  int num_threads = 1024;
  int half_k = k / 2;
  int num_roll = (half_k - 1 + num_threads) / num_threads;
  dim3 grid(m, num_roll);
  dim3 block(num_threads);
  Cat<<<grid, block, 0, stream>>>(a, b, output, m1, m2, k);
}

}  // namespace iluvatar::inferrt::transformer

std::vector<at::Tensor> one_test_opt_2(at::Tensor a, at::Tensor b) {
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.is_contiguous());

  TORCH_CHECK(b.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.is_contiguous());

  TORCH_CHECK(a.dim() == 2);
  TORCH_CHECK(b.dim() == 2);

  int m1 = a.size(0);
  int m2 = b.size(0);

  int k = a.size(1);

  TORCH_CHECK(b.size(1) == k);

  at::Tensor output = a.new_empty({(m1 + m2), k});

  half* p_a = (half*)a.data_ptr();
  half* p_b = (half*)b.data_ptr();
  half* p_out = (half*)output.data_ptr();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  iluvatar::inferrt::transformer::IxinferCatLauncher(p_a, p_b, p_out, m1, m2, k,
                                                     stream);
  return {output};
}
