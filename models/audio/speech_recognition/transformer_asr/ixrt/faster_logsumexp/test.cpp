#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/extension.h>

#include <cassert>
#include <iostream>
#include <vector>

std::vector<at::Tensor> one_test_opt(at::Tensor input);

std::vector<at::Tensor> test_opt(at::Tensor input) {
  return one_test_opt(input);
}

std::vector<at::Tensor> one_test_dim0(at::Tensor input);

std::vector<at::Tensor> test_opt_dim0(at::Tensor input) {
  return one_test_dim0(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_opt", &test_opt, "");
  m.def("test_opt_dim0", &test_opt_dim0, "");
}
