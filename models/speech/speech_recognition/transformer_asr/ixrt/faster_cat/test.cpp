#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/extension.h>

#include <cassert>
#include <iostream>
#include <vector>


std::vector<at::Tensor> one_test_opt_2(at::Tensor a, at::Tensor b);

std::vector<at::Tensor> test_opt_2(at::Tensor a, at::Tensor b) {
  return one_test_opt_2(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_opt_2", &test_opt_2, "");
}
