#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/extension.h>

#include <cassert>
#include <iostream>
#include <vector>

std::vector<at::Tensor> one_test_opt(at::Tensor input, at::Tensor ln_weight,
                                     at::Tensor ln_bias);

std::vector<at::Tensor> test_opt(at::Tensor input, at::Tensor ln_weight,
                                 at::Tensor ln_bias) {
  return one_test_opt(input, ln_weight, ln_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_opt", &test_opt, "fast depthwise conv1d forward");
}
