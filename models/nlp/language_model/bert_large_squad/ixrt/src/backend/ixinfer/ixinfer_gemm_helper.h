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
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <ixinfer.h>

#include <stdexcept>

namespace nvinfer1 {
namespace ixrt_plugin {
namespace backend {

void cuinfer_i8_gemm(const int8_t *input_a, const int8_t *input_b, int8_t *output_c, int batch_count, int m, int n,
                     int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                     cuinferHandle_t cuinfer_handle, cudaStream_t stream);

void cuinfer_i8_gemm(const int8_t *input_a, const int8_t *input_b, const float *bias, int8_t *output_c, int batch_count,
                     int m, int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                     const float beta, const int act_type, cuinferHandle_t &cuinfer_handle, cudaStream_t &stream);

void cuinfer_nn_i8_gemm(const int8_t *input_a, const int8_t *input_b, int8_t *output_c, int batch_count, int m, int n,
                        int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                        cuinferHandle_t cuinfer_handle, cudaStream_t stream);

void cuinfer_nt_i8_gemm(const int8_t *input_a, const int8_t *input_b, int8_t *output_c, int batch_count, int m, int n,
                        int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                        cuinferHandle_t cuinfer_handle, cudaStream_t stream);

void cuinfer_tt_i8_gemm(const int8_t *input_a, const int8_t *input_b, int8_t *output_c, int batch_count, int m, int n,
                        int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                        cuinferHandle_t cuinfer_handle, cudaStream_t stream);

void cuinfer_gemm(const half *input_a, const half *input_b, half *output_c, int batch_count, int m, int n, int k,
                  int64_t stridea, int64_t strideb, int64_t stridec, const float alpha, cublasHandle_t cublas_handle,
                  cudaStream_t stream);

void cuinfer_nn_gemm(const half *input_a, const half *input_b, half *output_c, int batch_count, int m, int n, int k,
                     int64_t stridea, int64_t strideb, int64_t stridec, const float alpha, cublasHandle_t cublas_handle,
                     cudaStream_t stream);

void cuinfer_gemm(const half *input_a, const half *input_b, const half *bias, half *output_c, int batch_count, int m,
                  int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                  const int act_type, cudaStream_t &stream, cuinferHandle_t &cuinfer_handle);
void cuinfer_gemm(const half *input_a, const half *input_b, const half *bias, half *output_c, int batch_count, int m,
                  int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha, const float beta,
                  const int act_type, cudaStream_t &stream, cuinferHandle_t &cuinfer_handle);
void cuinfer_nn_gemm(const half *input_a, const half *input_b, const half *bias, half *output_c, int batch_count, int m,
                     int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                     const int act_type, cudaStream_t &stream, cuinferHandle_t &cuinfer_handle);
void cuinfer_nt_gemm(const half *input_a, const half *input_b, const half *bias, half *output_c, int batch_count, int m,
                     int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                     const int act_type, cudaStream_t &stream, cuinferHandle_t &cuinfer_handle);
void cuinfer_tt_gemm(const half *input_a, const half *input_b, const half *bias, half *output_c, int batch_count, int m,
                     int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                     const int act_type, cudaStream_t &stream, cuinferHandle_t &cuinfer_handle);
}  // namespace bert
}  // namespace ixrt_plugin
}  // namespace nvinfer1
