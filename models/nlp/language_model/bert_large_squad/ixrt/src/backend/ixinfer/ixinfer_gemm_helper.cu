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
#include "ixinfer_gemm_helper.h"

namespace nvinfer1::ixrt_plugin {
namespace backend {

void cuinfer_i8_gemm(const int8_t *input_a, const int8_t *input_b, int8_t *output_c, int batch_count, int m, int n,
                     int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                     cuinferHandle_t cuinfer_handle, cudaStream_t stream) {
    /* TN: input_a: m,k input_b: n,k  output_c: n,m */
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_T;
    cuinferOperation_t transb = CUINFER_OP_N;

    cudaDataType_t Atype = CUDA_R_8I;
    cudaDataType_t Btype = CUDA_R_8I;
    cudaDataType_t Ctype = CUDA_R_8I;
    cudaDataType_t computeType = CUDA_R_32I;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;

    int lda = k;
    int ldb = k;
    int ldc = m;

    float beta = 0.f;

    cuinferStatus_t status =
        cuinferCustomGemm(cuinfer_handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &alpha, input_a, Atype,
                          lda, stridea, input_b, Btype, ldb, strideb, &beta, output_c, Ctype, ldc, stridec, batch_count,
                          computeType, scaleType, nullptr, nullptr, customOption);

    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error!, error type: " + std::to_string((int)status) + " !");
    }
}

void cuinfer_i8_gemm(const int8_t *input_a, const int8_t *input_b, const float *bias, int8_t *output_c, int batch_count,
                     int m, int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                     const float beta, const int act_type, cuinferHandle_t &cuinfer_handle, cudaStream_t &stream) {
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_T;
    cuinferOperation_t transb = CUINFER_OP_N;
    cudaDataType_t Atype = CUDA_R_8I;
    cudaDataType_t Btype = CUDA_R_8I;
    cudaDataType_t Ctype = CUDA_R_8I;
    cudaDataType_t computeType = CUDA_R_32I;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption;
    if (bias != nullptr) {
        if (act_type == 3) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_GELU;
        } else if (act_type == 4) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_RELU;
        } else {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS;
        }
    } else {
        customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;
    }

    int lda = k;
    int ldb = k;
    int ldc = m;

    cuinferStatus_t status =
        cuinferCustomGemm(cuinfer_handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &alpha, input_a, Atype,
                          lda, stridea, input_b, Btype, ldb, strideb, &beta, output_c, Ctype, ldc, stridec, batch_count,
                          computeType, scaleType, nullptr, (void *)bias, customOption);
    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error, error type: " + std::to_string((int)status) + " !");
    }
}

void cuinfer_nn_i8_gemm(const int8_t *input_a, const int8_t *input_b, int8_t *output_c, int batch_count, int m, int n,
                        int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                        cuinferHandle_t cuinfer_handle, cudaStream_t stream) {
    /* TN: input_a: k,m input_b: n,k  output_c: n,m */
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_N;
    cuinferOperation_t transb = CUINFER_OP_N;

    cudaDataType_t Atype = CUDA_R_8I;
    cudaDataType_t Btype = CUDA_R_8I;
    cudaDataType_t Ctype = CUDA_R_8I;
    cudaDataType_t computeType = CUDA_R_32I;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;

    int lda = m;
    int ldb = k;
    int ldc = m;

    float beta = 0.f;

    cuinferStatus_t status =
        cuinferCustomGemm(cuinfer_handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &alpha, input_a, Atype,
                          lda, stridea, input_b, Btype, ldb, strideb, &beta, output_c, Ctype, ldc, stridec, batch_count,
                          computeType, scaleType, nullptr, nullptr, customOption);

    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error!");
    }
}

void cuinfer_nt_i8_gemm(const int8_t *input_a, const int8_t *input_b, int8_t *output_c, int batch_count, int m, int n,
                        int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                        cuinferHandle_t cuinfer_handle, cudaStream_t stream) {
    /* TN: input_a: k,m input_b: k,n  output_c: n,m */
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_N;
    cuinferOperation_t transb = CUINFER_OP_T;

    cudaDataType_t Atype = CUDA_R_8I;
    cudaDataType_t Btype = CUDA_R_8I;
    cudaDataType_t Ctype = CUDA_R_8I;
    cudaDataType_t computeType = CUDA_R_32I;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;

    int lda = m;
    int ldb = n;
    int ldc = m;

    float beta = 0.f;

    cuinferStatus_t status =
        cuinferCustomGemm(cuinfer_handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &alpha, input_a, Atype,
                          lda, stridea, input_b, Btype, ldb, strideb, &beta, output_c, Ctype, ldc, stridec, batch_count,
                          computeType, scaleType, nullptr, nullptr, customOption);

    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error!");
    }
}

void cuinfer_tt_i8_gemm(const int8_t *input_a, const int8_t *input_b, int8_t *output_c, int batch_count, int m, int n,
                        int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                        cuinferHandle_t cuinfer_handle, cudaStream_t stream) {
    /* TN: input_a: k,m input_b: k,n  output_c: n,m */
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_T;
    cuinferOperation_t transb = CUINFER_OP_T;

    cudaDataType_t Atype = CUDA_R_8I;
    cudaDataType_t Btype = CUDA_R_8I;
    cudaDataType_t Ctype = CUDA_R_8I;
    cudaDataType_t computeType = CUDA_R_32I;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;

    int lda = k;
    int ldb = n;
    int ldc = m;

    float beta = 0.f;

    cuinferStatus_t status =
        cuinferCustomGemm(cuinfer_handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &alpha, input_a, Atype,
                          lda, stridea, input_b, Btype, ldb, strideb, &beta, output_c, Ctype, ldc, stridec, batch_count,
                          computeType, scaleType, nullptr, nullptr, customOption);

    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error!");
    }
}

void cuinfer_gemm(const half *input_a, const half *input_b, half *output_c, int batch_count, int m, int n, int k,
                  int64_t stridea, int64_t strideb, int64_t stridec, const float alpha, cublasHandle_t handle,
                  cudaStream_t stream) {
    /* Performs operation using cublas */
    float beta = 0.0f;
    cublasSetStream(handle, stream);
    cublasStatus_t status;
    if (batch_count <= 1) {
        status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, input_a, CUDA_R_16F, k, input_b,
                              CUDA_R_16F, k, &beta, output_c, CUDA_R_16F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, input_a, CUDA_R_16F, k,
                                            stridea, input_b, CUDA_R_16F, k, strideb, &beta, output_c, CUDA_R_16F, m,
                                            stridec, batch_count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuinfer_gemm error!");
    }
}

void cuinfer_nn_gemm(const half *input_a, const half *input_b, half *output_c, int batch_count, int m, int n, int k,
                     int64_t stridea, int64_t strideb, int64_t stridec, const float alpha, cublasHandle_t handle,
                     cudaStream_t stream) {
    /* Performs operation using cublas */
    float beta = 0.0f;
    cublasSetStream(handle, stream);
    cublasStatus_t status;
    if (batch_count <= 1) {
        // k,m n,k
        status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, input_a, CUDA_R_16F, m, input_b,
                              CUDA_R_16F, k, &beta, output_c, CUDA_R_16F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, input_a, CUDA_R_16F, m,
                                            stridea, input_b, CUDA_R_16F, k, strideb, &beta, output_c, CUDA_R_16F, m,
                                            stridec, batch_count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuinfer_gemm error!");
    }
}

void cuinfer_gemm(const half *input_a, const half *input_b, const half *bias, half *output_c, int batch_count, int m,
                  int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                  const int act_type, cudaStream_t &stream, cuinferHandle_t &cuinfer_handle) {
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_T;
    cuinferOperation_t transb = CUINFER_OP_N;
    cudaDataType_t Atype = CUDA_R_16F;
    cudaDataType_t Btype = CUDA_R_16F;
    cudaDataType_t Ctype = CUDA_R_16F;
    cudaDataType_t computeType = CUDA_R_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption;
    if (bias != nullptr) {
        if (act_type == 3) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_GELU;
        } else if (act_type == 4) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_RELU;
        } else {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS;
        }
    } else {
        customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;
        // std::cout << "CUINFER_BLAS_GEMM_CUSTOM_NONE" << std::endl;
    }

    int lda = k;
    int ldb = k;
    int ldc = m;
    float beta = 0.f;

    cuinferStatus_t status =
        cuinferCustomGemm(cuinfer_handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &alpha, input_a, Atype,
                          lda, stridea, input_b, Btype, ldb, strideb, &beta, output_c, Ctype, ldc, stridec, batch_count,
                          computeType, scaleType, nullptr, (void *)bias, customOption);
    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error, error type: " + std::to_string((int)status) + " !");
    }
}
void cuinfer_gemm(const half *input_a, const half *input_b, const half *bias, half *output_c, int batch_count, int m,
                  int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha, const float beta,
                  const int act_type, cudaStream_t &stream, cuinferHandle_t &cuinfer_handle) {
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_T;
    cuinferOperation_t transb = CUINFER_OP_N;
    cudaDataType_t Atype = CUDA_R_16F;
    cudaDataType_t Btype = CUDA_R_16F;
    cudaDataType_t Ctype = CUDA_R_16F;
    cudaDataType_t computeType = CUDA_R_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption;
    if (bias != nullptr) {
        if (act_type == 3) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_GELU;
        } else if (act_type == 4) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_RELU;
        } else {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS;
        }
    } else {
        customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;
        // std::cout << "CUINFER_BLAS_GEMM_CUSTOM_NONE" << std::endl;
    }

    int lda = k;
    int ldb = k;
    int ldc = m;
    // float beta = 0.f;

    cuinferStatus_t status =
        cuinferCustomGemm(cuinfer_handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &alpha, input_a, Atype,
                          lda, stridea, input_b, Btype, ldb, strideb, &beta, output_c, Ctype, ldc, stridec, batch_count,
                          computeType, scaleType, nullptr, (void *)bias, customOption);
    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error, error type: " + std::to_string((int)status) + " !");
    }
}
void cuinfer_nn_gemm(const half *input_a, const half *input_b, const half *bias, half *output_c, int batch_count, int m,
                     int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                     const int act_type, cudaStream_t &stream, cuinferHandle_t &cuinfer_handle) {
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_N;
    cuinferOperation_t transb = CUINFER_OP_N;
    cudaDataType_t Atype = CUDA_R_16F;
    cudaDataType_t Btype = CUDA_R_16F;
    cudaDataType_t Ctype = CUDA_R_16F;
    cudaDataType_t computeType = CUDA_R_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption;
    if (bias != nullptr) {
        if (act_type == 3) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_GELU;

        } else if (act_type == 4) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_RELU;
        } else {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS;
        }
    } else {
        customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;
    }

    int lda = m;
    int ldb = k;
    int ldc = m;
    float beta = 0.f;

    cuinferStatus_t status =
        cuinferCustomGemm(cuinfer_handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &alpha, input_a, Atype,
                          lda, stridea, input_b, Btype, ldb, strideb, &beta, output_c, Ctype, ldc, stridec, batch_count,
                          computeType, scaleType, nullptr, (void *)bias, customOption);
    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error, error type: " + std::to_string((int)status) + " !");
    }
}
void cuinfer_nt_gemm(const half *input_a, const half *input_b, const half *bias, half *output_c, int batch_count, int m,
                     int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                     const int act_type, cudaStream_t &stream, cuinferHandle_t &cuinfer_handle) {
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_N;
    cuinferOperation_t transb = CUINFER_OP_T;
    cudaDataType_t Atype = CUDA_R_16F;
    cudaDataType_t Btype = CUDA_R_16F;
    cudaDataType_t Ctype = CUDA_R_16F;
    cudaDataType_t computeType = CUDA_R_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption;
    if (bias != nullptr) {
        if (act_type == 3) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_GELU;

        } else if (act_type == 4) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_RELU;
        } else {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS;
        }
    } else {
        customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;
    }

    int lda = m;
    int ldb = n;
    int ldc = m;
    float beta = 0.f;

    cuinferStatus_t status =
        cuinferCustomGemm(cuinfer_handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &alpha, input_a, Atype,
                          lda, stridea, input_b, Btype, ldb, strideb, &beta, output_c, Ctype, ldc, stridec, batch_count,
                          computeType, scaleType, nullptr, (void *)bias, customOption);
    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error, error type: " + std::to_string((int)status) + " !");
    }
}

void cuinfer_tt_gemm(const half *input_a, const half *input_b, const half *bias, half *output_c, int batch_count, int m,
                     int n, int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                     const int act_type, cudaStream_t &stream, cuinferHandle_t &cuinfer_handle) {
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_T;
    cuinferOperation_t transb = CUINFER_OP_T;
    cudaDataType_t Atype = CUDA_R_16F;
    cudaDataType_t Btype = CUDA_R_16F;
    cudaDataType_t Ctype = CUDA_R_16F;
    cudaDataType_t computeType = CUDA_R_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption;
    if (bias != nullptr) {
        if (act_type == 3) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_GELU;

        } else if (act_type == 4) {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_RELU;
        } else {
            customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS;
        }
    } else {
        customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;
    }

    int lda = k;
    int ldb = n;
    int ldc = m;
    float beta = 0.f;

    cuinferStatus_t status =
        cuinferCustomGemm(cuinfer_handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &alpha, input_a, Atype,
                          lda, stridea, input_b, Btype, ldb, strideb, &beta, output_c, Ctype, ldc, stridec, batch_count,
                          computeType, scaleType, nullptr, (void *)bias, customOption);
    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error, error type: " + std::to_string((int)status) + " !");
    }
}

}  // namespace backend
}  // namespace nvinfer1::ixrt_plugin
