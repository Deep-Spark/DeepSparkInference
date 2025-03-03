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
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "checkMacrosPlugin.h"

namespace nvinfer1 {
namespace ixrt_plugin {
namespace backend {

/* GPU function guard */

/**
 * @brief cublasLt gemm without imma
 *
 * @tparam OutType output dtype
 * @tparam ScaleType scale dtype
 * @param input_a
 * @param input_b
 * @param output_c
 * @param batch_count
 * @param m
 * @param n
 * @param k
 * @param stridea
 * @param strideb
 * @param stridec
 * @param alpha
 * @param cublasLt_handle
 * @param stream
 */
template <typename OutType, typename ScaleType>
void cublaslt_gemm(const int8_t* input_a, const int8_t* input_b, OutType* output_c, int batch_count, int m, int n,
                   int k, int64_t stridea, int64_t strideb, int64_t stridec, const ScaleType alpha,
                   cublasLtHandle_t cublasLt_handle, cudaStream_t stream) {
    cublasOperation_t transpose = CUBLAS_OP_T;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32I;
#else
    cudaDataType_t compute_type = CUDA_R_32I;
#endif
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t desc_a = NULL;
    cublasLtMatrixLayout_t desc_b = NULL;
    cublasLtMatrixLayout_t desc_c = NULL;

    cudaDataType_t out_dtype;
    cudaDataType_t scale_dtype;
    if (std::is_same<OutType, int32_t>::value) {
        out_dtype = CUDA_R_32I;
        scale_dtype = CUDA_R_32I;
    } else if (std::is_same<OutType, int8_t>::value) {
        out_dtype = CUDA_R_8I;
        scale_dtype = CUDA_R_32F;
    } else {
        throw std::runtime_error("Unsupported output type");
    }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_dtype));
#else
    CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&matmul_desc, compute_type));
    CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_dtype,
                                                   sizeof(scale_dtype)));
#endif
    CHECK_GPU_ERROR(
        cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));

    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_a, CUDA_R_8I, k, m, k));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_b, CUDA_R_8I, k, n, k));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_c, out_dtype, m, n, m));

    if (batch_count > 1) {
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
                                                         sizeof(stridea)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
                                                         sizeof(strideb)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
                                                         sizeof(stridec)));
    }

    ScaleType beta = ScaleType(0);
    CHECK_GPU_ERROR(cublasLtMatmul(cublasLt_handle, matmul_desc, &alpha, input_a, desc_a, input_b, desc_b, &beta,
                                   output_c, desc_c, output_c, desc_c, NULL, NULL, 0, stream));

    CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(matmul_desc));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_a));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_b));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_c));
}

inline void cublaslt_gemm(const half* input_a, const half* input_b, half* output_c, int batch_count, int m, int n,
                          int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                          cublasLtHandle_t cublasLt_handle, cudaStream_t stream) {
    cublasOperation_t transpose = CUBLAS_OP_T;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#else
    cudaDataType_t compute_type = CUDA_R_32F;
#endif
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t desc_a = NULL;
    cublasLtMatrixLayout_t desc_b = NULL;
    cublasLtMatrixLayout_t desc_c = NULL;

    cudaDataType_t out_dtype = CUDA_R_16F;
    cudaDataType_t scale_dtype = CUDA_R_32F;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_dtype));
#else
    CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&matmul_desc, compute_type));
    CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_dtype,
                                                   sizeof(scale_dtype)));
#endif
    CHECK_GPU_ERROR(
        cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));

    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_a, CUDA_R_16F, k, m, k));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_b, CUDA_R_16F, k, n, k));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_c, out_dtype, m, n, m));

    if (batch_count > 1) {
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
                                                         sizeof(stridea)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
                                                         sizeof(strideb)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
                                                         sizeof(stridec)));
    }

    float beta = 0.0;
    CHECK_GPU_ERROR(cublasLtMatmul(cublasLt_handle, matmul_desc, &alpha, input_a, desc_a, input_b, desc_b, &beta,
                                   output_c, desc_c, output_c, desc_c, NULL, NULL, 0, stream));

    CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(matmul_desc));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_a));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_b));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_c));
}

template void cublaslt_gemm<int32_t, int32_t>(const int8_t* input_a, const int8_t* input_b, int32_t* output_c,
                                              int batchCount, int m, int n, int k, int64_t stridea, int64_t strideb,
                                              int64_t stridec, const int32_t alpha, cublasLtHandle_t cublasLt_handle,
                                              cudaStream_t stream);

template void cublaslt_gemm<int8_t, float>(const int8_t* input_a, const int8_t* input_b, int8_t* output_c,
                                           int batchCount, int m, int n, int k, int64_t stridea, int64_t strideb,
                                           int64_t stridec, const float alpha, cublasLtHandle_t cublasLt_handle,
                                           cudaStream_t stream);

/************add by pxl *************/
template <typename OutType, typename ScaleType>
void cublaslt_gemm_nn(const int8_t* input_a, const int8_t* input_b, OutType* output_c, int batch_count, int m, int n,
                      int k, int64_t stridea, int64_t strideb, int64_t stridec, const ScaleType alpha,
                      cublasLtHandle_t cublasLt_handle, cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32I;
#else
    cudaDataType_t compute_type = CUDA_R_32I;
#endif
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t desc_a = NULL;
    cublasLtMatrixLayout_t desc_b = NULL;
    cublasLtMatrixLayout_t desc_c = NULL;

    cudaDataType_t out_dtype;
    cudaDataType_t scale_dtype;
    if (std::is_same<OutType, int32_t>::value) {
        out_dtype = CUDA_R_32I;
        scale_dtype = CUDA_R_32I;
    } else if (std::is_same<OutType, int8_t>::value) {
        out_dtype = CUDA_R_8I;
        scale_dtype = CUDA_R_32F;
    } else {
        throw std::runtime_error("Unsupported output type");
    }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_dtype));
#else
    CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&matmul_desc, compute_type));
    CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_dtype,
                                                   sizeof(scale_dtype)));
#endif

    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_a, CUDA_R_8I, m, k, m));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_b, CUDA_R_8I, k, n, k));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_c, out_dtype, m, n, m));

    if (batch_count > 1) {
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
                                                         sizeof(stridea)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
                                                         sizeof(strideb)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
                                                         sizeof(stridec)));
    }

    ScaleType beta = ScaleType(0);
    CHECK_GPU_ERROR(cublasLtMatmul(cublasLt_handle, matmul_desc, &alpha, input_a, desc_a, input_b, desc_b, &beta,
                                   output_c, desc_c, output_c, desc_c, NULL, NULL, 0, stream));

    CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(matmul_desc));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_a));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_b));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_c));
}

template void cublaslt_gemm_nn<int32_t, int32_t>(const int8_t* input_a, const int8_t* input_b, int32_t* output_c,
                                                 int batchCount, int m, int n, int k, int64_t stridea, int64_t strideb,
                                                 int64_t stridec, const int32_t alpha, cublasLtHandle_t cublasLt_handle,
                                                 cudaStream_t stream);

template void cublaslt_gemm_nn<int8_t, float>(const int8_t* input_a, const int8_t* input_b, int8_t* output_c,
                                              int batchCount, int m, int n, int k, int64_t stridea, int64_t strideb,
                                              int64_t stridec, const float alpha, cublasLtHandle_t cublasLt_handle,
                                              cudaStream_t stream);

inline void cublaslt_gemm_nn(const half* input_a, const half* input_b, half* output_c, int batch_count, int m, int n,
                          int k, int64_t stridea, int64_t strideb, int64_t stridec, const float alpha,
                          cublasLtHandle_t cublasLt_handle, cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#else
    cudaDataType_t compute_type = CUDA_R_32F;
#endif
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t desc_a = NULL;
    cublasLtMatrixLayout_t desc_b = NULL;
    cublasLtMatrixLayout_t desc_c = NULL;

    cudaDataType_t out_dtype = CUDA_R_16F;
    cudaDataType_t scale_dtype = CUDA_R_32F;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_dtype));
#else
    CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&matmul_desc, compute_type));
    CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_dtype,
                                                   sizeof(scale_dtype)));
#endif

    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_a, CUDA_R_16F, m, k, m));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_b, CUDA_R_16F, k, n, k));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_c, out_dtype, m, n, m));

    if (batch_count > 1) {
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
                                                         sizeof(stridea)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
                                                         sizeof(strideb)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                                         sizeof(batch_count)));
        CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(desc_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
                                                         sizeof(stridec)));
    }

    float beta = 0.0;
    CHECK_GPU_ERROR(cublasLtMatmul(cublasLt_handle, matmul_desc, &alpha, input_a, desc_a, input_b, desc_b, &beta,
                                   output_c, desc_c, output_c, desc_c, NULL, NULL, 0, stream));

    CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(matmul_desc));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_a));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_b));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_c));
}

}  // namespace backend
}  // namespace ixrt_plugin
}  // namespace nvinfer1
