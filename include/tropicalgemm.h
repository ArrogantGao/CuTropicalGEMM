#ifndef TROPICALGEMM_H
#define TROPICALGEMM_H

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t cutmsDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);

cublasStatus_t cutmsSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);

// cublasStatus_t cutmsHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc);

#ifdef __cplusplus
}
#endif

#endif // TROPICALGEMM_H