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

#include <cblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void tmsDgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);

void tmsSgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);

cublasStatus_t cutmsDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);

cublasStatus_t cutmsSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);

// Standard algebra GEMM (C = alpha * op(A) * op(B) + beta * C)

typedef enum {
    CLASSIC_TILE_32x16x32 = 0,
    CLASSIC_TILE_64x32x32 = 1,
} ClassicTile;

cublasStatus_t cuClassicDgemmTiled(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                   double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc,
                                   ClassicTile tile);

cublasStatus_t cuClassicSgemmTiled(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                   float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc,
                                   ClassicTile tile);

cublasStatus_t cuClassicDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);

cublasStatus_t cuClassicSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);

#ifdef __cplusplus
}
#endif

#endif // TROPICALGEMM_H