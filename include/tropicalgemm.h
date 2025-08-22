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

template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void kernel_tt(int m, int n, int k, const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void kernel_tn(int m, int n, int k, const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void kernel_nt(int m, int n, int k, const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void kernel_nn(int m, int n, int k, const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

// tropical max-sum gemm template
template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
cublasStatus_t cutmsgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc){

    constexpr int PADDING = 1;
    constexpr int shared_mem_size = ((BLOCK_SIZE_M + PADDING) * BLOCK_SIZE_K + (BLOCK_SIZE_N + PADDING) * BLOCK_SIZE_K) * sizeof(T);

    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    const dim3 threads(BLOCK_SIZE_M / THREAD_SIZE_M, BLOCK_SIZE_N / THREAD_SIZE_N);
    const dim3 grid((m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, (n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

    if (transa == CUBLAS_OP_N && transb == CUBLAS_OP_N){
        kernel_nn<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_M, THREAD_SIZE_N><<<grid, threads, shared_mem_size, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return CUBLAS_STATUS_SUCCESS;
    } else if (transa == CUBLAS_OP_N && transb == CUBLAS_OP_T){
        kernel_nt<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_M, THREAD_SIZE_N><<<grid, threads, shared_mem_size, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return CUBLAS_STATUS_SUCCESS;
    } else if (transa == CUBLAS_OP_T && transb == CUBLAS_OP_N){
        kernel_tn<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_M, THREAD_SIZE_N><<<grid, threads, shared_mem_size, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return CUBLAS_STATUS_SUCCESS;
    } else if (transa == CUBLAS_OP_T && transb == CUBLAS_OP_T){
        kernel_tt<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_M, THREAD_SIZE_N><<<grid, threads, shared_mem_size, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return CUBLAS_STATUS_SUCCESS;
    } else {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t cutmsDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);

cublasStatus_t cutmsSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);

cublasStatus_t cutmsHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc);

#ifdef __cplusplus
}
#endif

#endif // TROPICALGEMM_H