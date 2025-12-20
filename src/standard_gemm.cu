#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <type_traits>

#include "tropicalgemm.h"

namespace {

template <typename T>
__device__ inline T load_elem_device(const T *mat, bool trans, int row, int col, int ld) {
    return mat[(trans ? col : row) + (trans ? row : col) * ld];
}

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__device__ inline void load_shared_tiles(T *sharedA, T *sharedB,
                                         const T *A, const T *B,
                                         int m, int n, int k,
                                         int lda, int ldb,
                                         int block_row, int block_col, int block_k_idx,
                                         int tid_x, int tid_y, int block_dim_x, int block_dim_y,
                                         bool transA, bool transB) {
    constexpr int PADDING = 1;
    const int strideA = BLOCK_M + PADDING;
    const int strideB = BLOCK_N + PADDING;

    const int tid = tid_y * block_dim_x + tid_x;
    const int total_threads = block_dim_x * block_dim_y;

    // load A tile
    #pragma unroll
    for (int i = 0; i < (BLOCK_M * BLOCK_K + total_threads - 1) / total_threads; ++i) {
        const int linear_idx = tid + i * total_threads;
        if (linear_idx < BLOCK_M * BLOCK_K) {
            const int k_idx = linear_idx / BLOCK_M;
            const int m_idx = linear_idx % BLOCK_M;

            const int global_row = block_row * BLOCK_M + m_idx;
            const int global_col = block_k_idx * BLOCK_K + k_idx;

            const bool within = (global_row < m) && (global_col < k);
            const T value = within
                ? load_elem_device(A, transA, global_row, global_col, lda)
                : static_cast<T>(0);

            sharedA[k_idx * strideA + m_idx] = value;
        }
    }

    // load B tile
    #pragma unroll
    for (int i = 0; i < (BLOCK_K * BLOCK_N + total_threads - 1) / total_threads; ++i) {
        const int linear_idx = tid + i * total_threads;
        if (linear_idx < BLOCK_K * BLOCK_N) {
            const int n_idx = linear_idx % BLOCK_N;
            const int k_idx = linear_idx / BLOCK_N;

            const int global_row = block_k_idx * BLOCK_K + k_idx;
            const int global_col = block_col * BLOCK_N + n_idx;

            const bool within = (global_row < k) && (global_col < n);
            const T value = within
                ? load_elem_device(B, transB, global_row, global_col, ldb)
                : static_cast<T>(0);

            sharedB[k_idx * strideB + n_idx] = value;
        }
    }

    __syncthreads();
}

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void classical_gemm_kernel(int m, int n, int k, T alpha,
                                      const T *A, int lda,
                                      const T *B, int ldb,
                                      T beta, T *C, int ldc,
                                      bool transA, bool transB) {
    static_assert(BLOCK_M % THREAD_M == 0, "BLOCK_M must be divisible by THREAD_M");
    static_assert(BLOCK_N % THREAD_N == 0, "BLOCK_N must be divisible by THREAD_N");

    constexpr int PADDING = 1;
    constexpr int strideA = BLOCK_M + PADDING;
    constexpr int strideB = BLOCK_N + PADDING;

    __shared__ T sharedA[BLOCK_K * strideA];
    __shared__ T sharedB[BLOCK_K * strideB];

    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;

    const int c_row_start = block_x * BLOCK_M;
    const int c_col_start = block_y * BLOCK_N;
    const int thread_row_start = tid_x * THREAD_M;
    const int thread_col_start = tid_y * THREAD_N;

    T accum[THREAD_M * THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M * THREAD_N; ++i) {
        accum[i] = static_cast<T>(0);
    }

    const int block_k_count = (k + BLOCK_K - 1) / BLOCK_K;
    for (int block_k = 0; block_k < block_k_count; ++block_k) {
        load_shared_tiles<T, BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N>(
            sharedA, sharedB, A, B, m, n, k, lda, ldb,
            block_x, block_y, block_k,
            tid_x, tid_y, blockDim.x, blockDim.y,
            transA, transB);

        #pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_K; ++k_idx) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                const T a_val = sharedA[k_idx * strideA + (thread_row_start + i)];
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    const T b_val = sharedB[k_idx * strideB + (thread_col_start + j)];
                    accum[i * THREAD_N + j] += a_val * b_val;
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        const int global_row = c_row_start + thread_row_start + i;
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            const int global_col = c_col_start + thread_col_start + j;
            if (global_row < m && global_col < n) {
                const int idx = global_col * ldc + global_row;
                const T c_val = C[idx];
                C[idx] = alpha * accum[i * THREAD_N + j] + beta * c_val;
            }
        }
    }
}

inline bool to_bool_trans(cublasOperation_t op, bool &is_trans) {
    if (op == CUBLAS_OP_N) {
        is_trans = false;
        return true;
    }
    if (op == CUBLAS_OP_T) {
        is_trans = true;
        return true;
    }
    return false;
}

} // namespace

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
static cublasStatus_t classical_gemm_dispatch(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k, T alpha,
                                              const T *A, int lda,
                                              const T *B, int ldb,
                                              T beta, T *C, int ldc) {
    bool transA = false;
    bool transB = false;

    if (!to_bool_trans(transa, transA) || !to_bool_trans(transb, transB)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    dim3 threads(BLOCK_M / THREAD_M, BLOCK_N / THREAD_N);
    dim3 grid((m + BLOCK_M - 1) / BLOCK_M, (n + BLOCK_N - 1) / BLOCK_N);

    cudaStream_t stream = nullptr;
    cublasGetStream(handle, &stream);

    classical_gemm_kernel<T, BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N><<<grid, threads, 0, stream>>>(
        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, transA, transB);
    return CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t cuClassicDgemm(cublasHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int m, int n, int k,
                                          double alpha,
                                          const double *A, int lda,
                                          const double *B, int ldb,
                                          double beta, double *C, int ldc) {
    return classical_gemm_dispatch<double, 32, 16, 32, 4, 4>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern "C" cublasStatus_t cuClassicSgemm(cublasHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int m, int n, int k,
                                          float alpha,
                                          const float *A, int lda,
                                          const float *B, int ldb,
                                          float beta, float *C, int ldc) {
    return classical_gemm_dispatch<float, 32, 16, 32, 4, 4>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

