#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <tuple>
#include <string>
#include <algorithm>
#include <type_traits>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "tropicalgemm.h"

__device__ __forceinline__ float tropical_add(float a, float b) {
    return fmaxf(a, b);
}

__device__ __forceinline__ double tropical_add(double a, double b) {
    return fmax(a, b);
}

__device__ __forceinline__ __half tropical_add(__half a, __half b) {
    return __hmax(a, b);
}

template<typename T>
__device__ __forceinline__ T tropical_multiply(T a, T b) {
    return a + b;
}

template<typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K>
__device__ __forceinline__ void load_shared_memory(
    T* shared_A, T* shared_B, 
    const T* global_A, const T* global_B,
    int m, int n, int k, int lda, int ldb,
    int block_row, int block_col, int block_k_idx,
    int tid_x, int tid_y, int block_dim_x, int block_dim_y,
    bool transA, bool transB) {
    
    const int tid = tid_y * block_dim_x + tid_x;
    const int total_threads = block_dim_x * block_dim_y;
    
    constexpr int PADDING = 1;
    const int SHARED_A_STRIDE = BLOCK_SIZE_M + PADDING;
    const int SHARED_B_STRIDE = BLOCK_SIZE_N + PADDING;
    
    // shared_A 加载 - 修正版本
    #pragma unroll
    for (int i = 0; i < (BLOCK_SIZE_M * BLOCK_SIZE_K + total_threads - 1) / total_threads; ++i) {
        int linear_idx = tid + i * total_threads;
        if (linear_idx < BLOCK_SIZE_M * BLOCK_SIZE_K) {
            int k_idx = linear_idx / BLOCK_SIZE_M;
            int m_idx = linear_idx % BLOCK_SIZE_M;
            
            int global_row = block_row * BLOCK_SIZE_M + m_idx;
            int global_col = block_k_idx * BLOCK_SIZE_K + k_idx;
            
            T value = (T)(-INFINITY);
            if (!transA) {
                if (global_row < m && global_col < k) {
                    value = global_A[global_row * lda + global_col];
                }
            } else {
                if (global_row < m && global_col < k) {
                    value = global_A[global_col * lda + global_row];
                }
            }
            
            shared_A[k_idx * SHARED_A_STRIDE + m_idx] = value;
        }
    }
    
    // shared_B 加载 - 修正版本
    #pragma unroll
    for (int i = 0; i < (BLOCK_SIZE_K * BLOCK_SIZE_N + total_threads - 1) / total_threads; ++i) {
        int linear_idx = tid + i * total_threads;
        if (linear_idx < BLOCK_SIZE_K * BLOCK_SIZE_N) {
            int n_idx = linear_idx % BLOCK_SIZE_N;
            int k_idx = linear_idx / BLOCK_SIZE_N;
            
            int global_row = block_k_idx * BLOCK_SIZE_K + k_idx;
            int global_col = block_col * BLOCK_SIZE_N + n_idx;
            
            T value = (T)(-INFINITY);
            if (!transB) {
                if (global_row < k && global_col < n) {
                    value = global_B[global_row * ldb + global_col];
                }
            } else {
                if (global_row < k && global_col < n) {
                    value = global_B[global_col * ldb + global_row];
                }
            }
            
            shared_B[k_idx * SHARED_B_STRIDE + n_idx] = value;
        }
    }
    
    __syncthreads();
}

// =====================================================
// Kernel 1: kernel_nn (A不转置, B不转置)
// =====================================================
template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void kernel_nn(int m, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc) {

    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;

    constexpr int PADDING = 1;
    constexpr int SHARED_A_STRIDE = BLOCK_SIZE_M + PADDING;
    constexpr int SHARED_B_STRIDE = BLOCK_SIZE_N + PADDING;
    
    __shared__ T shared_A[BLOCK_SIZE_K * SHARED_A_STRIDE];
    __shared__ T shared_B[BLOCK_SIZE_K * SHARED_B_STRIDE];
    
    const int c_row_start = block_x * BLOCK_SIZE_M;
    const int c_col_start = block_y * BLOCK_SIZE_N;
    const int thread_row_start = tid_x * THREAD_SIZE_M;
    const int thread_col_start = tid_y * THREAD_SIZE_N;
    
    T accumulator[THREAD_SIZE_M * THREAD_SIZE_N];
    
    // 初始化累积器
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; ++j) {
            accumulator[i * THREAD_SIZE_N + j] = static_cast<T>(-INFINITY);
        }
    }

    // K维度循环
    for (int block_k = 0; block_k < (k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++block_k) {
        
        load_shared_memory<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(
            shared_A, shared_B, A, B,
            m, n, k, lda, ldb,
            block_x, block_y, block_k,
            tid_x, tid_y, blockDim.x, blockDim.y,
            false, false
        );
        
        #pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE_N; ++j) {
                    T a_val = shared_A[k_idx * SHARED_A_STRIDE + (thread_row_start + i)];  // 列主序访问
                    T b_val = shared_B[k_idx * SHARED_B_STRIDE + (thread_col_start + j)];  // 行主序访问
                    
                    T product = tropical_multiply(a_val, b_val);
                    accumulator[i * THREAD_SIZE_N + j] = tropical_add(accumulator[i * THREAD_SIZE_N + j], product);
                }
            }
        }
        
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; ++j) {

            int global_row = c_row_start + thread_row_start + i;
            int global_col = c_col_start + thread_col_start + j;
            int c_offset = global_row * ldc + global_col;
            
            if (global_row < m && global_col < n && c_offset < m * n) {
                T result = tropical_multiply(alpha, accumulator[i * THREAD_SIZE_N + j]);
                if (beta != static_cast<T>(-INFINITY)) {
                    T old_val = tropical_multiply(beta, C[c_offset]);
                    result = tropical_add(result, old_val);
                }
                C[c_offset] = result;
            }
        }
    }
}

// =====================================================
// Kernel 2: kernel_nt (A不转置, B转置)
// =====================================================
template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void kernel_nt(int m, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc) {
    
    constexpr int PADDING = 1;
    constexpr int SHARED_A_STRIDE = BLOCK_SIZE_M + PADDING;  // 统一为列主序
    constexpr int SHARED_B_STRIDE = BLOCK_SIZE_N + PADDING;  // 统一为行主序
    
    __shared__ T shared_A[BLOCK_SIZE_K * SHARED_A_STRIDE];   // 统一布局
    __shared__ T shared_B[BLOCK_SIZE_K * SHARED_B_STRIDE];   // 统一布局
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    
    const int c_row_start = block_x * BLOCK_SIZE_M;
    const int c_col_start = block_y * BLOCK_SIZE_N;
    const int thread_row_start = tid_x * THREAD_SIZE_M;
    const int thread_col_start = tid_y * THREAD_SIZE_N;
    
    T accumulator[THREAD_SIZE_M * THREAD_SIZE_N];
    
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; ++j) {
            accumulator[i * THREAD_SIZE_N + j] = static_cast<T>(-INFINITY);
        }
    }
    
    for (int block_k = 0; block_k < (k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++block_k) {
        
        load_shared_memory<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(
            shared_A, shared_B, A, B,
            m, n, k, lda, ldb,
            block_x, block_y, block_k,
            tid_x, tid_y, blockDim.x, blockDim.y,
            false, true
        );
        
        // 统一的访问模式
        #pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE_N; ++j) {
                    T a_val = shared_A[k_idx * SHARED_A_STRIDE + (thread_row_start + i)];  // 统一列主序访问
                    T b_val = shared_B[k_idx * SHARED_B_STRIDE + (thread_col_start + j)];  // 统一行主序访问
                    
                    T product = tropical_multiply(a_val, b_val);
                    accumulator[i * THREAD_SIZE_N + j] = tropical_add(accumulator[i * THREAD_SIZE_N + j], product);
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回结果
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; ++j) {
            int global_row = c_row_start + thread_row_start + i;
            int global_col = c_col_start + thread_col_start + j;
            
            if (global_row < m && global_col < n) {
                T result = tropical_multiply(alpha, accumulator[i * THREAD_SIZE_N + j]);
                if (beta != static_cast<T>(-INFINITY)) {
                    T old_val = tropical_multiply(beta, C[global_row * ldc + global_col]);
                    result = tropical_add(result, old_val);
                }
                C[global_row * ldc + global_col] = result;
            }
        }
    }
}

// =====================================================
// Kernel 3: kernel_tn (A转置, B不转置)
// =====================================================
template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void kernel_tn(int m, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc) {
    
    constexpr int PADDING = 1;
    constexpr int SHARED_A_STRIDE = BLOCK_SIZE_M + PADDING;
    constexpr int SHARED_B_STRIDE = BLOCK_SIZE_N + PADDING;
    
    __shared__ T shared_A[BLOCK_SIZE_K * SHARED_A_STRIDE];
    __shared__ T shared_B[BLOCK_SIZE_K * SHARED_B_STRIDE];
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    
    const int c_row_start = block_x * BLOCK_SIZE_M;
    const int c_col_start = block_y * BLOCK_SIZE_N;
    const int thread_row_start = tid_x * THREAD_SIZE_M;
    const int thread_col_start = tid_y * THREAD_SIZE_N;
    
    T accumulator[THREAD_SIZE_M * THREAD_SIZE_N];
    
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; ++j) {
            accumulator[i * THREAD_SIZE_N + j] = static_cast<T>(-INFINITY);
        }
    }
    
    for (int block_k = 0; block_k < (k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++block_k) {
        
        load_shared_memory<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(
            shared_A, shared_B, A, B,
            m, n, k, lda, ldb,
            block_x, block_y, block_k,
            tid_x, tid_y, blockDim.x, blockDim.y,
            true, false
        );
        
        // 统一的访问模式
        #pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE_N; ++j) {
                    T a_val = shared_A[k_idx * SHARED_A_STRIDE + (thread_row_start + i)];  // 统一列主序访问
                    T b_val = shared_B[k_idx * SHARED_B_STRIDE + (thread_col_start + j)];  // 统一行主序访问
                    
                    T product = tropical_multiply(a_val, b_val);
                    accumulator[i * THREAD_SIZE_N + j] = tropical_add(accumulator[i * THREAD_SIZE_N + j], product);
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回结果
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; ++j) {
            int global_row = c_row_start + thread_row_start + i;
            int global_col = c_col_start + thread_col_start + j;
            
            if (global_row < m && global_col < n) {
                T result = tropical_multiply(alpha, accumulator[i * THREAD_SIZE_N + j]);
                if (beta != static_cast<T>(-INFINITY)) {
                    T old_val = tropical_multiply(beta, C[global_row * ldc + global_col]);
                    result = tropical_add(result, old_val);
                }
                C[global_row * ldc + global_col] = result;
            }
        }
    }
}

// =====================================================
// Kernel 4: kernel_tt (A转置, B转置)
// =====================================================
template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void kernel_tt(int m, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc) {
    
    constexpr int PADDING = 1;
    constexpr int SHARED_A_STRIDE = BLOCK_SIZE_M + PADDING;  // 统一为列主序
    constexpr int SHARED_B_STRIDE = BLOCK_SIZE_N + PADDING;  // 统一为行主序

    __shared__ T shared_A[BLOCK_SIZE_K * SHARED_A_STRIDE];   // 统一布局
    __shared__ T shared_B[BLOCK_SIZE_K * SHARED_B_STRIDE];   // 统一布局

    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    
    const int c_row_start = block_x * BLOCK_SIZE_M;
    const int c_col_start = block_y * BLOCK_SIZE_N;
    const int thread_row_start = tid_x * THREAD_SIZE_M;
    const int thread_col_start = tid_y * THREAD_SIZE_N;
    
    T accumulator[THREAD_SIZE_M * THREAD_SIZE_N];
    
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; ++j) {
            accumulator[i * THREAD_SIZE_N + j] = static_cast<T>(-INFINITY);
        }
    }

    for (int block_k = 0; block_k < (k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++block_k) {
        
        load_shared_memory<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(
            shared_A, shared_B, A, B,
            m, n, k, lda, ldb,
            block_x, block_y, block_k,
            tid_x, tid_y, blockDim.x, blockDim.y,
            true, true
        );
        
        // 统一的访问模式
        #pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE_N; ++j) {
                    T a_val = shared_A[k_idx * SHARED_A_STRIDE + (thread_row_start + i)];  // 统一列主序访问
                    T b_val = shared_B[k_idx * SHARED_B_STRIDE + (thread_col_start + j)];  // 统一行主序访问
                    
                    T product = tropical_multiply(a_val, b_val);
                    accumulator[i * THREAD_SIZE_N + j] = tropical_add(accumulator[i * THREAD_SIZE_N + j], product);
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回结果
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; ++j) {
            int global_row = c_row_start + thread_row_start + i;
            int global_col = c_col_start + thread_col_start + j;
            
            if (global_row < m && global_col < n) {
                T result = tropical_multiply(alpha, accumulator[i * THREAD_SIZE_N + j]);
                if (beta != static_cast<T>(-INFINITY)) {
                    T old_val = tropical_multiply(beta, C[global_row * ldc + global_col]);
                    result = tropical_add(result, old_val);
                }
                C[global_row * ldc + global_col] = result;
            }
        }
    }
}

// tropical max-sum gemm template
template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
cublasStatus_t cutmsgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc){

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

cublasStatus_t cutmsDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, const double beta, double *C, int ldc){
    return cutmsgemm<double, 32, 16, 32, 4, 4>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cutmsSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc){
    return cutmsgemm<float, 32, 16, 32, 4, 4>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// cublasStatus_t cutmsHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half alpha, const __half *A, int lda, const __half *B, int ldb, const __half beta, __half *C, int ldc){
//     return cutmsgemm<__half, 64, 64, 64, 4, 4>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
// }

#ifdef __cplusplus
}
#endif
