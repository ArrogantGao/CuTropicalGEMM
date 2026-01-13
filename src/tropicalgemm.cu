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
    return max(a, b);
}

__device__ __forceinline__ double tropical_add(double a, double b) {
    return max(a, b);
}

template<typename T>
__device__ __forceinline__ T tropical_add(T a, T b) {
    return max(a, b);
}

template<typename T>
__device__ __forceinline__ T tropical_multiply(T a, T b) {
    return a + b;
}

__device__ __forceinline__ float tropical_muladd(float a, float b, float c) {
    return max(a + b, c);
}

__device__ __forceinline__ double tropical_muladd(double a, double b, double c) {
    return max(a + b, c);
}

__device__ __forceinline__ int tropical_muladd(int a, int b, int c) {
    return max(a + b, c);
}

__device__ __forceinline__ long tropical_muladd(long a, long b, long c) {
    return max(a + b, c);
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
    
    // load shared_A
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
                    value = global_A[global_row + global_col * lda];
                }
            } else {
                if (global_row < m && global_col < k) {
                    value = global_A[global_col + global_row * lda];
                }
            }
            
            shared_A[k_idx * SHARED_A_STRIDE + m_idx] = value;
        }
    }
    
    // load shared_B
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
                    value = global_B[global_row + global_col * ldb];
                }
            } else {
                if (global_row < k && global_col < n) {
                    value = global_B[global_col + global_row * ldb];
                }
            }
            
            shared_B[k_idx * SHARED_B_STRIDE + n_idx] = value;
        }
    }
    
    __syncthreads();
}

template <typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void gemm_kernel(int m, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc, bool transA, bool transB) {

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
            transA, transB
        );
        
        #pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE_N; ++j) {
                    T a_val = shared_A[k_idx * SHARED_A_STRIDE + (thread_row_start + i)];  // 列主序访问
                    T b_val = shared_B[k_idx * SHARED_B_STRIDE + (thread_col_start + j)];  // 行主序访问
                    
                    // T product = tropical_multiply(a_val, b_val);
                    // accumulator[i * THREAD_SIZE_N + j] = tropical_add(accumulator[i * THREAD_SIZE_N + j], product);
                    accumulator[i * THREAD_SIZE_N + j] = tropical_muladd(a_val, b_val, accumulator[i * THREAD_SIZE_N + j]);
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
            int c_offset = global_col * ldc + global_row;
            
            if (global_row < m && global_col < n && c_offset < m * n) {
                T result = tropical_multiply(alpha, accumulator[i * THREAD_SIZE_N + j]);
                if (beta != static_cast<T>(-INFINITY)) {
                    // T old_val = tropical_multiply(beta, C[c_offset]);
                    // result = tropical_add(result, old_val);
                    result = tropical_muladd(beta, C[c_offset], result);
                }
                C[c_offset] = result;
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

    bool transA, transB;
    if (transa == CUBLAS_OP_T) {
        transA = true;
    } else if (transa == CUBLAS_OP_N) {
        transA = false;
    } else {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (transb == CUBLAS_OP_T) {
        transB = true;
    } else if (transb == CUBLAS_OP_N) {
        transB = false;
    } else {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    gemm_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_M, THREAD_SIZE_N><<<grid, threads, shared_mem_size, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, transA, transB);

    return CUBLAS_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t cutmsDgemm_legacy(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, const double beta, double *C, int ldc){
    return cutmsgemm<double, 32, 16, 32, 4, 4>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cutmsSgemm_legacy(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc){
    return cutmsgemm<float, 32, 16, 32, 4, 4>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

#ifdef __cplusplus
}
#endif

// ============================================================================
// OPTIMIZED TROPICAL GEMM KERNEL
// Features: Vectorized loads, double buffering, optimal register blocking
// ============================================================================

namespace {

// Vectorized load helpers
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ double2 load_double2(const double* ptr) {
    return *reinterpret_cast<const double2*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ void store_double2(double* ptr, double2 val) {
    *reinterpret_cast<double2*>(ptr) = val;
}

// ============================================================================
// HIGHLY OPTIMIZED FLOAT KERNEL - v4
// Configuration: 128x128 output tile, 8x8 per thread, 256 threads
// Uses 3D shared memory indexing for better compiler optimization
// ============================================================================
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void __launch_bounds__(256, 2)
tropical_gemm_float_optimized(
    int m, int n, int k,
    float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta,
    float* __restrict__ C, int ldc,
    bool transA, bool transB)
{
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;

    const int block_m_idx = blockIdx.x;
    const int block_n_idx = blockIdx.y;

    // 3D shared memory: [buffer][k][m/n + padding]
    constexpr int PAD = 4;
    __shared__ float smem_A[2][BLOCK_K][BLOCK_M + PAD];
    __shared__ float smem_B[2][BLOCK_K][BLOCK_N + PAD];

    const int thread_row = thread_x * THREAD_M;
    const int thread_col = thread_y * THREAD_N;

    // Accumulators
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            acc[i][j] = -INFINITY;
        }
    }

    const int bm = block_m_idx * BLOCK_M;
    const int bn = block_n_idx * BLOCK_N;
    const int num_k = (k + BLOCK_K - 1) / BLOCK_K;

    // Load first tile - coalesced access pattern
    // 256 threads loading BLOCK_K * BLOCK_M = 16 * 128 = 2048 elements
    #pragma unroll
    for (int i = 0; i < (BLOCK_K * BLOCK_M + 255) / 256; ++i) {
        int idx = tid + i * 256;
        if (idx < BLOCK_K * BLOCK_M) {
            int kk = idx / BLOCK_M;
            int mm = idx % BLOCK_M;
            int gm = bm + mm;
            int gk = kk;

            float val = -INFINITY;
            if (gm < m && gk < k) {
                val = transA ? A[gk + gm * lda] : A[gm + gk * lda];
            }
            smem_A[0][kk][mm] = val;
        }
    }

    #pragma unroll
    for (int i = 0; i < (BLOCK_K * BLOCK_N + 255) / 256; ++i) {
        int idx = tid + i * 256;
        if (idx < BLOCK_K * BLOCK_N) {
            int kk = idx / BLOCK_N;
            int nn = idx % BLOCK_N;
            int gk = kk;
            int gn = bn + nn;

            float val = -INFINITY;
            if (gk < k && gn < n) {
                val = transB ? B[gn + gk * ldb] : B[gk + gn * ldb];
            }
            smem_B[0][kk][nn] = val;
        }
    }
    __syncthreads();

    // Main loop with double buffering
    for (int kt = 0; kt < num_k; ++kt) {
        int rb = kt & 1;
        int wb = 1 - rb;
        int next_k = (kt + 1) * BLOCK_K;

        // Load next tile
        if (kt + 1 < num_k) {
            #pragma unroll
            for (int i = 0; i < (BLOCK_K * BLOCK_M + 255) / 256; ++i) {
                int idx = tid + i * 256;
                if (idx < BLOCK_K * BLOCK_M) {
                    int kk = idx / BLOCK_M;
                    int mm = idx % BLOCK_M;
                    int gm = bm + mm;
                    int gk = next_k + kk;

                    float val = -INFINITY;
                    if (gm < m && gk < k) {
                        val = transA ? A[gk + gm * lda] : A[gm + gk * lda];
                    }
                    smem_A[wb][kk][mm] = val;
                }
            }

            #pragma unroll
            for (int i = 0; i < (BLOCK_K * BLOCK_N + 255) / 256; ++i) {
                int idx = tid + i * 256;
                if (idx < BLOCK_K * BLOCK_N) {
                    int kk = idx / BLOCK_N;
                    int nn = idx % BLOCK_N;
                    int gk = next_k + kk;
                    int gn = bn + nn;

                    float val = -INFINITY;
                    if (gk < k && gn < n) {
                        val = transB ? B[gn + gk * ldb] : B[gk + gn * ldb];
                    }
                    smem_B[wb][kk][nn] = val;
                }
            }
        }

        // Compute with register blocking
        float fA[THREAD_M], fB[THREAD_N];

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            // Load fragments from shared memory
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                fA[i] = smem_A[rb][kk][thread_row + i];
            }
            #pragma unroll
            for (int j = 0; j < THREAD_N; ++j) {
                fB[j] = smem_B[rb][kk][thread_col + j];
            }

            // Outer product
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    acc[i][j] = fmaxf(fA[i] + fB[j], acc[i][j]);
                }
            }
        }

        if (kt + 1 < num_k) {
            __syncthreads();
        }
    }

    // Store results
    const int cr = bm + thread_row;
    const int cc = bn + thread_col;

    #pragma unroll
    for (int j = 0; j < THREAD_N; ++j) {
        int gc = cc + j;
        if (gc < n) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                int gr = cr + i;
                if (gr < m) {
                    int idx = gc * ldc + gr;
                    float res = alpha + acc[i][j];
                    if (beta != -INFINITY) {
                        res = fmaxf(res, beta + C[idx]);
                    }
                    C[idx] = res;
                }
            }
        }
    }
}

// ============================================================================
// HIGHLY OPTIMIZED DOUBLE KERNEL - v4
// Configuration: 64x64 output tile, 8x8 per thread, 64 threads
// Uses 3D shared memory indexing
// ============================================================================
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void __launch_bounds__(64, 4)
tropical_gemm_double_optimized(
    int m, int n, int k,
    double alpha,
    const double* __restrict__ A, int lda,
    const double* __restrict__ B, int ldb,
    double beta,
    double* __restrict__ C, int ldc,
    bool transA, bool transB)
{
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;

    const int block_m_idx = blockIdx.x;
    const int block_n_idx = blockIdx.y;

    constexpr int PAD = 2;
    __shared__ double smem_A[2][BLOCK_K][BLOCK_M + PAD];
    __shared__ double smem_B[2][BLOCK_K][BLOCK_N + PAD];

    const int thread_row = thread_x * THREAD_M;
    const int thread_col = thread_y * THREAD_N;

    double acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            acc[i][j] = -INFINITY;
        }
    }

    const int bm = block_m_idx * BLOCK_M;
    const int bn = block_n_idx * BLOCK_N;
    const int num_k = (k + BLOCK_K - 1) / BLOCK_K;

    // Load first tile
    #pragma unroll
    for (int i = 0; i < (BLOCK_K * BLOCK_M + 63) / 64; ++i) {
        int idx = tid + i * 64;
        if (idx < BLOCK_K * BLOCK_M) {
            int kk = idx / BLOCK_M;
            int mm = idx % BLOCK_M;
            int gm = bm + mm;
            int gk = kk;

            double val = -INFINITY;
            if (gm < m && gk < k) {
                val = transA ? A[gk + gm * lda] : A[gm + gk * lda];
            }
            smem_A[0][kk][mm] = val;
        }
    }

    #pragma unroll
    for (int i = 0; i < (BLOCK_K * BLOCK_N + 63) / 64; ++i) {
        int idx = tid + i * 64;
        if (idx < BLOCK_K * BLOCK_N) {
            int kk = idx / BLOCK_N;
            int nn = idx % BLOCK_N;
            int gk = kk;
            int gn = bn + nn;

            double val = -INFINITY;
            if (gk < k && gn < n) {
                val = transB ? B[gn + gk * ldb] : B[gk + gn * ldb];
            }
            smem_B[0][kk][nn] = val;
        }
    }
    __syncthreads();

    // Main loop
    for (int kt = 0; kt < num_k; ++kt) {
        int rb = kt & 1;
        int wb = 1 - rb;
        int next_k = (kt + 1) * BLOCK_K;

        if (kt + 1 < num_k) {
            #pragma unroll
            for (int i = 0; i < (BLOCK_K * BLOCK_M + 63) / 64; ++i) {
                int idx = tid + i * 64;
                if (idx < BLOCK_K * BLOCK_M) {
                    int kk = idx / BLOCK_M;
                    int mm = idx % BLOCK_M;
                    int gm = bm + mm;
                    int gk = next_k + kk;

                    double val = -INFINITY;
                    if (gm < m && gk < k) {
                        val = transA ? A[gk + gm * lda] : A[gm + gk * lda];
                    }
                    smem_A[wb][kk][mm] = val;
                }
            }

            #pragma unroll
            for (int i = 0; i < (BLOCK_K * BLOCK_N + 63) / 64; ++i) {
                int idx = tid + i * 64;
                if (idx < BLOCK_K * BLOCK_N) {
                    int kk = idx / BLOCK_N;
                    int nn = idx % BLOCK_N;
                    int gk = next_k + kk;
                    int gn = bn + nn;

                    double val = -INFINITY;
                    if (gk < k && gn < n) {
                        val = transB ? B[gn + gk * ldb] : B[gk + gn * ldb];
                    }
                    smem_B[wb][kk][nn] = val;
                }
            }
        }

        double fA[THREAD_M], fB[THREAD_N];

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                fA[i] = smem_A[rb][kk][thread_row + i];
            }
            #pragma unroll
            for (int j = 0; j < THREAD_N; ++j) {
                fB[j] = smem_B[rb][kk][thread_col + j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    acc[i][j] = fmax(fA[i] + fB[j], acc[i][j]);
                }
            }
        }

        if (kt + 1 < num_k) {
            __syncthreads();
        }
    }

    // Store results
    const int cr = bm + thread_row;
    const int cc = bn + thread_col;

    #pragma unroll
    for (int j = 0; j < THREAD_N; ++j) {
        int gc = cc + j;
        if (gc < n) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                int gr = cr + i;
                if (gr < m) {
                    int idx = gc * ldc + gr;
                    double res = alpha + acc[i][j];
                    if (beta != -INFINITY) {
                        res = fmax(res, beta + C[idx]);
                    }
                    C[idx] = res;
                }
            }
        }
    }
}

} // anonymous namespace

// Dispatch functions for optimized kernels
template <typename T>
static cublasStatus_t tropical_gemm_optimized_dispatch(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    T alpha, const T* A, int lda,
    const T* B, int ldb,
    T beta, T* C, int ldc,
    TropicalTile tile);

template <>
cublasStatus_t tropical_gemm_optimized_dispatch<float>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc,
    TropicalTile tile)
{
    bool transA = (transa == CUBLAS_OP_T);
    bool transB = (transb == CUBLAS_OP_T);

    if (transa != CUBLAS_OP_N && transa != CUBLAS_OP_T) return CUBLAS_STATUS_INVALID_VALUE;
    if (transb != CUBLAS_OP_N && transb != CUBLAS_OP_T) return CUBLAS_STATUS_INVALID_VALUE;

    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    switch (tile) {
        case TROPICAL_TILE_32x16x32:
            // Legacy kernel
            return cutmsgemm<float, 32, 16, 32, 4, 4>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

        case TROPICAL_TILE_128x128x32:
        default: {
            // Optimized kernel: 128x128x16 tiles
            // 8x8 work per thread, 256 threads (16x16)
            constexpr int BLOCK_M = 128;
            constexpr int BLOCK_N = 128;
            constexpr int BLOCK_K = 16;
            constexpr int THREAD_M = 8;
            constexpr int THREAD_N = 8;

            dim3 threads(BLOCK_M / THREAD_M, BLOCK_N / THREAD_N);  // 16x16 = 256
            dim3 grid((m + BLOCK_M - 1) / BLOCK_M, (n + BLOCK_N - 1) / BLOCK_N);

            tropical_gemm_float_optimized<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N>
                <<<grid, threads, 0, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, transA, transB);

            return CUBLAS_STATUS_SUCCESS;
        }
    }
}

template <>
cublasStatus_t tropical_gemm_optimized_dispatch<double>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    double alpha, const double* A, int lda,
    const double* B, int ldb,
    double beta, double* C, int ldc,
    TropicalTile tile)
{
    bool transA = (transa == CUBLAS_OP_T);
    bool transB = (transb == CUBLAS_OP_T);

    if (transa != CUBLAS_OP_N && transa != CUBLAS_OP_T) return CUBLAS_STATUS_INVALID_VALUE;
    if (transb != CUBLAS_OP_N && transb != CUBLAS_OP_T) return CUBLAS_STATUS_INVALID_VALUE;

    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    switch (tile) {
        case TROPICAL_TILE_32x16x32:
            // Legacy kernel
            return cutmsgemm<double, 32, 16, 32, 4, 4>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

        case TROPICAL_TILE_64x64x32:
        default: {
            // Optimized kernel: 64x64x16 tiles
            // 8x8 work per thread, 64 threads (8x8)
            constexpr int BLOCK_M = 64;
            constexpr int BLOCK_N = 64;
            constexpr int BLOCK_K = 16;
            constexpr int THREAD_M = 8;
            constexpr int THREAD_N = 8;

            dim3 threads(BLOCK_M / THREAD_M, BLOCK_N / THREAD_N);  // 8x8 = 64
            dim3 grid((m + BLOCK_M - 1) / BLOCK_M, (n + BLOCK_N - 1) / BLOCK_N);

            tropical_gemm_double_optimized<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N>
                <<<grid, threads, 0, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, transA, transB);

            return CUBLAS_STATUS_SUCCESS;
        }
    }
}

// New C API functions
#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t cutmsDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc) {
    // Use optimized kernel by default
    return tropical_gemm_optimized_dispatch<double>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, TROPICAL_TILE_64x64x32);
}

cublasStatus_t cutmsSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    // Use optimized kernel by default
    return tropical_gemm_optimized_dispatch<float>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, TROPICAL_TILE_128x128x32);
}

cublasStatus_t cutmsDgemmTiled(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc, TropicalTile tile) {
    return tropical_gemm_optimized_dispatch<double>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, tile);
}

cublasStatus_t cutmsSgemmTiled(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, TropicalTile tile) {
    return tropical_gemm_optimized_dispatch<float>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, tile);
}

#ifdef __cplusplus
}
#endif
