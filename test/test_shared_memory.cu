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


#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// 与你的当前实现完全一致的加载函数
template<typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K>
__device__ __forceinline__ void load_shared_memory_test(
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
    
    // shared_A 加载 - 你的列主序设计
    #pragma unroll
    for (int i = 0; i < (BLOCK_SIZE_M * BLOCK_SIZE_K + total_threads - 1) / total_threads; ++i) {
        int linear_idx = tid + i * total_threads;
        if (linear_idx < BLOCK_SIZE_M * BLOCK_SIZE_K) {
            int k_idx = linear_idx / BLOCK_SIZE_M;  // 你说的：除以M得到K索引
            int m_idx = linear_idx % BLOCK_SIZE_M;  // 模M得到M索引
            
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
    
    // shared_B 加载
    #pragma unroll
    for (int i = 0; i < (BLOCK_SIZE_K * BLOCK_SIZE_N + total_threads - 1) / total_threads; ++i) {
        int linear_idx = tid + i * total_threads;
        if (linear_idx < BLOCK_SIZE_K * BLOCK_SIZE_N) {
            int n_idx = linear_idx % BLOCK_SIZE_N;  // 你的设计
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

// 测试NN模式的kernel
template<typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K>
__global__ void test_NN_kernel(
    const T* global_A, const T* global_B,
    T* output_A, T* output_B,
    int m, int n, int k, int lda, int ldb) {
    
    constexpr int PADDING = 1;
    constexpr int SHARED_A_STRIDE = BLOCK_SIZE_M + PADDING;
    constexpr int SHARED_B_STRIDE = BLOCK_SIZE_N + PADDING;
    
    __shared__ T shared_A[BLOCK_SIZE_K * SHARED_A_STRIDE];
    __shared__ T shared_B[BLOCK_SIZE_K * SHARED_B_STRIDE];
    
    load_shared_memory_test<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(
        shared_A, shared_B, global_A, global_B,
        m, n, k, lda, ldb,
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, blockDim.x, blockDim.y,
        false, false
    );
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;

    // 测试 shared_A 的列主序访问
    for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
        for (int i = tid; i < BLOCK_SIZE_M; i += total_threads) {
            int global_row = blockIdx.x * BLOCK_SIZE_M + i;
            int global_col = blockIdx.z * BLOCK_SIZE_K + k_idx;
            
            if (global_row < m && global_col < k) {
                int output_idx = global_row * k + global_col;
                output_A[output_idx] = shared_A[k_idx * SHARED_A_STRIDE + i];
            }
        }
    }

    // 测试 shared_B 的行主序访问
    for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
        for (int j = tid; j < BLOCK_SIZE_N; j += total_threads) {
            int global_row = blockIdx.z * BLOCK_SIZE_K + k_idx;
            int global_col = blockIdx.y * BLOCK_SIZE_N + j;
            
            if (global_row < k && global_col < n) {
                int output_idx = global_row * n + global_col;
                // 统一的行主序访问：shared_B[k_idx][n_idx]
                output_B[output_idx] = shared_B[k_idx * SHARED_B_STRIDE + j];
            }
        }
    }
}

// 测试NT模式的kernel
template<typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K>
__global__ void test_NT_kernel(
    const T* global_A, const T* global_B,
    T* output_A, T* output_B,
    int m, int n, int k, int lda, int ldb) {
    
    constexpr int PADDING = 1;
    constexpr int SHARED_A_STRIDE = BLOCK_SIZE_M + PADDING;
    constexpr int SHARED_B_STRIDE = BLOCK_SIZE_N + PADDING;
    
    __shared__ T shared_A[BLOCK_SIZE_K * SHARED_A_STRIDE];
    __shared__ T shared_B[BLOCK_SIZE_K * SHARED_B_STRIDE];
    
    load_shared_memory_test<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(
        shared_A, shared_B, global_A, global_B,
        m, n, k, lda, ldb,
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, blockDim.x, blockDim.y,
        false, true
    );
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;
    
    // 测试 shared_A 的列主序访问 (与NN/TN/TT模式一致)
    for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
        for (int i = tid; i < BLOCK_SIZE_M; i += total_threads) {
            int global_row = blockIdx.x * BLOCK_SIZE_M + i;
            int global_col = blockIdx.z * BLOCK_SIZE_K + k_idx;
            
            if (global_row < m && global_col < k) {
                int output_idx = global_row * k + global_col;
                // 统一的列主序访问：shared_A[k_idx][m_idx]
                output_A[output_idx] = shared_A[k_idx * SHARED_A_STRIDE + i];
            }
        }
    }
    
    // 测试 shared_B 的行主序访问
    for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
        for (int j = tid; j < BLOCK_SIZE_N; j += total_threads) {
            int global_row = blockIdx.z * BLOCK_SIZE_K + k_idx;
            int global_col = blockIdx.y * BLOCK_SIZE_N + j;
            
            if (global_row < k && global_col < n) {
                int output_idx = global_row * n + global_col;
                // 统一的行主序访问：shared_B[k_idx][n_idx]
                output_B[output_idx] = shared_B[k_idx * SHARED_B_STRIDE + j];
            }
        }
    }
}

// 测试TN模式的kernel
template<typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K>
__global__ void test_TN_kernel(
    const T* global_A, const T* global_B,
    T* output_A, T* output_B,
    int m, int n, int k, int lda, int ldb) {
    
    constexpr int PADDING = 1;
    constexpr int SHARED_A_STRIDE = BLOCK_SIZE_M + PADDING;
    constexpr int SHARED_B_STRIDE = BLOCK_SIZE_N + PADDING;
    
    __shared__ T shared_A[BLOCK_SIZE_K * SHARED_A_STRIDE];
    __shared__ T shared_B[BLOCK_SIZE_K * SHARED_B_STRIDE];
    
    load_shared_memory_test<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(
        shared_A, shared_B, global_A, global_B,
        m, n, k, lda, ldb,
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, blockDim.x, blockDim.y,
        true, false
    );
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;

    // 测试 shared_A 的列主序访问
    for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
        for (int i = tid; i < BLOCK_SIZE_M; i += total_threads) {
            int global_row = blockIdx.x * BLOCK_SIZE_M + i;
            int global_col = blockIdx.z * BLOCK_SIZE_K + k_idx;
            
            if (global_row < m && global_col < k) {
                int output_idx = global_row * k + global_col;
                output_A[output_idx] = shared_A[k_idx * SHARED_A_STRIDE + i];
            }
        }
    }

    // 测试 shared_B 的行主序访问
    for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
        for (int j = tid; j < BLOCK_SIZE_N; j += total_threads) {
            int global_row = blockIdx.z * BLOCK_SIZE_K + k_idx;
            int global_col = blockIdx.y * BLOCK_SIZE_N + j;
            
            if (global_row < k && global_col < n) {
                int output_idx = global_row * n + global_col;
                // 统一的行主序访问：shared_B[k_idx][n_idx]
                output_B[output_idx] = shared_B[k_idx * SHARED_B_STRIDE + j];
            }
        }
    }
}

// 测试TT模式的kernel
template<typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K>
__global__ void test_TT_kernel(
    const T* global_A, const T* global_B,
    T* output_A, T* output_B,
    int m, int n, int k, int lda, int ldb) {
    
    constexpr int PADDING = 1;
    const int SHARED_A_STRIDE = BLOCK_SIZE_M + PADDING;
    const int SHARED_B_STRIDE = BLOCK_SIZE_N + PADDING;  // 统一为行主序
    
    __shared__ T shared_A[BLOCK_SIZE_K * SHARED_A_STRIDE];
    __shared__ T shared_B[BLOCK_SIZE_K * SHARED_B_STRIDE];  // 统一为行主序布局
    
    load_shared_memory_test<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>(
        shared_A, shared_B, global_A, global_B,
        m, n, k, lda, ldb,
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, blockDim.x, blockDim.y,
        true, true
    );
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;
    
    // 测试 shared_A 的列主序访问
    for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
        for (int i = tid; i < BLOCK_SIZE_M; i += total_threads) {
            int global_row = blockIdx.x * BLOCK_SIZE_M + i;
            int global_col = blockIdx.z * BLOCK_SIZE_K + k_idx;
            
            if (global_row < m && global_col < k) {
                int output_idx = global_row * k + global_col;
                output_A[output_idx] = shared_A[k_idx * SHARED_A_STRIDE + i];
            }
        }
    }
    
    // 测试 shared_B 的行主序访问
    for (int k_idx = 0; k_idx < BLOCK_SIZE_K; ++k_idx) {
        for (int j = tid; j < BLOCK_SIZE_N; j += total_threads) {
            int global_row = blockIdx.z * BLOCK_SIZE_K + k_idx;
            int global_col = blockIdx.y * BLOCK_SIZE_N + j;
            
            if (global_row < k && global_col < n) {
                int output_idx = global_row * n + global_col;
                // 统一的行主序访问：shared_B[k_idx][n_idx]
                output_B[output_idx] = shared_B[k_idx * SHARED_B_STRIDE + j];
            }
        }
    }
}

// 生成测试矩阵
template<typename T>
void generate_test_matrix(std::vector<T>& matrix, int rows, int cols, T min_val = -5.0, T max_val = 5.0) {
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    matrix.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<T>(dis(gen));
    }
}

// CPU参考实现
template<typename T>
void cpu_reference_load(const std::vector<T>& A, const std::vector<T>& B,
                       std::vector<T>& ref_A, std::vector<T>& ref_B,
                       int m, int n, int k, int lda, int ldb,
                       bool transA, bool transB) {
    
    ref_A.assign(m * k, static_cast<T>(-INFINITY));
    ref_B.assign(k * n, static_cast<T>(-INFINITY));
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            if (!transA) {
                ref_A[i * k + j] = A[i * lda + j];
            } else {
                ref_A[i * k + j] = A[j * lda + i];
            }
        }
    }
    
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!transB) {
                ref_B[i * n + j] = B[i * ldb + j];
            } else {
                ref_B[i * n + j] = B[j * ldb + i];
            }
        }
    }
}

// 比较结果
template<typename T>
bool compare_results(const std::vector<T>& gpu_result, const std::vector<T>& cpu_result, 
                    const std::string& matrix_name, T tolerance = 1e-6) {
    bool passed = true;
    int errors = 0;
    const int max_errors_to_show = 5;
    
    for (int i = 0; i < gpu_result.size(); ++i) {
        T gpu_val = gpu_result[i];
        T cpu_val = cpu_result[i];
        T diff = std::abs(gpu_val - cpu_val);
        
        if (diff > tolerance) {
            if (errors < max_errors_to_show) {
                std::cout << matrix_name << " mismatch at [" << i << "]: GPU=" 
                         << gpu_val << ", CPU=" << cpu_val 
                         << ", diff=" << diff << std::endl;
            }
            errors++;
            passed = false;
        }
    }
    
    if (errors > 0) {
        std::cout << matrix_name << " total errors: " << errors << " out of " << gpu_result.size() << " elements" << std::endl;
    }
    
    return passed;
}

// 通用测试函数
template<typename T>
bool test_transpose_mode(bool transA, bool transB, int m, int n, int k, const std::string& test_name) {
    
    std::cout << "\nTesting " << test_name << " (" << m << "x" << n << "x" << k << ")..." << std::endl;
    
    constexpr int BLOCK_SIZE_M = 8;
    constexpr int BLOCK_SIZE_N = 16;
    constexpr int BLOCK_SIZE_K = 32;
    
    // 生成测试数据
    std::vector<T> h_A, h_B;
    
    int lda = transA ? m : k;
    int ldb = transB ? k : n;
    
    generate_test_matrix(h_A, transA ? k : m, transA ? m : k);
    generate_test_matrix(h_B, transB ? n : k, transB ? k : n);
    
    // 分配GPU内存
    T *d_A, *d_B, *d_output_A, *d_output_B;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_output_A, m * k * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_output_B, k * n * sizeof(T)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(T), cudaMemcpyHostToDevice));
    
    // 启动对应的测试kernel
    dim3 threads(4, 4);
    dim3 grid((m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 
              (n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
              (k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K);
    
    if (!transA && !transB) {
        test_NN_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K><<<grid, threads>>>(
            d_A, d_B, d_output_A, d_output_B, m, n, k, lda, ldb);
    } else if (!transA && transB) {
        test_NT_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K><<<grid, threads>>>(
            d_A, d_B, d_output_A, d_output_B, m, n, k, lda, ldb);
    } else if (transA && !transB) {
        test_TN_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K><<<grid, threads>>>(
            d_A, d_B, d_output_A, d_output_B, m, n, k, lda, ldb);
    } else {
        test_TT_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K><<<grid, threads>>>(
            d_A, d_B, d_output_A, d_output_B, m, n, k, lda, ldb);
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 获取GPU结果
    std::vector<T> gpu_A(m * k);
    CHECK_CUDA(cudaMemcpy(gpu_A.data(), d_output_A, m * k * sizeof(T), cudaMemcpyDeviceToHost));
    
    // 生成CPU参考结果
    std::vector<T> ref_A, ref_B;
    cpu_reference_load(h_A, h_B, ref_A, ref_B, m, n, k, lda, ldb, transA, transB);
    
    // 比较结果
    bool passed = compare_results(gpu_A, ref_A, "Matrix A");
    
    // 清理
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_output_A));
    CHECK_CUDA(cudaFree(d_output_B));
    
    std::cout << test_name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

int main() {
    std::cout << "=== Testing All Transpose Modes ===" << std::endl;
    
    bool all_passed = true;
    
    // 测试所有转置组合
    std::vector<std::pair<bool, bool>> transpose_combinations = {
        {false, false},  // NN
        {false, true},   // NT
        {true, false},   // TN
        {true, true}     // TT
    };
    
    std::vector<std::string> names = {"NN", "NT", "TN", "TT"};
    
    // 测试不同大小
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {32, 32, 32},
        {48, 32, 16},
        {64, 64, 64},
        {128, 128, 128}
    };
    
    for (const auto& size : test_sizes) {
        int m, n, k;
        std::tie(m, n, k) = size;
        
        for (int i = 0; i < transpose_combinations.size(); ++i) {
            bool transA = transpose_combinations[i].first;
            bool transB = transpose_combinations[i].second;
            std::string test_name = "Double_" + names[i] + "_" + 
                                  std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
            
            bool passed = test_transpose_mode<double>(transA, transB, m, n, k, test_name);
            all_passed = all_passed && passed;
        }
    }
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Overall result: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    
    return all_passed ? 0 : 1;
}