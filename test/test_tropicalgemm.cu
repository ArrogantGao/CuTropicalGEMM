#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <tuple>
#include <string>
#include <type_traits>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "tropicalgemm.h"

// 辅助函数：检查CUDA错误
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << " - " << status << std::endl; \
        exit(1); \
    } \
} while(0)

// 生成随机热带矩阵
template<typename T>
void generate_tropical_matrix(std::vector<T>& matrix, int rows, int cols, T min_val = -10.0, T max_val = 10.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    matrix.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<T>(dis(gen));
    }
}

// CPU参考实现
template<typename T>
void tropical_gemm_cpu_reference(
    bool transA, bool transB,
    int m, int n, int k,
    T alpha, const T* A, int lda,
    const T* B, int ldb,
    T beta, T* C, int ldc) {
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T sum = static_cast<T>(-INFINITY);
            
            for (int l = 0; l < k; ++l) {
                T a_val, b_val;
                
                if (!transA) {
                    a_val = A[i * lda + l];
                } else {
                    a_val = A[l * lda + i];
                }
                
                if (!transB) {
                    b_val = B[l * ldb + j];
                } else {
                    b_val = B[j * ldb + l];
                }
                
                // 热带代数运算: (a ⊗ b) = a + b, (a ⊕ b) = max(a, b)
                T product = a_val + b_val;
                sum = std::max(sum, product);
            }
            
            // 应用alpha和beta
            T result = alpha + sum;
            if (beta != static_cast<T>(-INFINITY)) {
                T old_val = beta + C[i * ldc + j];
                result = std::max(result, old_val);
            }
            
            C[i * ldc + j] = result;
        }
    }
}

// 检查结果正确性
template<typename T>
bool check_results(const std::vector<T>& gpu_result, const std::vector<T>& cpu_result, 
                  int m, int n, T tolerance = 1e-4) {
    bool passed = true;
    int errors = 0;
    const int max_errors_to_show = 10;
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            T gpu_val = gpu_result[idx];
            T cpu_val = cpu_result[idx];
            T diff = std::abs(gpu_val - cpu_val);
            
            if (diff > tolerance) {
                if (errors < max_errors_to_show) {
                    std::cout << "Mismatch at (" << i << "," << j << "): GPU=" 
                             << gpu_val << ", CPU=" << cpu_val 
                             << ", diff=" << diff << std::endl;
                }
                errors++;
                passed = false;
            }
        }
    }
    
    if (errors > 0) {
        std::cout << "Total errors: " << errors << " out of " << (m * n) << " elements" << std::endl;
    }
    
    return passed;
}

// 类型特化的GPU调用函数
template<typename T>
cublasStatus_t call_tropical_gemm(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, 
                                 int m, int n, int k, const T *alpha, const T *A, int lda, 
                                 const T *B, int ldb, const T *beta, T *C, int ldc);

// float特化
template<>
cublasStatus_t call_tropical_gemm<float>(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, 
                                        int m, int n, int k, const float *alpha, const float *A, int lda, 
                                        const float *B, int ldb, const float *beta, float *C, int ldc) {
    return cutmsSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// double特化
template<>
cublasStatus_t call_tropical_gemm<double>(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, 
                                         int m, int n, int k, const double *alpha, const double *A, int lda, 
                                         const double *B, int ldb, const double *beta, double *C, int ldc) {
    return cutmsDgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// 测试单个配置
template<typename T>
bool test_configuration(cublasOperation_t transA, cublasOperation_t transB, 
                       int m, int n, int k, const std::string& test_name) {
    
    std::cout << "\nTesting " << test_name << " (" << m << "x" << n << "x" << k << ")..." << std::endl;
    
    // 生成测试数据
    std::vector<T> h_A, h_B, h_C_gpu, h_C_cpu;
    
    int lda = (transA == CUBLAS_OP_N) ? k : m;
    int ldb = (transB == CUBLAS_OP_N) ? n : k;
    int ldc = n;
    
    generate_tropical_matrix(h_A, (transA == CUBLAS_OP_N) ? m : k, 
                           (transA == CUBLAS_OP_N) ? k : m);
    generate_tropical_matrix(h_B, (transB == CUBLAS_OP_N) ? k : n, 
                           (transB == CUBLAS_OP_N) ? n : k);
    
    h_C_gpu.resize(m * n, static_cast<T>(1.0));
    h_C_cpu.resize(m * n, static_cast<T>(-INFINITY));
    
    T alpha = static_cast<T>(0.0);
    T beta = static_cast<T>(-INFINITY);
    
    // GPU计算
    T *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C_gpu.size() * sizeof(T)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C_gpu.data(), h_C_gpu.size() * sizeof(T), cudaMemcpyHostToDevice));
    
    // 创建cuBLAS句柄
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // 调用GPU函数 - 使用特化函数
    cublasStatus_t status = call_tropical_gemm(handle, transA, transB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "GPU kernel failed with status: " << status << std::endl;
        return false;
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, h_C_gpu.size() * sizeof(T), cudaMemcpyDeviceToHost));
    
    // CPU参考计算
    tropical_gemm_cpu_reference(
        transA == CUBLAS_OP_T, transB == CUBLAS_OP_T,
        m, n, k, alpha, h_A.data(), lda, h_B.data(), ldb, beta, h_C_cpu.data(), ldc
    );
    
    // 检查结果
    bool passed = check_results(h_C_gpu, h_C_cpu, m, n, static_cast<T>(1e-4));
    
    // 清理
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    std::cout << test_name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

int main() {
    std::cout << "=== CuTropicalGEMM Test Suite ===" << std::endl;
    
    // 检查CUDA设备
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    
    bool all_passed = true;
    
    // 测试不同的矩阵大小和转置组合
    std::vector<std::tuple<int, int, int>> test_sizes = {
        std::make_tuple(64, 64, 64),      // 小矩阵
        std::make_tuple(128, 128, 128),   // 中等矩阵
        std::make_tuple(256, 256, 256),   // 大矩阵
        std::make_tuple(512, 256, 128),   // 非正方形矩阵
        std::make_tuple(100, 200, 150)    // 非2的幂矩阵
    };
    
    std::vector<std::pair<cublasOperation_t, std::string>> operations = {
        std::make_pair(CUBLAS_OP_N, "N"),
        std::make_pair(CUBLAS_OP_T, "T")
    };
    
    // 测试float类型
    std::cout << "\n=== Testing Float Precision ===" << std::endl;
    for (const auto& size : test_sizes) {
        int m, n, k;
        std::tie(m, n, k) = size;
        
        for (const auto& opA : operations) {
            for (const auto& opB : operations) {
                std::string test_name = "Float_" + opA.second + opB.second;
                bool passed = test_configuration<float>(opA.first, opB.first, m, n, k, test_name);
                all_passed = all_passed && passed;
            }
        }
    }
    
    // 测试double类型
    std::cout << "\n=== Testing Double Precision ===" << std::endl;
    for (const auto& size : test_sizes) {
        int m, n, k;
        std::tie(m, n, k) = size;
        
        for (const auto& opA : operations) {
            for (const auto& opB : operations) {
                std::string test_name = "Double_" + opA.second + opB.second;
                bool passed = test_configuration<double>(opA.first, opB.first, m, n, k, test_name);
                all_passed = all_passed && passed;
            }
        }
    }
    
    // 边界情况测试
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;
    
    // 很小的矩阵
    all_passed = all_passed && test_configuration<float>(CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, "Float_1x1x1");
    all_passed = all_passed && test_configuration<float>(CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, "Float_2x2x2");
    
    // 很大的K维度
    all_passed = all_passed && test_configuration<float>(CUBLAS_OP_N, CUBLAS_OP_T, 64, 64, 1024, "Float_LargeK");
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Overall result: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    
    return all_passed ? 0 : 1;
}