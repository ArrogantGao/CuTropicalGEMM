#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <algorithm>
#include <type_traits>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "tropicalgemm.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << status << std::endl; \
        exit(1); \
    } \
} while(0)

template<typename T>
void generate_random_matrix(std::vector<T>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
    
    matrix.resize(size);
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<T>(dis(gen));
    }
}

// 特化的调用函数
template<typename T>
cublasStatus_t call_benchmark_gemm(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
                                  int m, int n, int k, T* alpha, T* d_A, int lda, T* d_B, int ldb, T* beta, T* d_C, int ldc);

template<>
cublasStatus_t call_benchmark_gemm<float>(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
                                         int m, int n, int k, float* alpha, float* d_A, int lda, float* d_B, int ldb, float* beta, float* d_C, int ldc) {
    return cutmsSgemm(handle, transA, transB, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
}

template<>
cublasStatus_t call_benchmark_gemm<double>(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
                                          int m, int n, int k, double* alpha, double* d_A, int lda, double* d_B, int ldb, double* beta, double* d_C, int ldc) {
    return cutmsDgemm(handle, transA, transB, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
}

template<typename T>
double benchmark_tropical_gemm(cublasOperation_t transA, cublasOperation_t transB,
                              int m, int n, int k, int warmup_runs = 5, int benchmark_runs = 20) {
    
    // 生成数据
    int lda = (transA == CUBLAS_OP_N) ? k : m;
    int ldb = (transB == CUBLAS_OP_N) ? n : k;
    int ldc = n;
    
    std::vector<T> h_A, h_B, h_C;
    generate_random_matrix(h_A, ((transA == CUBLAS_OP_N) ? m : k) * ((transA == CUBLAS_OP_N) ? k : m));
    generate_random_matrix(h_B, ((transB == CUBLAS_OP_N) ? k : n) * ((transB == CUBLAS_OP_N) ? n : k));
    h_C.resize(m * n, static_cast<T>(-INFINITY));
    
    T alpha = 1.0;
    T beta = static_cast<T>(-INFINITY);
    
    // GPU内存分配
    T *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(T)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(T), cudaMemcpyHostToDevice));
    
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        call_benchmark_gemm(handle, transA, transB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark runs
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < benchmark_runs; ++i) {
        call_benchmark_gemm(handle, transA, transB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time_ms = duration.count() / 1000.0 / benchmark_runs;
    
    // 清理
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    return avg_time_ms;
}

void print_performance_table() {
    std::cout << "\n=== Performance Benchmark Results ===" << std::endl;
    std::cout << std::setw(12) << "Size" 
              << std::setw(8) << "Type"
              << std::setw(6) << "TransA"
              << std::setw(6) << "TransB"
              << std::setw(12) << "Time(ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "Bandwidth(GB/s)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    std::vector<int> sizes = {256, 512, 1024, 2048};
    std::vector<std::pair<cublasOperation_t, std::string>> ops = {
        std::make_pair(CUBLAS_OP_N, "N"),
        std::make_pair(CUBLAS_OP_T, "T")
    };
    
    for (int size : sizes) {
        for (const auto& opA : ops) {
            for (const auto& opB : ops) {
                // Float测试
                double time_ms = benchmark_tropical_gemm<float>(opA.first, opB.first, size, size, size);
                
                // 计算性能指标
                long long flops = 2LL * size * size * size;  // 热带代数：加法 + max运算
                double gflops = (flops / 1e9) / (time_ms / 1000.0);
                
                // 估算内存带宽 (简化计算)
                long long bytes = (long long)(size * size * 3) * sizeof(float);  // A, B, C矩阵
                double bandwidth = (bytes / 1e9) / (time_ms / 1000.0);
                
                std::cout << std::setw(12) << (std::to_string(size) + "x" + std::to_string(size))
                          << std::setw(8) << "float"
                          << std::setw(6) << opA.second
                          << std::setw(6) << opB.second
                          << std::setw(12) << std::fixed << std::setprecision(3) << time_ms
                          << std::setw(12) << std::fixed << std::setprecision(1) << gflops
                          << std::setw(15) << std::fixed << std::setprecision(1) << bandwidth
                          << std::endl;
            }
        }
        
        // Double测试 (只测试一种组合以节省时间)
        double time_ms = benchmark_tropical_gemm<double>(CUBLAS_OP_N, CUBLAS_OP_T, size, size, size);
        long long flops = 2LL * size * size * size;
        double gflops = (flops / 1e9) / (time_ms / 1000.0);
        long long bytes = (long long)(size * size * 3) * sizeof(double);
        double bandwidth = (bytes / 1e9) / (time_ms / 1000.0);
        
        std::cout << std::setw(12) << (std::to_string(size) + "x" + std::to_string(size))
                  << std::setw(8) << "double"
                  << std::setw(6) << "N"
                  << std::setw(6) << "T"
                  << std::setw(12) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << gflops
                  << std::setw(15) << std::fixed << std::setprecision(1) << bandwidth
                  << std::endl;
    }
}

int main() {
    std::cout << "=== CuTropicalGEMM Performance Benchmark ===" << std::endl;
    
    // 设备信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "Memory Bandwidth: " << (prop.memoryBusWidth * prop.memoryClockRate * 2 / 8 / 1000000) << " GB/s" << std::endl;
    
    print_performance_table();
    
    std::cout << "\nNote: GFLOPS calculated as 2*M*N*K operations (add + max per element)" << std::endl;
    std::cout << "      Best performance expected for NT (A not transposed, B transposed)" << std::endl;
    
    return 0;
}