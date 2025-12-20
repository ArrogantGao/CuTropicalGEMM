#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <random>
#include <vector>
#include <type_traits>
#include <tuple>

#include "tropicalgemm.h"

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << status << std::endl;                         \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

namespace {

std::mt19937 &rng() {
    static std::mt19937 gen(42);
    return gen;
}

template <typename T>
void fill_random(std::vector<T> &data, int rows, int cols) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    data.resize(static_cast<size_t>(rows) * cols);
    for (auto &v : data) {
        v = static_cast<T>(dist(rng()));
    }
}

template <typename T>
float benchmark_once(bool use_custom, cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
                     int m, int n, int k, int repeats) {
    const bool transA = opA == CUBLAS_OP_T;
    const bool transB = opB == CUBLAS_OP_T;

    const int lda = transA ? k : m;
    const int ldb = transB ? n : k;
    const int ldc = m;

    std::vector<T> hA, hB, hC;
    fill_random(hA, transA ? k : m, transA ? m : k);
    fill_random(hB, transB ? n : k, transB ? k : n);
    hC.assign(static_cast<size_t>(m) * n, static_cast<T>(0));

    T *dA = nullptr;
    T *dB = nullptr;
    T *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(T) * hA.size()));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(T) * hB.size()));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(T) * hC.size()));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(T) * hA.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(T) * hB.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeof(T) * hC.size(), cudaMemcpyHostToDevice));

    const T alpha = static_cast<T>(1.0);
    const T beta = static_cast<T>(0.0);

    // Warmup
    if (use_custom) {
        if constexpr (std::is_same_v<T, float>) {
            CHECK_CUBLAS(cuClassicSgemm(handle, opA, opB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc));
        } else {
            CHECK_CUBLAS(cuClassicDgemm(handle, opA, opB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc));
        }
    } else {
        if constexpr (std::is_same_v<T, float>) {
            CHECK_CUBLAS(cublasSgemm(handle, opA, opB, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
        } else {
            CHECK_CUBLAS(cublasDgemm(handle, opA, opB, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        if (use_custom) {
            if constexpr (std::is_same_v<T, float>) {
                CHECK_CUBLAS(cuClassicSgemm(handle, opA, opB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc));
            } else {
                CHECK_CUBLAS(cuClassicDgemm(handle, opA, opB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc));
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                CHECK_CUBLAS(cublasSgemm(handle, opA, opB, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
            } else {
                CHECK_CUBLAS(cublasDgemm(handle, opA, opB, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
            }
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return ms / repeats;
}

void print_header() {
    std::cout << "Type,Layout,opA,opB,M,N,K,AvgMS" << std::endl;
}

template <typename T>
void run_suite(const char *type_name, const std::vector<std::tuple<int, int, int>> &sizes) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::vector<std::pair<cublasOperation_t, std::string>> ops = {
        {CUBLAS_OP_N, "N"},
        {CUBLAS_OP_T, "T"},
    };

    const int repeats = 20;

    for (const auto &[m, n, k] : sizes) {
        for (const auto &opA : ops) {
            for (const auto &opB : ops) {
                float ms_custom = benchmark_once<T>(true, handle, opA.first, opB.first, m, n, k, repeats);
                float ms_cublas = benchmark_once<T>(false, handle, opA.first, opB.first, m, n, k, repeats);
                std::cout << type_name << ",custom," << opA.second << ',' << opB.second << ','
                          << m << ',' << n << ',' << k << ',' << ms_custom << std::endl;
                std::cout << type_name << ",cublas," << opA.second << ',' << opB.second << ','
                          << m << ',' << n << ',' << k << ',' << ms_cublas << std::endl;
            }
        }
    }

    CHECK_CUBLAS(cublasDestroy(handle));
}

} // namespace

int main() {
    std::vector<std::tuple<int, int, int>> sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 1024, 512},
    };

    print_header();
    run_suite<float>("float", sizes);
    run_suite<double>("double", sizes);

    return 0;
}
