#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cblas.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include <type_traits>

#include "tropicalgemm.h"

#define CHECK_CUDA(call)                                                                          \
    do {                                                                                          \
        cudaError_t err = (call);                                                                 \
        if (err != cudaSuccess) {                                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "              \
                      << cudaGetErrorString(err) << std::endl;                                    \
            std::exit(EXIT_FAILURE);                                                              \
        }                                                                                         \
    } while (0)

#define CHECK_CUBLAS(call)                                                                       \
    do {                                                                                         \
        cublasStatus_t status = (call);                                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                   \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - "         \
                      << status << std::endl;                                                   \
            std::exit(EXIT_FAILURE);                                                             \
        }                                                                                        \
    } while (0)

template <typename T>
void fill_random(std::vector<T> &data, int rows, int cols, T min_val = static_cast<T>(-2.0),
                 T max_val = static_cast<T>(2.0)) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    data.resize(static_cast<size_t>(rows) * cols);
    for (auto &v : data) {
        v = static_cast<T>(dist(gen));
    }
}

template <typename T>
void gemm_cpu_reference(bool transA, bool transB, int m, int n, int k, T alpha, const T *A, int lda,
                        const T *B, int ldb, T beta, std::vector<T> &C) {
    const CBLAS_TRANSPOSE opA = transA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE opB = transB ? CblasTrans : CblasNoTrans;

    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasColMajor, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C.data(), m);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasColMajor, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C.data(), m);
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type for CBLAS reference");
    }
}

template <typename T>
bool nearly_equal(const std::vector<T> &lhs, const std::vector<T> &rhs) {
    const T abs_tol = std::is_same_v<T, float> ? static_cast<T>(1e-3) : static_cast<T>(1e-9);
    const T rel_tol = std::is_same_v<T, float> ? static_cast<T>(1e-3) : static_cast<T>(1e-9);

    for (size_t i = 0; i < lhs.size(); ++i) {
        const T diff = std::abs(lhs[i] - rhs[i]);
        const T scale = std::max(std::abs(lhs[i]), std::abs(rhs[i]));
        if (diff > abs_tol + rel_tol * scale) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool run_case(cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k, const std::string &name) {
    const bool transA = opA == CUBLAS_OP_T;
    const bool transB = opB == CUBLAS_OP_T;

    const int lda = transA ? k : m;
    const int ldb = transB ? n : k;
    const int ldc = m;

    std::vector<T> hA, hB, hC, refC;
    fill_random(hA, transA ? k : m, transA ? m : k);
    fill_random(hB, transB ? n : k, transB ? k : n);
    hC.assign(static_cast<size_t>(m) * n, static_cast<T>(1));
    refC = hC;

    T *dA = nullptr;
    T *dB = nullptr;
    T *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeof(T) * hA.size()));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(T) * hB.size()));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(T) * hC.size()));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(T) * hA.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(T) * hB.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeof(T) * hC.size(), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(1.0);

    cublasStatus_t status;
    if constexpr (std::is_same_v<T, float>) {
        status = cuClassicSgemm(handle, opA, opB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    } else {
        status = cuClassicDgemm(handle, opA, opB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << name << " launch failed with status " << status << std::endl;
        return false;
    }

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeof(T) * hC.size(), cudaMemcpyDeviceToHost));
    gemm_cpu_reference(transA, transB, m, n, k, alpha, hA.data(), lda, hB.data(), ldb, beta, refC);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUBLAS(cublasDestroy(handle));

    bool ok = nearly_equal(hC, refC);
    std::cout << name << ": " << (ok ? "PASSED" : "FAILED") << std::endl;
    return ok;
}

int main() {
    std::vector<std::tuple<int, int, int>> sizes = {
        {16, 16, 16},
        {32, 24, 28},
        {64, 64, 32},
    };

    std::vector<std::pair<cublasOperation_t, std::string>> ops = {
        {CUBLAS_OP_N, "N"},
        {CUBLAS_OP_T, "T"},
    };

    bool all_passed = true;

    for (const auto &size : sizes) {
        int m, n, k;
        std::tie(m, n, k) = size;

        for (const auto &opA : ops) {
            for (const auto &opB : ops) {
                std::string name = "float_" + opA.second + opB.second + "_" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
                all_passed = run_case<float>(opA.first, opB.first, m, n, k, name) && all_passed;
            }
        }
    }

    for (const auto &size : sizes) {
        int m, n, k;
        std::tie(m, n, k) = size;

        for (const auto &opA : ops) {
            for (const auto &opB : ops) {
                std::string name = "double_" + opA.second + opB.second + "_" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
                all_passed = run_case<double>(opA.first, opB.first, m, n, k, name) && all_passed;
            }
        }
    }

    std::cout << (all_passed ? "All standard GEMM tests passed" : "Standard GEMM tests failed") << std::endl;
    return all_passed ? 0 : 1;
}

