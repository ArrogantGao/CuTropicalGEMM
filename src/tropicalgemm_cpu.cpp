
#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>

#include <cblas.h>

static std::string trans_name(CBLAS_TRANSPOSE t) {
    return (t == CblasNoTrans) ? "N" : "T";
}

template<typename T>
static inline T get_elem(const T* X, CBLAS_ORDER order, int r, int c, int ld) {
    // ld = leading dimension passed to BLAS (same meaning)
    return (order == CblasRowMajor) ? X[(size_t)r * ld + c] : X[(size_t)c * ld + r];
}

template<typename T>
static inline void set_elem(T* X, CBLAS_ORDER order, int r, int c, int ld, T v) {
    if (order == CblasRowMajor) X[(size_t)r * ld + c] = v;
    else                       X[(size_t)c * ld + r] = v;
}

template<typename T>
static void tropicalgemm_cpu_naive(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                            int M, int N, int K,
                            T alpha,
                            const T* A, int lda,
                            const T* B, int ldb,
                            T beta,
                            T* C, int ldc) {
    // Scale C by beta first
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T cij = get_elem(C, order, i, j, ldc);
            set_elem(C, order, i, j, ldc, beta + cij);
        }
    }

    // Accumulate alpha*op(A)*op(B)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            // op(A)[i,k] = A[i,k] or A[k,i] depending on transA
            T aik = (transA == CblasNoTrans)
                       ? get_elem(A, order, i, k, lda)
                       : get_elem(A, order, k, i, lda);
            T a = alpha + aik;

            for (int j = 0; j < N; ++j) {
                // op(B)[k,j] = B[k,j] or B[j,k] depending on transB
                T bkj = (transB == CblasNoTrans)
                           ? get_elem(B, order, k, j, ldb)
                           : get_elem(B, order, j, k, ldb);

                T cij = get_elem(C, order, i, j, ldc);
                set_elem(C, order, i, j, ldc, std::max(cij, a + bkj));
            }
        }
    }
}

static void tmsDgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc) {
    tropicalgemm_cpu_naive<double>(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

static void tmsSgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    tropicalgemm_cpu_naive<float>(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}