# CuTropicalGEMM
CUDA implementation of gemm under tropical algebra

## Layout

All matrices use column-major (Julia/BLAS) storage by default. Leading dimensions follow the standard
BLAS convention: `lda`/`ldb`/`ldc` specify the number of rows in the underlying, possibly transposed,
operand.
