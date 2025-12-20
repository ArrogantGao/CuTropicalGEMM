# Standard GEMM Benchmarks

This directory contains a CUDA benchmark that compares the handwritten shared-memory GEMM kernel (`cuClassicSgemm`/`cuClassicDgemm`) against cuBLAS `sgemm`/`dgemm` across NN/NT/TN/TT layouts.

## Build

```bash
cmake -S .. -B ../build
cmake --build ../build --target benchmark_standardgemm
```

## Run

```bash
../build/benchmark_standardgemm
```

The benchmark prints CSV rows with columns `Type,Layout,opA,opB,M,N,K,AvgMS` where `Layout` identifies whether the row reports the handwritten kernel (`custom`) or cuBLAS.

All benchmark operands use column-major (BLAS/Julia) layout; leading dimensions follow BLAS conventions.
