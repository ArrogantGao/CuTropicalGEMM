# Standard GEMM Benchmarks

This directory contains a CUDA benchmark that compares the handwritten shared-memory GEMM kernel (`cuClassicSgemm`/`cuClassicDgemm`) against cuBLAS `sgemm`/`dgemm` across NN/NT/TN/TT layouts.

## Build

```bash
cmake -S .. -B ../build
cmake --build ../build --target benchmark_standardgemm
```

## Run

```bash
../build/benchmark_standardgemm [--include-2p16|--include-large] [--repeats=N]
```

The benchmark prints CSV rows with columns `Type,Layout,opA,opB,M,N,K,AvgMS,FLOPs,GFLOPS` where `Layout` identifies whether the row reports the handwritten kernel (`custom`) or cuBLAS and `GFLOPS` uses the standard `2*M*N*K` operation count.

Use `--include-2p16` to append a `2^16 x 2^16 x 2^16` case. This size requires enormous device memory (tens of terabytes), so only enable it on systems provisioned for that scale. `--repeats` controls how many iterations each configuration is averaged over (default 20).

All benchmark operands use column-major (BLAS/Julia) layout; leading dimensions follow BLAS conventions.
