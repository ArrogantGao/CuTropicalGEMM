# Standard GEMM Benchmarks

This directory contains a CUDA benchmark that compares the handwritten shared-memory GEMM kernel (`cuClassicSgemm`/`cuClassicDgemm`) against cuBLAS `sgemm`/`dgemm` across NN/NT/TN/TT layouts.

## Build

```bash
cmake -S .. -B ../build
cmake --build ../build --target benchmark_standardgemm
```

## Run

```bash
../build/benchmark_standardgemm [--include-small] [--tile=32x16x32|--tile=64x32x32] [--repeats=N]
```

The benchmark prints CSV rows with columns `Type,Layout,opA,opB,M,N,K,AvgMS,FLOPs,GFLOPS` where `Layout` identifies whether the row reports the handwritten kernel (`custom`) or cuBLAS and `GFLOPS` uses the standard `2*M*N*K` operation count.

By default a single `2^16 x 2^16 x 2^16` case is benchmarked; this size consumes tens of terabytes of device memory, so only run it on suitably provisioned systems. Use `--include-small` to append the smaller legacy sweep (`256`–`2048`) after the `2^16` case. `--repeats` controls how many iterations each configuration is averaged over (default 20).

The handwritten kernel uses a configurable tile size: `--tile=64x32x32` (default) or `--tile=32x16x32`. Try both to see which performs better on your GPU.

All benchmark operands use column-major (BLAS/Julia) layout; leading dimensions follow BLAS conventions.
