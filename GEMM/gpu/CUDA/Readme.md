# GEMM CUDA Optimization

This project is based on the [Leimao's GEMM optimization](https://github.com/leimao/CUDA-GEMM-Optimization.git). 
The project uses the execution and timing structure of Leimao's realization. And the project contains three parts:
1. Replicate GEMM according to [Leimao's GEMM optimization blog](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization).
2. **New**: Realize the FP8 GEMM
3. **New**: Realize the multiple GPUs GEMM

# Timeline:

- [ ] Finish the first part before Mar.1 2025.
- [ ] Finish the third part before Mar.7 2025.

# Log

- [ ] Use nsight to analyse the memory coalescing.

## Phase I

### V00

```bash
Device Name: Tesla V100-SXM3-32GB
Memory Size: 31.7394 GB
Peak Bandwitdh: 980.992 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V00
cuBLAS GEMM Kernel Performance
Latency: 12.1866 ms
Effective Bandwidth: 16.5203 GB/s
Effective TFLOPS: 11.2779 TFLOPS
Custom GEMM Kernel Performance
Latency: 771.75 ms
Effective Bandwidth: 0.26087 GB/s
Effective TFLOPS: 0.178087 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.57909%
```

### v01

In this section, ...

```bash
Custom GEMM Kernel V01
cuBLAS GEMM Kernel Performance
Latency: 12.1815 ms
Effective Bandwidth: 16.5272 GB/s
Effective TFLOPS: 11.2826 TFLOPS
Custom GEMM Kernel Performance
Latency: 77.2659 ms
Effective Bandwidth: 2.60563 GB/s
Effective TFLOPS: 1.77878 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 15.7657%
```



### v02

#### v02.1
```bash
Device Name: Tesla V100-SXM3-32GB
Memory Size: 31.7394 GB
Peak Bandwitdh: 980.992 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V02
cuBLAS GEMM Kernel Performance
Latency: 12.1795 ms
Effective Bandwidth: 16.53 GB/s
Effective TFLOPS: 11.2845 TFLOPS
Custom GEMM Kernel Performance
Latency: 42.7428 ms
Effective Bandwidth: 4.71019 GB/s
Effective TFLOPS: 3.21549 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 28.4948%

Matrix Size: M = 8192 N = 4096 K = 4096
Matrix A: 8192 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 8192 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V02
cuBLAS GEMM Kernel Performance
Latency: 23.7763 ms
Effective Bandwidth: 14.1126 GB/s
Effective TFLOPS: 11.561 TFLOPS
Custom GEMM Kernel Performance
Latency: 85.3269 ms
Effective Bandwidth: 3.93246 GB/s
Effective TFLOPS: 3.22147 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 27.8649%
```

Partition the matrix A and matrix B into tiles. While in version 02.1, we only partition the matrices once. And there is a bug in the program: the size of M, N, K tiles can't be changed. This bug will be fixed in version 02.2
