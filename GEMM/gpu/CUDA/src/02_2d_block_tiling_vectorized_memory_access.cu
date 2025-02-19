#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"

// GEMM kernel v02.
// Coalesced read and write from global memory.
// We guarantee that matrix A, B, and C are 32 byte aligned.
// This implementation is slower because we waste a lot of threads.
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02_vectorized(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    
}

template <typename T>
void launch_gemm_kernel_v02_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream)
{
    
}

// Explicit instantiation.
template void launch_gemm_kernel_v02_vectorized<float>(
    size_t m, size_t n, size_t k, float const* alpha, float const* A,
    size_t lda, float const* B, size_t ldb, float const* beta, float* C,
    size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v02_vectorized<double>(
    size_t m, size_t n, size_t k, double const* alpha, double const* A,
    size_t lda, double const* B, size_t ldb, double const* beta, double* C,
    size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v02_vectorized<__half>(
    size_t m, size_t n, size_t k, __half const* alpha, __half const* A,
    size_t lda, __half const* B, size_t ldb, __half const* beta, __half* C,
    size_t ldc, cudaStream_t stream);