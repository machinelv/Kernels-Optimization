#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

// GEMM kernel v00.
// Non-coalesced read and write from global memory.
template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    size_t const row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    if (row_idx < m && col_idx < n)
    {
        T c_val{0};
        for (size_t i = 0; i < k; ++i)
        {
            T a_val{A[row_idx * lda + i]};
            T b_val{B[i * ldb + col_idx]};

            c_val += a_val * b_val;
        }

        C[row_idx * ldc + col_idx] = alpha * c_val + beta * C[row_idx * ldc + col_idx];
    }
    
}

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{

    dim3 const block_size{32U, 32U};
    size_t const grid_size_x = (m + block_size.x - 1U) / block_size.x;
    size_t const grid_size_y = (n + block_size.y - 1U) / block_size.y;
    dim3 const grid_size{grid_size_x, grid_size_y};

    gemm_v00<T><<<grid_size, block_size, 0U, stream>>>(m, n, k, *alpha, A, lda,
                                                      B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v00<float>(size_t m, size_t n, size_t k,
                                            float const* alpha, float const* A,
                                            size_t lda, float const* B,
                                            size_t ldb, float const* beta,
                                            float* C, size_t ldc,
                                            cudaStream_t stream);
template void launch_gemm_kernel_v00<double>(size_t m, size_t n, size_t k,
                                             double const* alpha,
                                             double const* A, size_t lda,
                                             double const* B, size_t ldb,
                                             double const* beta, double* C,
                                             size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v00<__half>(size_t m, size_t n, size_t k,
                                             __half const* alpha,
                                             __half const* A, size_t lda,
                                             __half const* B, size_t ldb,
                                             __half const* beta, __half* C,
                                             size_t ldc, cudaStream_t stream);