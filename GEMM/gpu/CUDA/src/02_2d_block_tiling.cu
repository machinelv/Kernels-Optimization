#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"

// GEMM kernel v02.
// Coalesced read and write from global memory.
template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS>
__global__ void gemm_v02(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C, size_t ldc) {
    __shared__ T A_block_tile[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K];
    __shared__ T B_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    // Compute the A's and B's block tile index, which is same among all threads within a block
    unsigned int const A_block_tile_id{blockIdx.y};
    unsigned int const B_block_tile_id{blockIdx.x};
    unsigned int const K_block_tile_num{(k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K};

    size_t const c_row_idx{A_block_tile_id * BLOCK_TILE_SIZE_M + threadIdx.y};
    size_t const c_col_idx{B_block_tile_id * BLOCK_TILE_SIZE_N + threadIdx.x};

    size_t const threadId{threadIdx.y * blockDim.x + threadIdx.x};
    
    T sum_thread{0};
    if (c_row_idx < m && c_col_idx < n) {
        //Move K_block tile in the matrix A and matrix B
        unsigned int K_block_tile_id{0};
        while (K_block_tile_id < K_block_tile_num) {
            // Load A and B into block_tile,
            // and be careful to handle BLOCK_TILE_SIZE_M != BLOCK_TILE_SIZE_N
            //      and BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K != BLOCK_TILE_SIZE_N * BLOCK_TILE_SIZE_K
            load_data_from_global_memory_to_shared_memory<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, NUM_THREADS>(
                A, lda, B, ldb, A_block_tile, B_block_tile, K_block_tile_id, threadId, m, n, k);
            K_block_tile_id++;
            __syncthreads();
            // Compute the sum
            for (size_t k_thread{0}; k_thread < BLOCK_TILE_SIZE_K; ++k_thread) {
                sum_thread += A_block_tile[threadIdx.y][k_thread] * B_block_tile[k_thread][threadIdx.x];
            }
            __syncthreads();
        }
        
        // Store the result
        C[c_row_idx * ldc + c_col_idx] = alpha * sum_thread + beta * C[c_row_idx * ldc + c_col_idx];
    }

} 

template <typename T>
void launch_gemm_kernel_v02(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_M{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_N{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N};
    static_assert(NUM_THREADS <= 1024U);
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    dim3 const block_dim{BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v02<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, NUM_THREADS>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v02<float>(size_t m, size_t n, size_t k,
                                            float const* alpha, float const* A,
                                            size_t lda, float const* B,
                                            size_t ldb, float const* beta,
                                            float* C, size_t ldc,
                                            cudaStream_t stream);
template void launch_gemm_kernel_v02<double>(size_t m, size_t n, size_t k,
                                             double const* alpha,
                                             double const* A, size_t lda,
                                             double const* B, size_t ldb,
                                             double const* beta, double* C,
                                             size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v02<__half>(size_t m, size_t n, size_t k,
                                             __half const* alpha,
                                             __half const* A, size_t lda,
                                             __half const* B, size_t ldb,
                                             __half const* beta, __half* C,
                                             size_t ldc, cudaStream_t stream);