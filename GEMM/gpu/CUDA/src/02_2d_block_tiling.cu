#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"

// GEMM kernel v02.




// Coalesced read and write from global memory.
template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C, size_t ldc) {
    __shared__ T A_block_tile[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K];
    __shared__ T B_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    constexpr size_t THREAD_NUM{(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N)};

    constexpr size_t A_block_tile_size{BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K};
    constexpr size_t A_block_tile_thread_size{(A_block_tile_size + THREAD_NUM - 1) / THREAD_NUM};

    constexpr size_t B_block_tile_size{BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N};
    constexpr size_t B_block_tile_thread_size{(B_block_tile_size + THREAD_NUM - 1) / THREAD_NUM};
    
    // Compute the A's and B's block tile index, which is same among all threads within a block
    unsigned int const A_block_tile_id{blockIdx.y};
    unsigned int const B_block_tile_id{blockIdx.x};
    unsigned int const K_block_tile_num{(k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K};
    unsigned int const K_thread_tile_num_x{(BLOCK_TILE_SIZE_K + blockDim.x - 1) / blockDim.x};
    unsigned int const K_thread_tile_num_y{(BLOCK_TILE_SIZE_K + blockDim.y - 1) / blockDim.y};

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

            unsigned int K_block_tile_start{K_block_tile_id * BLOCK_TILE_SIZE_K};

            for (size_t thread_tile_id{threadId * A_block_tile_thread_size}; thread_tile_id < ((threadId + 1) * A_block_tile_thread_size); thread_tile_id++) {
                size_t const tile_index_m{thread_tile_id / BLOCK_TILE_SIZE_K};
                size_t const tile_index_k{thread_tile_id % BLOCK_TILE_SIZE_K};

                if (tile_index_m < BLOCK_TILE_SIZE_M && tile_index_k < BLOCK_TILE_SIZE_K) {
                    size_t const A_index_m{A_block_tile_id * BLOCK_TILE_SIZE_M + tile_index_m};
                    size_t const A_index_k{K_block_tile_start + tile_index_k};
                    T val{0};

                    if (A_index_m < m && A_index_k < k) {
                        val = A[A_index_m * lda + A_index_k];
                    }

                    A_block_tile[tile_index_m][tile_index_k] = val;
                }
            }
            for (size_t thread_tile_id{threadId * B_block_tile_thread_size}; thread_tile_id < ((threadId + 1) * B_block_tile_thread_size); thread_tile_id++) {
                size_t const tile_index_k{thread_tile_id / BLOCK_TILE_SIZE_N};
                size_t const tile_index_n{thread_tile_id % BLOCK_TILE_SIZE_N};

                if (tile_index_k < BLOCK_TILE_SIZE_K && tile_index_n < BLOCK_TILE_SIZE_N) {
                    size_t const B_index_k{K_block_tile_start + tile_index_k};
                    size_t const B_index_n{B_block_tile_id * BLOCK_TILE_SIZE_N + tile_index_n};
                    
                    T val{0};
                    if (B_index_k < k && B_index_n < n) {
                        val = B[B_index_k * ldb + B_index_n];
                    }
                    B_block_tile[tile_index_k][tile_index_n] = val;
                }
            }

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
    gemm_v02<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K>
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