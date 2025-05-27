#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"

// GEMM kernel v04.
// Coalesced read and write from global memory.
template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N,
          size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_M, size_t THREAD_TILE_SIZE_N, size_t NUM_THREADS>
__global__ void gemm_v04(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C, size_t ldc) {
    
    __shared__ T A_block_tile[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K];
    __shared__ T B_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    // Compute the A's and B's block tile index, which is same among all threads within a block
    // size_t const M_block_tile_id{blockIdx.y};
    // size_t const N_block_tile_id{blockIdx.x};
    // size_t const K_block_tile_num{(k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K};

    // size_t const M_thread_tile_id{threadIdx.y};
    // size_t const N_thread_tile_id{threadIdx.x};

    size_t const threadId{threadIdx.y * blockDim.x + threadIdx.x};
    size_t constexpr N_thread_tile_num{BLOCK_TILE_SIZE_N / THREAD_TILE_SIZE_N};

    size_t const M_thread_tile_index_start{threadId / N_thread_tile_num * THREAD_TILE_SIZE_M};
    size_t const N_thread_tile_index_start{threadId % N_thread_tile_num * THREAD_TILE_SIZE_N};

    T C_thread_tile[THREAD_TILE_SIZE_M][THREAD_TILE_SIZE_N] = {static_cast<T>(0)};
    T A_thread_tile[THREAD_TILE_SIZE_M] = {static_cast<T>(0)};
    T B_thread_tile[THREAD_TILE_SIZE_N] = {static_cast<T>(0)};
    //Move K_block tile in the matrix A and matrix B

    for (size_t K_block_tile_id{0}; K_block_tile_id < (k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K; K_block_tile_id++) {
        // Load A and B into block_tile,
        // and be careful to handle BLOCK_TILE_SIZE_M != BLOCK_TILE_SIZE_N
        //      and BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K != BLOCK_TILE_SIZE_N * BLOCK_TILE_SIZE_K
        load_data_from_global_memory_to_shared_memory<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, lda, B, ldb, A_block_tile, B_block_tile, K_block_tile_id, threadId, m, n, k);
            
        __syncthreads();
        #pragma unroll
        for (size_t k_block_tile_idx{0}; k_block_tile_idx < BLOCK_TILE_SIZE_K; ++k_block_tile_idx) {
            // load data from shared memory to register
            // Load A_block_tile into A_thread_tile
            #pragma unroll
            for (size_t m_thread_tile_idx{0}; m_thread_tile_idx < THREAD_TILE_SIZE_M; ++m_thread_tile_idx) {
                A_thread_tile[m_thread_tile_idx] = A_block_tile[M_thread_tile_index_start + m_thread_tile_idx][k_block_tile_idx];
            }
            // Load B_block_tile into B_thread_tile
            #pragma unroll
            for (size_t n_thread_tile_idx{0}; n_thread_tile_idx < THREAD_TILE_SIZE_N; ++n_thread_tile_idx) {
                B_thread_tile[n_thread_tile_idx] = B_block_tile[k_block_tile_idx][N_thread_tile_index_start + n_thread_tile_idx];
            }

            // Compute the outer product
            for (size_t m_thread_tile_idx{0}; m_thread_tile_idx < THREAD_TILE_SIZE_M; ++m_thread_tile_idx) {
                for (size_t n_thread_tile_idx{0}; n_thread_tile_idx < THREAD_TILE_SIZE_N; ++n_thread_tile_idx) {
                    // Compute the sum
                    C_thread_tile[m_thread_tile_idx][n_thread_tile_idx] += A_thread_tile[m_thread_tile_idx] * B_thread_tile[n_thread_tile_idx];
                }
            }
            
        }
        __syncthreads();
    }
    size_t const A_block_tile_id{blockIdx.y};
    size_t const B_block_tile_id{blockIdx.x};

    // Store the result
    #pragma unroll
    for (size_t m_thread_tile_idx{0}; m_thread_tile_idx < THREAD_TILE_SIZE_M; ++m_thread_tile_idx) {
        #pragma unroll
        for (size_t n_thread_tile_idx{0}; n_thread_tile_idx < THREAD_TILE_SIZE_N; ++n_thread_tile_idx) {
            size_t C_idx_M{m_thread_tile_idx + M_thread_tile_index_start + A_block_tile_id * BLOCK_TILE_SIZE_M};
            size_t C_idx_N{n_thread_tile_idx + N_thread_tile_index_start + B_block_tile_id * BLOCK_TILE_SIZE_N};
            if (C_idx_M < m && C_idx_N < n)
                C[C_idx_M * ldc + C_idx_N] = alpha * C_thread_tile[m_thread_tile_idx][n_thread_tile_idx] + beta * C[C_idx_M * ldc + C_idx_N];
        }
    } 
}

template <typename T>
void launch_gemm_kernel_v04(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream) {
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_M{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_N{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};
    
    constexpr unsigned int THREAD_TILE_SIZE_M{8U};
    constexpr unsigned int THREAD_TILE_SIZE_N{8U};                     

    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N / (THREAD_TILE_SIZE_N * THREAD_TILE_SIZE_M)};

    static_assert(NUM_THREADS <= 1024U);
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_N % THREAD_TILE_SIZE_N == 0U);
    static_assert(BLOCK_TILE_SIZE_M % THREAD_TILE_SIZE_M == 0U);
    static_assert(NUM_THREADS % THREAD_TILE_SIZE_M == 0U);
    static_assert(NUM_THREADS % THREAD_TILE_SIZE_N == 0U);

    dim3 const block_dim{NUM_THREADS, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_N - 1U) / BLOCK_TILE_SIZE_N,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_M - 1U) / BLOCK_TILE_SIZE_M, 1U};
    gemm_v04<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_M, THREAD_TILE_SIZE_N, NUM_THREADS>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v04<float>(size_t m, size_t n, size_t k,
                                            float const* alpha, float const* A,
                                            size_t lda, float const* B,
                                            size_t ldb, float const* beta,
                                            float* C, size_t ldc,
                                            cudaStream_t stream);
template void launch_gemm_kernel_v04<double>(size_t m, size_t n, size_t k,
                                             double const* alpha,
                                             double const* A, size_t lda,
                                             double const* B, size_t ldb,
                                             double const* beta, double* C,
                                             size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v04<__half>(size_t m, size_t n, size_t k,
                                             __half const* alpha,
                                             __half const* A, size_t lda,
                                             __half const* B, size_t ldb,
                                             __half const* beta, __half* C,
                                             size_t ldc, cudaStream_t stream);