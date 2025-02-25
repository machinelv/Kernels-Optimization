#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"
      



// GEMM kernel v06.
// Each thread in the block processes THREAD_TILE_SIZE_Y *
// THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N,
          size_t BLOCK_TILE_SIZE_K, size_t WARP_SIZE_M, size_t WARP_SIZE_N,
          size_t THREAD_TILE_SIZE_M, size_t THREAD_TILE_SIZE_N, size_t NUM_THREADS>
__global__ void gemm_v06_vectorized_double_buffered(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc) 
{

    size_t constexpr WARP_SIZE = 32;
    static_assert(WARP_SIZE_M * WARP_SIZE_N == WARP_SIZE);
    size_t constexpr WARP_NUM{NUM_THREADS / WARP_SIZE};
    size_t const threadId{threadIdx.y * blockDim.x + threadIdx.x};
    size_t const warp_id = threadId / WARP_SIZE;
    size_t const lane_id = threadId % WARP_SIZE;

    constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
    static_assert(sizeof(int4) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0U);
    static_assert(THREAD_TILE_SIZE_N % NUM_VECTOR_UNITS == 0U);

    size_t constexpr VECTORIZED_THREAD_TILE_SIZE_M{THREAD_TILE_SIZE_M / NUM_VECTOR_UNITS};
    size_t constexpr VECTORIZED_THREAD_TILE_SIZE_N{THREAD_TILE_SIZE_N / NUM_VECTOR_UNITS};
    
    __shared__ T A_block_tile[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M];
    __shared__ T B_block_tile[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    size_t constexpr N_thread_tile_num{BLOCK_TILE_SIZE_N / THREAD_TILE_SIZE_N};

    // element number (T) in a warp tile's row
    size_t constexpr M_WARP_TILE_ELEMENT_NUM{(WARP_SIZE_M * VECTORIZED_THREAD_TILE_SIZE_M)};
    size_t constexpr N_WARP_TILE_ELEMENT_NUM{(WARP_SIZE_N * VECTORIZED_THREAD_TILE_SIZE_N)};

    static_assert(BLOCK_TILE_SIZE_M % (WARP_SIZE_M) == 0U);
    static_assert(BLOCK_TILE_SIZE_N % (WARP_SIZE_N) == 0U);

    size_t constexpr M_thread_tile_vector_num{THREAD_TILE_SIZE_M / NUM_VECTOR_UNITS};
    size_t constexpr M_thread_tile_id_stride{BLOCK_TILE_SIZE_M / VECTORIZED_THREAD_TILE_SIZE_M};
    static_assert(M_thread_tile_id_stride % M_WARP_TILE_ELEMENT_NUM == 0);

    size_t constexpr N_thread_tile_vector_num{THREAD_TILE_SIZE_N / NUM_VECTOR_UNITS};
    size_t constexpr N_thread_tile_id_stride{BLOCK_TILE_SIZE_N / VECTORIZED_THREAD_TILE_SIZE_N};
    static_assert(N_thread_tile_id_stride % N_WARP_TILE_ELEMENT_NUM == 0);


    size_t constexpr M_warp_num_in_stride = M_thread_tile_id_stride / M_WARP_TILE_ELEMENT_NUM;
    size_t constexpr N_warp_num_in_stride = N_thread_tile_id_stride / N_WARP_TILE_ELEMENT_NUM;


    size_t const M_thread_tile_index_start{(warp_id / 2) * 16 + (lane_id / WARP_SIZE_N) * NUM_VECTOR_UNITS};
    size_t const N_thread_tile_index_start{(warp_id % 2) * 32 + (lane_id % WARP_SIZE_N) * NUM_VECTOR_UNITS};

    T C_thread_tile[THREAD_TILE_SIZE_M][THREAD_TILE_SIZE_N] = {static_cast<T>(0)};
    T A_thread_tile[2][THREAD_TILE_SIZE_M] = {static_cast<T>(0)};
    T B_thread_tile[2][THREAD_TILE_SIZE_N] = {static_cast<T>(0)};


    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    

    size_t const A_block_tile_id{blockIdx.y};
    size_t const B_block_tile_id{blockIdx.x};

    /****Pre Load Data****/

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
        size_t const thread_tile_id{threadId + load_idx * NUM_THREADS};
        size_t const tile_index_m{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t const tile_index_k{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_K) * NUM_VECTOR_UNITS};

        size_t const A_index_m{A_block_tile_id * BLOCK_TILE_SIZE_M + tile_index_m};
        size_t const A_index_k{tile_index_k};

        int4 A_row_vector_vals{0, 0, 0, 0};
        if (A_index_m < m && A_index_k < k) {
            A_row_vector_vals = *reinterpret_cast<int4 const*>(&A[A_index_m * lda + A_index_k]);
        }
        if (tile_index_m < BLOCK_TILE_SIZE_M && tile_index_k < BLOCK_TILE_SIZE_K) {
            for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i) {
                A_block_tile[0][tile_index_k + i][tile_index_m] = reinterpret_cast<T const*>(&A_row_vector_vals)[i];
            }
        }
    }

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_N + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
        size_t const thread_tile_id{threadId + load_idx * NUM_THREADS};
        size_t const tile_index_k{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_N};
        size_t const tile_index_n{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_N) * NUM_VECTOR_UNITS};

        size_t const B_index_k{tile_index_k};
        size_t const B_index_n{B_block_tile_id * BLOCK_TILE_SIZE_N + tile_index_n};
        
        int4 B_row_vector_vals{0, 0, 0, 0};
        if (B_index_k < k && B_index_n < n) {
            B_row_vector_vals = *reinterpret_cast<int4 const*>(&B[B_index_k * ldb + B_index_n]);
        }

        if (tile_index_k < BLOCK_TILE_SIZE_K && tile_index_n < BLOCK_TILE_SIZE_N) {
            *reinterpret_cast<int4*>(&B_block_tile[0][tile_index_k][tile_index_n]) = B_row_vector_vals;
        }
    }

    __syncthreads();
    
    #pragma unroll
    for (size_t m_thread_tile_idx{0}; m_thread_tile_idx < VECTORIZED_THREAD_TILE_SIZE_M; ++m_thread_tile_idx) {
        *reinterpret_cast<int4*>(&A_thread_tile[0][m_thread_tile_idx * NUM_VECTOR_UNITS]) = *reinterpret_cast<int4*>(&A_block_tile[0][0][M_thread_tile_index_start + m_thread_tile_idx * M_thread_tile_id_stride]);
    }
    // Load B_block_tile into B_thread_tile
    #pragma unroll
    for (size_t n_thread_tile_idx{0}; n_thread_tile_idx < VECTORIZED_THREAD_TILE_SIZE_N; ++n_thread_tile_idx) {
        *reinterpret_cast<int4*>(&B_thread_tile[0][n_thread_tile_idx * NUM_VECTOR_UNITS]) = *reinterpret_cast<int4*>(&B_block_tile[0][0][N_thread_tile_index_start + n_thread_tile_idx * N_thread_tile_id_stride]); 
    }

    /**** MAIN LOOP ****/
    unsigned int write_stage_idx{1};

    //Move K_block tile in the matrix A and matrix B
    for (size_t K_block_tile_id{0}; K_block_tile_id < (k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K; K_block_tile_id++) {
        // Load A and B into block_tile,
        // and be careful to handle BLOCK_TILE_SIZE_M != BLOCK_TILE_SIZE_N
        //      and BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K != BLOCK_TILE_SIZE_N * BLOCK_TILE_SIZE_K
        unsigned int load_stage_idx{write_stage_idx^1};
        size_t const K_block_tile_start{(K_block_tile_id + 1) * BLOCK_TILE_SIZE_K};

        #pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
            size_t const thread_tile_id{threadId + load_idx * NUM_THREADS};
            size_t const tile_index_m{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_K};
            size_t const tile_index_k{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_K) * NUM_VECTOR_UNITS};
            size_t const A_index_m{A_block_tile_id * BLOCK_TILE_SIZE_M + tile_index_m};
            size_t const A_index_k{K_block_tile_start + tile_index_k};
            *reinterpret_cast<int4*>(&A_block_tile_reg[load_idx * NUM_VECTOR_UNITS]) = *reinterpret_cast<int4 const*>(&A[A_index_m * lda + A_index_k]);
        }
    
        #pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_N + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
            size_t const thread_tile_id{threadId + load_idx * NUM_THREADS};
            size_t const tile_index_k{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_N};
            size_t const tile_index_n{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_N) * NUM_VECTOR_UNITS};
            size_t const B_index_k{K_block_tile_start + tile_index_k};
            size_t const B_index_n{B_block_tile_id * BLOCK_TILE_SIZE_N + tile_index_n};
            
            *reinterpret_cast<int4*>(&B_block_tile_reg[load_idx * NUM_VECTOR_UNITS]) = *reinterpret_cast<int4 const*>(&B[B_index_k * ldb + B_index_n]);
        }

        #pragma unroll
        for (size_t k_block_tile_idx{1}; k_block_tile_idx < BLOCK_TILE_SIZE_K; ++k_block_tile_idx) {
            // load data from shared memory to register
            // Load A_block_tile into A_thread_tile

            #pragma unroll
            for (size_t m_thread_tile_idx{0}; m_thread_tile_idx < VECTORIZED_THREAD_TILE_SIZE_M; ++m_thread_tile_idx) {
                *reinterpret_cast<int4*>(&A_thread_tile[k_block_tile_idx % 2][m_thread_tile_idx * NUM_VECTOR_UNITS]) = *reinterpret_cast<int4*>(&A_block_tile[load_stage_idx][k_block_tile_idx][M_thread_tile_index_start + m_thread_tile_idx * M_thread_tile_id_stride]);
            }
            // Load B_block_tile into B_thread_tile
            #pragma unroll
            for (size_t n_thread_tile_idx{0}; n_thread_tile_idx < VECTORIZED_THREAD_TILE_SIZE_N; ++n_thread_tile_idx) {
                // B_thread_tile[n_thread_tile_idx] = B_block_tile[k_block_tile_idx][N_thread_tile_index_start + n_thread_tile_idx];
                *reinterpret_cast<int4*>(&B_thread_tile[k_block_tile_idx % 2][n_thread_tile_idx * NUM_VECTOR_UNITS]) = *reinterpret_cast<int4*>(&B_block_tile[load_stage_idx][k_block_tile_idx][N_thread_tile_index_start + n_thread_tile_idx * N_thread_tile_id_stride]); 
            }

            // Compute the outer product
            for (size_t m_thread_tile_idx{0}; m_thread_tile_idx < THREAD_TILE_SIZE_M; ++m_thread_tile_idx) {
                for (size_t n_thread_tile_idx{0}; n_thread_tile_idx < THREAD_TILE_SIZE_N; ++n_thread_tile_idx) {
                    // Compute the sum
                    C_thread_tile[m_thread_tile_idx][n_thread_tile_idx] += A_thread_tile[(k_block_tile_idx - 1) % 2][m_thread_tile_idx] * B_thread_tile[(k_block_tile_idx - 1) % 2][n_thread_tile_idx];
                }
            }
            
        }

        /**************Tail Process**************/
        #pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
            size_t const thread_tile_id{threadId + load_idx * NUM_THREADS};
            size_t const tile_index_m{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_K};
            size_t const tile_index_k{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_K) * NUM_VECTOR_UNITS};
    
            if (tile_index_m < BLOCK_TILE_SIZE_M && tile_index_k < BLOCK_TILE_SIZE_K) {
                for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i) {
                    A_block_tile[write_stage_idx][tile_index_k + i][tile_index_m] = A_block_tile_reg[load_idx * NUM_VECTOR_UNITS + i];
                }
            }
        }

        #pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_N + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
            size_t const thread_tile_id{threadId + load_idx * NUM_THREADS};
            size_t const tile_index_k{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_N};
            size_t const tile_index_n{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_N) * NUM_VECTOR_UNITS};

            if (tile_index_k < BLOCK_TILE_SIZE_K && tile_index_n < BLOCK_TILE_SIZE_N) {
                *reinterpret_cast<int4*>(&B_block_tile[write_stage_idx][tile_index_k][tile_index_n]) = *reinterpret_cast<int4 const*>(&B_block_tile_reg[load_idx * NUM_VECTOR_UNITS]);;
            }
        }

        __syncthreads();

        write_stage_idx ^= 1;


        // Compute the outer product
        for (size_t m_thread_tile_idx{0}; m_thread_tile_idx < THREAD_TILE_SIZE_M; ++m_thread_tile_idx) {
            for (size_t n_thread_tile_idx{0}; n_thread_tile_idx < THREAD_TILE_SIZE_N; ++n_thread_tile_idx) {
                // Compute the sum
                C_thread_tile[m_thread_tile_idx][n_thread_tile_idx] += A_thread_tile[1][m_thread_tile_idx] * B_thread_tile[1][n_thread_tile_idx];
            }
        }
        
        // load data from shared memory to register for next iteration
        #pragma unroll
        for (size_t m_thread_tile_idx{0}; m_thread_tile_idx < VECTORIZED_THREAD_TILE_SIZE_M; ++m_thread_tile_idx) {
            *reinterpret_cast<int4*>(&A_thread_tile[0][m_thread_tile_idx * NUM_VECTOR_UNITS]) = *reinterpret_cast<int4*>(&A_block_tile[load_stage_idx^1][0][M_thread_tile_index_start + m_thread_tile_idx * M_thread_tile_id_stride]);
        }
        // Load B_block_tile into B_thread_tile
        #pragma unroll
        for (size_t n_thread_tile_idx{0}; n_thread_tile_idx < VECTORIZED_THREAD_TILE_SIZE_N; ++n_thread_tile_idx) {
            // B_thread_tile[n_thread_tile_idx] = B_block_tile[k_block_tile_idx][N_thread_tile_index_start + n_thread_tile_idx];
            *reinterpret_cast<int4*>(&B_thread_tile[0][n_thread_tile_idx * NUM_VECTOR_UNITS]) = *reinterpret_cast<int4*>(&B_block_tile[load_stage_idx^1][0][N_thread_tile_index_start + n_thread_tile_idx * N_thread_tile_id_stride]); 
        }

        
    }

    // Store the result
    #pragma unroll
    for (size_t m_thread_tile_idx{0}; m_thread_tile_idx < THREAD_TILE_SIZE_M; ++m_thread_tile_idx) {
        #pragma unroll
        for (size_t n_thread_tile_idx{0}; n_thread_tile_idx < VECTORIZED_THREAD_TILE_SIZE_N; ++n_thread_tile_idx) {
            // size_t C_idx_M{m_thread_tile_idx + M_thread_tile_index_start + blockIdx.y * BLOCK_TILE_SIZE_M};
            // size_t C_idx_N{n_thread_tile_idx + N_thread_tile_index_start + blockIdx.x * BLOCK_TILE_SIZE_N};
            // if (C_idx_M < m && C_idx_N < n)
            //     C[C_idx_M * ldc + C_idx_N] = alpha * C_thread_tile[m_thread_tile_idx][n_thread_tile_idx] + beta * C[C_idx_M * ldc + C_idx_N];
            size_t C_idx_M{m_thread_tile_idx % NUM_VECTOR_UNITS + (m_thread_tile_idx / NUM_VECTOR_UNITS) * M_thread_tile_id_stride + M_thread_tile_index_start + blockIdx.y * BLOCK_TILE_SIZE_M};
            size_t C_idx_N{n_thread_tile_idx * N_thread_tile_id_stride + N_thread_tile_index_start + blockIdx.x * BLOCK_TILE_SIZE_N};
            int4 C_reg = *reinterpret_cast<int4*>(&C[C_idx_M * ldc + C_idx_N]);
            int4 C_thread_tile_reg = *reinterpret_cast<int4 const*>(&C_thread_tile[m_thread_tile_idx][n_thread_tile_idx * NUM_VECTOR_UNITS]);
            for (size_t i{0}; i < NUM_VECTOR_UNITS; ++i) {
                reinterpret_cast<T*>(&C_reg)[i] = reinterpret_cast<T const*>(&C_thread_tile_reg)[i] * alpha + reinterpret_cast<T const*>(&C_reg)[i] * beta;
            }
            *reinterpret_cast<int4*>(&C[C_idx_M * ldc + C_idx_N]) = C_reg;
        }
    } 
}

template <typename T>
void launch_gemm_kernel_v06_vectorized_double_buffered(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream) {
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_M{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_N{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};
    constexpr unsigned int THREAD_TILE_SIZE_M{8U};
    constexpr unsigned int THREAD_TILE_SIZE_N{8U};      
    
    constexpr unsigned int WARP_SIZE_M{4U};
    constexpr unsigned int WARP_SIZE_N{8U};

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
    gemm_v06_vectorized_double_buffered<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, WARP_SIZE_M, WARP_SIZE_N, THREAD_TILE_SIZE_M, THREAD_TILE_SIZE_N, NUM_THREADS>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v06_vectorized_double_buffered<float>(
    size_t m, size_t n, size_t k, float const* alpha, float const* A,
    size_t lda, float const* B, size_t ldb, float const* beta, float* C,
    size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v06_vectorized_double_buffered<__half>(
    size_t m, size_t n, size_t k, __half const* alpha, __half const* A,
    size_t lda, __half const* B, size_t ldb, __half const* beta, __half* C,
    size_t ldc, cudaStream_t stream);