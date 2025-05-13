#include <cuda_fp16.h>
#include <mma.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"


using namespace nvcuda;

// GEMM kernel v07.
// Each thread in the block processes THREAD_TILE_SIZE_Y *
// THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K, 
            size_t WMMA_TILE_SIZE_M, size_t WMMA_TILE_SIZE_N, size_t WMMA_TILE_SIZE_K,
          size_t WMMA_TILE_NUM_M, size_t WMMA_TILE_NUM_N, size_t NUM_THREADS>
__global__ void launch_gemm_kernel_v07_vectorized_double_buffered(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc) 
{
    size_t constexpr WARP_SIZE = 32;
    size_t constexpr WARP_NUM{NUM_THREADS / WARP_SIZE};
    size_t const threadId{threadIdx.y * blockDim.x + threadIdx.x};
    size_t const warp_id = threadId / WARP_SIZE;
    size_t const lane_id = threadId % WARP_SIZE;

    constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
    static_assert(sizeof(int4) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0U);

    size_t const APAD = 0;
    size_t const BPAD = 0;
    
    __shared__ T A_block_tile[2][BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K + APAD];
    __shared__ T B_block_tile[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N + BPAD];

    T A_block_tile_reg[BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K / NUM_THREADS];
    T B_block_tile_reg[BLOCK_TILE_SIZE_N * BLOCK_TILE_SIZE_K / NUM_THREADS];


    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    
    size_t const A_block_tile_id{blockIdx.y};
    size_t const B_block_tile_id{blockIdx.x};

    size_t const comp_c_frag_m{warp_id & 1};
    size_t const comp_c_frag_n{warp_id >> 1};

    wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_K, WMMA_TILE_SIZE_N, T, wmma::row_major> frag_a[2][2][WMMA_TILE_NUM_M];
    wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_K, WMMA_TILE_SIZE_N, T, wmma::row_major> frag_b[2][2][WMMA_TILE_NUM_N];
    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_K, WMMA_TILE_SIZE_N, T> frag_c[WMMA_TILE_NUM_M][WMMA_TILE_NUM_N];

    for (size_t m_wmma_tile_idx{0}; m_wmma_tile_idx < WMMA_TILE_NUM_M; ++m_wmma_tile_idx) {
        for (size_t n_wmma_tile_idx{0}; n_wmma_tile_idx < WMMA_TILE_NUM_N; ++n_wmma_tile_idx) {
            wmma::fill_fragment(frag_c[m_wmma_tile_idx][n_wmma_tile_idx], static_cast<T>(0.0f));
        }
    }

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
            *reinterpret_cast<int4*>(&A_block_tile[0][tile_index_m][tile_index_k]) = A_row_vector_vals;
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
    
    wmma::load_matrix_sync(frag_a[0][0][0], &A_block_tile[0][comp_c_frag_m * 64     ][ 0], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[0][0][1], &A_block_tile[0][comp_c_frag_m * 64 + 16][ 0], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[0][0][2], &A_block_tile[0][comp_c_frag_m * 64 + 32][ 0], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[0][0][3], &A_block_tile[0][comp_c_frag_m * 64 + 48][ 0], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[0][1][0], &A_block_tile[0][comp_c_frag_m * 64     ][16], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[0][1][1], &A_block_tile[0][comp_c_frag_m * 64 + 16][16], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[0][1][2], &A_block_tile[0][comp_c_frag_m * 64 + 32][16], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[0][1][3], &A_block_tile[0][comp_c_frag_m * 64 + 48][16], BLOCK_TILE_SIZE_K + APAD);

    wmma::load_matrix_sync(frag_b[0][0][0], &B_block_tile[0][ 0][comp_c_frag_n * 64     ], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[0][0][1], &B_block_tile[0][ 0][comp_c_frag_n * 64 + 16], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[0][0][2], &B_block_tile[0][ 0][comp_c_frag_n * 64 + 32], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[0][0][3], &B_block_tile[0][ 0][comp_c_frag_n * 64 + 48], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[0][1][0], &B_block_tile[0][16][comp_c_frag_n * 64     ], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[0][1][1], &B_block_tile[0][16][comp_c_frag_n * 64 + 16], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[0][1][2], &B_block_tile[0][16][comp_c_frag_n * 64 + 32], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[0][1][3], &B_block_tile[0][16][comp_c_frag_n * 64 + 48], BLOCK_TILE_SIZE_N + BPAD);
    

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

        // Compute the outer product

        #pragma unroll
        for (int i = 0; i < WMMA_TILE_NUM_M; i++) {
            #pragma unroll
            for (int j = 0; j < WMMA_TILE_NUM_N; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[load_stage_idx][0][i], frag_b[load_stage_idx][0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[load_stage_idx][1][i], frag_b[load_stage_idx][1][j], frag_c[i][j]);
            }
        }
        
        unsigned int const block_tile_load_stage_idx = load_stage_idx;

        wmma::load_matrix_sync(frag_a[write_stage_idx][0][0], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64     ][ 0], BLOCK_TILE_SIZE_K + APAD);
        wmma::load_matrix_sync(frag_a[write_stage_idx][0][1], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 16][ 0], BLOCK_TILE_SIZE_K + APAD);
        wmma::load_matrix_sync(frag_a[write_stage_idx][0][2], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 32][ 0], BLOCK_TILE_SIZE_K + APAD);
        wmma::load_matrix_sync(frag_a[write_stage_idx][0][3], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 48][ 0], BLOCK_TILE_SIZE_K + APAD);
        wmma::load_matrix_sync(frag_a[write_stage_idx][1][0], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64     ][16], BLOCK_TILE_SIZE_K + APAD);
        wmma::load_matrix_sync(frag_a[write_stage_idx][1][1], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 16][16], BLOCK_TILE_SIZE_K + APAD);
        wmma::load_matrix_sync(frag_a[write_stage_idx][1][2], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 32][16], BLOCK_TILE_SIZE_K + APAD);
        wmma::load_matrix_sync(frag_a[write_stage_idx][1][3], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 48][16], BLOCK_TILE_SIZE_K + APAD);

        wmma::load_matrix_sync(frag_b[write_stage_idx][0][0], &B_block_tile[block_tile_load_stage_idx][ 0][comp_c_frag_n * 64     ], BLOCK_TILE_SIZE_N + BPAD);
        wmma::load_matrix_sync(frag_b[write_stage_idx][0][1], &B_block_tile[block_tile_load_stage_idx][ 0][comp_c_frag_n * 64 + 16], BLOCK_TILE_SIZE_N + BPAD);
        wmma::load_matrix_sync(frag_b[write_stage_idx][0][2], &B_block_tile[block_tile_load_stage_idx][ 0][comp_c_frag_n * 64 + 32], BLOCK_TILE_SIZE_N + BPAD);
        wmma::load_matrix_sync(frag_b[write_stage_idx][0][3], &B_block_tile[block_tile_load_stage_idx][ 0][comp_c_frag_n * 64 + 48], BLOCK_TILE_SIZE_N + BPAD);
        wmma::load_matrix_sync(frag_b[write_stage_idx][1][0], &B_block_tile[block_tile_load_stage_idx][16][comp_c_frag_n * 64     ], BLOCK_TILE_SIZE_N + BPAD);
        wmma::load_matrix_sync(frag_b[write_stage_idx][1][1], &B_block_tile[block_tile_load_stage_idx][16][comp_c_frag_n * 64 + 16], BLOCK_TILE_SIZE_N + BPAD);
        wmma::load_matrix_sync(frag_b[write_stage_idx][1][2], &B_block_tile[block_tile_load_stage_idx][16][comp_c_frag_n * 64 + 32], BLOCK_TILE_SIZE_N + BPAD);
        wmma::load_matrix_sync(frag_b[write_stage_idx][1][3], &B_block_tile[block_tile_load_stage_idx][16][comp_c_frag_n * 64 + 48], BLOCK_TILE_SIZE_N + BPAD);


        /**************Tail Process**************/
        #pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
            size_t const thread_tile_id{threadId + load_idx * NUM_THREADS};
            size_t const tile_index_m{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_K};
            size_t const tile_index_k{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_K) * NUM_VECTOR_UNITS};
    
            if (tile_index_m < BLOCK_TILE_SIZE_M && tile_index_k < BLOCK_TILE_SIZE_K) {
                *reinterpret_cast<int4*>(&A_block_tile[write_stage_idx][tile_index_m][tile_index_k]) = *reinterpret_cast<int4 const*>(&A_block_tile_reg[load_idx * NUM_VECTOR_UNITS]);
            }
        }

        #pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_N + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
            size_t const thread_tile_id{threadId + load_idx * NUM_THREADS};
            size_t const tile_index_k{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_N};
            size_t const tile_index_n{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_N) * NUM_VECTOR_UNITS};
            if (tile_index_k < BLOCK_TILE_SIZE_K && tile_index_n < BLOCK_TILE_SIZE_N) {
                *reinterpret_cast<int4*>(&B_block_tile[write_stage_idx][tile_index_k][tile_index_n]) = *reinterpret_cast<int4 const*>(&B_block_tile_reg[load_idx * NUM_VECTOR_UNITS]);
            }
        }

        __syncthreads();

        write_stage_idx ^= 1;

    }

    wmma::load_matrix_sync(frag_a[write_stage_idx][0][0], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64     ][ 0], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[write_stage_idx][0][1], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 16][ 0], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[write_stage_idx][0][2], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 32][ 0], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[write_stage_idx][0][3], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 48][ 0], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[write_stage_idx][1][0], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64     ][16], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[write_stage_idx][1][1], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 16][16], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[write_stage_idx][1][2], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 32][16], BLOCK_TILE_SIZE_K + APAD);
    wmma::load_matrix_sync(frag_a[write_stage_idx][1][3], &A_block_tile[block_tile_load_stage_idx][comp_c_frag_m * 64 + 48][16], BLOCK_TILE_SIZE_K + APAD);

    wmma::load_matrix_sync(frag_b[write_stage_idx][0][0], &B_block_tile[block_tile_load_stage_idx][ 0][comp_c_frag_n * 64     ], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[write_stage_idx][0][1], &B_block_tile[block_tile_load_stage_idx][ 0][comp_c_frag_n * 64 + 16], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[write_stage_idx][0][2], &B_block_tile[block_tile_load_stage_idx][ 0][comp_c_frag_n * 64 + 32], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[write_stage_idx][0][3], &B_block_tile[block_tile_load_stage_idx][ 0][comp_c_frag_n * 64 + 48], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[write_stage_idx][1][0], &B_block_tile[block_tile_load_stage_idx][16][comp_c_frag_n * 64     ], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[write_stage_idx][1][1], &B_block_tile[block_tile_load_stage_idx][16][comp_c_frag_n * 64 + 16], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[write_stage_idx][1][2], &B_block_tile[block_tile_load_stage_idx][16][comp_c_frag_n * 64 + 32], BLOCK_TILE_SIZE_N + BPAD);
    wmma::load_matrix_sync(frag_b[write_stage_idx][1][3], &B_block_tile[block_tile_load_stage_idx][16][comp_c_frag_n * 64 + 48], BLOCK_TILE_SIZE_N + BPAD);

    #pragma unroll
    for (int i = 0; i < WMMA_TILE_NUM_M; i++) {
        #pragma unroll
        for (int j = 0; j < WMMA_TILE_NUM_N; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[load_stage_idx][0][i], frag_b[load_stage_idx][0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[load_stage_idx][1][i], frag_b[load_stage_idx][1][j], frag_c[i][j]);
        }
    }



    size_t const C_idx_M_offset{blockIdx.y * BLOCK_TILE_SIZE_M + comp_c_frag_m * WMMA_TILE_NUM_M * WMMA_TILE_SIZE_M};
    size_t const C_idx_N_offset{blockIdx.x * BLOCK_TILE_SIZE_N + comp_c_frag_n * WMMA_TILE_NUM_N * WMMA_TILE_SIZE_N};
    // Store the result
    #pragma unroll
    for (size_t m_wmma_tile_idx{0}; m_wmma_tile_idx < WMMA_TILE_NUM_M; ++m_wmma_tile_idx) {
        #pragma unroll
        for (size_t n_wmma_tile_idx{0}; n_wmma_tile_idx < WMMA_TILE_NUM_N; ++n_wmma_tile_idx) {
            size_t const C_idx{(C_idx_M_offset + m_wmma_tile_idx * WMMA_TILE_SIZE_M) * ldc + (C_idx_N_offset + n_wmma_tile_idx * WMMA_TILE_SIZE_N)};
            wmma::store_matrix_sync(&C[C_idx], frag_c[m_wmma_tile_idx][n_wmma_tile_idx], ldc, wmma::mem_row_major);
        }
    } 


}

template <typename T>
void launch_gemm_kernel_v07_vectorized_double_buffered(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream) {
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_M{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_N{256U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};

    constexpr unsigned int WMMA_TILE_SIZE_M{16U};
    constexpr unsigned int WMMA_TILE_SIZE_N{16U};  
    constexpr unsigned int WMMA_TILE_SIZE_K{16U};      

    
    constexpr unsigned int WMMA_TILE_NUM_M{4U};
    constexpr unsigned int WMMA_TILE_NUM_N{4U};

    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N / (WMMA_TILE_SIZE_M * WMMA_TILE_NUM_M * WMMA_TILE_SIZE_N * WMMA_TILE_NUM_N) * 32U};

    static_assert(NUM_THREADS <= 1024U);
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);

    dim3 const block_dim{NUM_THREADS, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_N - 1U) / BLOCK_TILE_SIZE_N,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_M - 1U) / BLOCK_TILE_SIZE_M, 1U};
    launch_gemm_kernel_v07_vectorized_double_buffered<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, 
                                                        WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K,
                                                        WMMA_TILE_NUM_M, WMMA_TILE_NUM_N, NUM_THREADS>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v07_vectorized_double_buffered<__half>(
    size_t m, size_t n, size_t k, __half const* alpha, __half const* A,
    size_t lda, __half const* B, size_t ldb, __half const* beta, __half* C,
    size_t ldc, cudaStream_t stream);