#ifndef CUDA_GEMM_UTILS_CUH
#define CUDA_GEMM_UTILS_CUH

#include <cuda_runtime.h>

#include "cuda_gemm_utils.hpp"

template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS>
__device__ void load_data_from_global_memory_to_shared_memory(
    T const* A, size_t lda, T const* B, size_t ldb,
    T A_block_tile[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K],
    T B_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N],
    size_t K_block_tile_id, size_t thread_linear_idx, size_t m, size_t n, size_t k) {

    // size_t const A_block_tile_id{blockIdx.y};

    size_t const K_block_tile_start{K_block_tile_id * BLOCK_TILE_SIZE_K};
    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
        size_t const thread_tile_id{thread_linear_idx + load_idx * NUM_THREADS};
        size_t const tile_index_m{thread_tile_id / BLOCK_TILE_SIZE_K};
        size_t const tile_index_k{thread_tile_id % BLOCK_TILE_SIZE_K};

        size_t const A_index_m{blockIdx.y * BLOCK_TILE_SIZE_M + tile_index_m};
        size_t const A_index_k{K_block_tile_start + tile_index_k};
        T val{0};

        if (A_index_m < m && A_index_k < k) {
            val = A[A_index_m * lda + A_index_k];
        }

        A_block_tile[tile_index_m][tile_index_k] = val;
    }
    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
        size_t const thread_tile_id{thread_linear_idx + load_idx * NUM_THREADS};
        size_t const tile_index_k{thread_tile_id / BLOCK_TILE_SIZE_N};
        size_t const tile_index_n{thread_tile_id % BLOCK_TILE_SIZE_N};

        size_t const B_index_k{K_block_tile_start + tile_index_k};
        size_t const B_index_n{blockIdx.x * BLOCK_TILE_SIZE_N + tile_index_n};
        
        T val{0};
        if (B_index_k < k && B_index_n < n) {
            val = B[B_index_k * ldb + B_index_n];
        }
        B_block_tile[tile_index_k][tile_index_n] = val;
    }

}

template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS>
__device__ void load_data_from_global_memory_to_shared_memory_transposed(
    T const* A, size_t lda, T const* B, size_t ldb,
    T A_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M],
    T B_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N],
    size_t K_block_tile_id, size_t thread_linear_idx, size_t m, size_t n, size_t k) {

    size_t const A_block_tile_id{blockIdx.y};
    size_t const B_block_tile_id{blockIdx.x};

    size_t const K_block_tile_start{K_block_tile_id * BLOCK_TILE_SIZE_K};
    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
        size_t const thread_tile_id{thread_linear_idx + load_idx * NUM_THREADS};
        size_t const tile_index_m{thread_tile_id / BLOCK_TILE_SIZE_K};
        size_t const tile_index_k{thread_tile_id % BLOCK_TILE_SIZE_K};

        // if (tile_index_m < BLOCK_TILE_SIZE_M && tile_index_k < BLOCK_TILE_SIZE_K) {
        size_t const A_index_m{blockIdx.y * BLOCK_TILE_SIZE_M + tile_index_m};
        size_t const A_index_k{K_block_tile_start + tile_index_k};
        T val{0};

        if (A_index_m < m && A_index_k < k) {
            val = A[A_index_m * lda + A_index_k];
        }

        A_block_tile[tile_index_k][tile_index_m] = val;
        // }
    }
    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
        size_t const thread_tile_id{thread_linear_idx + load_idx * NUM_THREADS};
        size_t const tile_index_k{thread_tile_id / BLOCK_TILE_SIZE_N};
        size_t const tile_index_n{thread_tile_id % BLOCK_TILE_SIZE_N};

        // if (tile_index_k < BLOCK_TILE_SIZE_K && tile_index_n < BLOCK_TILE_SIZE_N) {
        size_t const B_index_k{K_block_tile_start + tile_index_k};
        size_t const B_index_n{blockIdx.x * BLOCK_TILE_SIZE_N + tile_index_n};
        
        T val{0};
        if (B_index_k < k && B_index_n < n) {
            val = B[B_index_k * ldb + B_index_n];
        }
        B_block_tile[tile_index_k][tile_index_n] = val;
        // }
    }

}


template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, typename VECTOR_TYPE = int4>
__device__ void load_data_from_global_memory_to_shared_memory_vectorized(
    T const* A, size_t lda, T const* B, size_t ldb,
    T A_thread_block_tile[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K],
    T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N],
    size_t K_block_tile_id, size_t thread_linear_idx, size_t m, size_t n, size_t k) {

    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0U);

    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};

    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0U);

    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    

    size_t const A_block_tile_id{blockIdx.y};
    size_t const B_block_tile_id{blockIdx.x};

    size_t const K_block_tile_start{K_block_tile_id * BLOCK_TILE_SIZE_K};

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
        size_t const thread_tile_id{thread_linear_idx + load_idx * NUM_THREADS};
        size_t const tile_index_m{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t const tile_index_k{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_K) * NUM_VECTOR_UNITS};

        size_t const A_index_m{A_block_tile_id * BLOCK_TILE_SIZE_M + tile_index_m};
        size_t const A_index_k{K_block_tile_start + tile_index_k};

        int4 A_row_vector_vals{0, 0, 0, 0};
        if (A_index_m < m && A_index_k < k) {
            A_row_vector_vals = *reinterpret_cast<int4 const*>(&A[A_index_m * lda + A_index_k]);
        }

        if (A_index_k + NUM_VECTOR_UNITS > k) {
            size_t const num_invalid_elements{A_index_k + NUM_VECTOR_UNITS - k};
            T* const A_row_vector_vals_ptr{reinterpret_cast<T*>(&A_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i) {
                A_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] = static_cast<T>(0);
            }
        }

        if (tile_index_m < BLOCK_TILE_SIZE_M && tile_index_k < BLOCK_TILE_SIZE_K) {
            // for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i) {
            //     A_thread_block_tile[tile_index_m][tile_index_k + i] = reinterpret_cast<T const*>(&A_row_vector_vals)[i];
            // }
            *reinterpret_cast<int4*>(&A_thread_block_tile[tile_index_m][tile_index_k]) = A_row_vector_vals;
        }
    }

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_N + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
        size_t const thread_tile_id{thread_linear_idx + load_idx * NUM_THREADS};
        size_t const tile_index_k{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_N};
        size_t const tile_index_n{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_N) * NUM_VECTOR_UNITS};

        size_t const B_index_k{K_block_tile_start + tile_index_k};
        size_t const B_index_n{B_block_tile_id * BLOCK_TILE_SIZE_N + tile_index_n};
        
        int4 B_row_vector_vals{0, 0, 0, 0};
        if (B_index_k < k && B_index_n < n) {
            B_row_vector_vals = *reinterpret_cast<int4 const*>(&B[B_index_k * ldb + B_index_n]);
        }

        if (B_index_n + NUM_VECTOR_UNITS > n) {
            size_t const num_invalid_elements{B_index_n + NUM_VECTOR_UNITS - n};
            T* const B_row_vector_vals_ptr{reinterpret_cast<T*>(&B_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i) {
                B_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] = static_cast<T>(0);
            }
        }

        if (tile_index_k < BLOCK_TILE_SIZE_K && tile_index_n < BLOCK_TILE_SIZE_N) {
            *reinterpret_cast<int4*>(&B_thread_block_tile[tile_index_k][tile_index_n]) = B_row_vector_vals;
        }
    }
}


template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, typename VECTOR_TYPE = int4>
__device__ void load_data_from_global_memory_to_shared_memory_vectorized_transposed(
    T const* A, size_t lda, T const* B, size_t ldb,
    T A_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M],
    T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N],
    size_t K_block_tile_id, size_t thread_linear_idx, size_t m, size_t n, size_t k) {

    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0U);

    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};

    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0U);

    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    

    size_t const A_block_tile_id{blockIdx.y};
    size_t const B_block_tile_id{blockIdx.x};

    size_t const K_block_tile_start{K_block_tile_id * BLOCK_TILE_SIZE_K};

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
        size_t const thread_tile_id{thread_linear_idx + load_idx * NUM_THREADS};
        size_t const tile_index_m{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t const tile_index_k{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_K) * NUM_VECTOR_UNITS};

        size_t const A_index_m{A_block_tile_id * BLOCK_TILE_SIZE_M + tile_index_m};
        size_t const A_index_k{K_block_tile_start + tile_index_k};

        int4 A_row_vector_vals{0, 0, 0, 0};
        if (A_index_m < m && A_index_k < k) {
            A_row_vector_vals = *reinterpret_cast<int4 const*>(&A[A_index_m * lda + A_index_k]);
        }

        if (A_index_k + NUM_VECTOR_UNITS > k) {
            size_t const num_invalid_elements{A_index_k + NUM_VECTOR_UNITS - k};
            T* const A_row_vector_vals_ptr{reinterpret_cast<T*>(&A_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i) {
                A_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] = static_cast<T>(0);
            }
        }

        if (tile_index_m < BLOCK_TILE_SIZE_M && tile_index_k < BLOCK_TILE_SIZE_K) {
            // *reinterpret_cast<int4*>(&A_thread_block_tile[tile_index_k][tile_index_m]) = A_row_vector_vals;
            for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i) {
                A_thread_block_tile[tile_index_k + i][tile_index_m] = reinterpret_cast<T const*>(&A_row_vector_vals)[i];
            }
        }
    }

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_N + NUM_THREADS - 1) / NUM_THREADS; load_idx ++) {
        size_t const thread_tile_id{thread_linear_idx + load_idx * NUM_THREADS};
        size_t const tile_index_k{thread_tile_id / VECTORIZED_BLOCK_TILE_SIZE_N};
        size_t const tile_index_n{(thread_tile_id % VECTORIZED_BLOCK_TILE_SIZE_N) * NUM_VECTOR_UNITS};

        size_t const B_index_k{K_block_tile_start + tile_index_k};
        size_t const B_index_n{B_block_tile_id * BLOCK_TILE_SIZE_N + tile_index_n};
        
        int4 B_row_vector_vals{0, 0, 0, 0};
        if (B_index_k < k && B_index_n < n) {
            B_row_vector_vals = *reinterpret_cast<int4 const*>(&B[B_index_k * ldb + B_index_n]);
        }

        if (B_index_n + NUM_VECTOR_UNITS > n) {
            size_t const num_invalid_elements{B_index_n + NUM_VECTOR_UNITS - n};
            T* const B_row_vector_vals_ptr{reinterpret_cast<T*>(&B_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i) {
                B_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] = static_cast<T>(0);
            }
        }

        if (tile_index_k < BLOCK_TILE_SIZE_K && tile_index_n < BLOCK_TILE_SIZE_N) {
            *reinterpret_cast<int4*>(&B_thread_block_tile[tile_index_k][tile_index_n]) = B_row_vector_vals;
        }
    }
}
#endif // CUDA_GEMM_UTILS_CUH