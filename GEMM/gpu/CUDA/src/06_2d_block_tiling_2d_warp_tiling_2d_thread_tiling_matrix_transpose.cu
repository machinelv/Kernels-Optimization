// #include <cuda_fp16.h>

// #include "cuda_gemm.hpp"
// #include "cuda_gemm_utils.cuh"
// #include "cuda_gemm_utils.hpp"

// template <typename T, size_t BLOCK_TILE_SIZE, size_t WARP_TILE_SIZE,
//           size_t NUM_THREAD_TILES_PER_WARP, size_t THREAD_TILE_SIZE>
// __device__ void load_data_from_shared_memory_to_register_file(
//     T const thread_block_tile[BLOCK_TILE_SIZE],
//     T register_values[NUM_THREAD_TILES_PER_WARP][THREAD_TILE_SIZE],
//     size_t warp_idx, size_t thread_idx)
// {
//     static_assert(BLOCK_TILE_SIZE % THREAD_TILE_SIZE == 0U);
// #pragma unroll
//     for (size_t thread_tile_repeat_idx{0U};
//          thread_tile_repeat_idx < NUM_THREAD_TILES_PER_WARP;
//          ++thread_tile_repeat_idx)
//     {
//         size_t const thread_block_tile_idx{
//             warp_idx * WARP_TILE_SIZE +
//             thread_tile_repeat_idx *
//                 (WARP_TILE_SIZE / NUM_THREAD_TILES_PER_WARP) +
//             thread_idx * THREAD_TILE_SIZE};
// #pragma unroll
//         for (size_t thread_tile_idx{0U}; thread_tile_idx < THREAD_TILE_SIZE;
//              ++thread_tile_idx)
//         {
//             register_values[thread_tile_repeat_idx][thread_tile_idx] =
//                 thread_block_tile[thread_block_tile_idx + thread_tile_idx];
//         }
//     }
// }

// template <typename T, size_t NUM_THREAD_TILES_PER_WARP_X,
//           size_t NUM_THREAD_TILES_PER_WARP_Y, size_t THREAD_TILE_SIZE_X,
//           size_t THREAD_TILE_SIZE_Y>
// __device__ void compute_thread_tile_results(
//     T const A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y],
//     T const B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X],
//     T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
//                       [THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X])
// {
// // Compute NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X outer
// // products.
// #pragma unroll
//     for (size_t thread_tile_repeat_row_idx{0U};
//          thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
//          ++thread_tile_repeat_row_idx)
//     {
// #pragma unroll
//         for (size_t thread_tile_repeat_col_idx{0U};
//              thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
//              ++thread_tile_repeat_col_idx)
//         {
// #pragma unroll
//             for (size_t thread_tile_y_idx{0U};
//                  thread_tile_y_idx < THREAD_TILE_SIZE_Y; ++thread_tile_y_idx)
//             {
// #pragma unroll
//                 for (size_t thread_tile_x_idx{0U};
//                      thread_tile_x_idx < THREAD_TILE_SIZE_X;
//                      ++thread_tile_x_idx)
//                 {
//                     C_thread_results[thread_tile_repeat_row_idx]
//                                     [thread_tile_repeat_col_idx]
//                                     [thread_tile_y_idx][thread_tile_x_idx] +=
//                         A_vals[thread_tile_repeat_row_idx][thread_tile_y_idx] *
//                         B_vals[thread_tile_repeat_col_idx][thread_tile_x_idx];
//                 }
//             }
//         }
//     }
// }

// template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
//           size_t WARP_TILE_SIZE_X, size_t WARP_TILE_SIZE_Y,
//           size_t THREAD_TILE_SIZE_X, size_t THREAD_TILE_SIZE_Y,
//           size_t NUM_THREAD_TILES_PER_WARP_X,
//           size_t NUM_THREAD_TILES_PER_WARP_Y>
// __device__ void write_results_from_register_file_to_global_memory(
//     T const C_thread_results[NUM_THREAD_TILES_PER_WARP_Y]
//                             [NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_Y]
//                             [THREAD_TILE_SIZE_X],
//     T alpha, T beta, T* C, size_t ldc, size_t m, size_t n, size_t block_row_idx,
//     size_t block_col_idx, size_t warp_row_idx, size_t warp_col_idx,
//     size_t thread_row_idx_in_warp, size_t thread_col_idx_in_warp)
// {
// // Write the results to DRAM.
// #pragma unroll
//     for (size_t thread_tile_repeat_row_idx{0U};
//          thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
//          ++thread_tile_repeat_row_idx)
//     {
// #pragma unroll
//         for (size_t thread_tile_repeat_col_idx{0U};
//              thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
//              ++thread_tile_repeat_col_idx)
//         {
// #pragma unroll
//             for (size_t thread_tile_y_idx{0U};
//                  thread_tile_y_idx < THREAD_TILE_SIZE_Y; ++thread_tile_y_idx)
//             {
// #pragma unroll
//                 for (size_t thread_tile_x_idx{0U};
//                      thread_tile_x_idx < THREAD_TILE_SIZE_X;
//                      ++thread_tile_x_idx)
//                 {
//                     size_t const C_row_idx{
//                         block_row_idx * BLOCK_TILE_SIZE_Y +
//                         warp_row_idx * WARP_TILE_SIZE_Y +
//                         thread_tile_repeat_row_idx *
//                             (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
//                         thread_row_idx_in_warp * THREAD_TILE_SIZE_Y +
//                         thread_tile_y_idx};
//                     size_t const C_col_idx{
//                         block_col_idx * BLOCK_TILE_SIZE_X +
//                         warp_col_idx * WARP_TILE_SIZE_X +
//                         thread_tile_repeat_col_idx *
//                             (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
//                         thread_col_idx_in_warp * THREAD_TILE_SIZE_X +
//                         thread_tile_x_idx};
//                     if (C_row_idx < m && C_col_idx < n)
//                     {
//                         C[C_row_idx * ldc + C_col_idx] =
//                             alpha * C_thread_results[thread_tile_repeat_row_idx]
//                                                     [thread_tile_repeat_col_idx]
//                                                     [thread_tile_y_idx]
//                                                     [thread_tile_x_idx] +
//                             beta * C[C_row_idx * ldc + C_col_idx];
//                     }
//                 }
//             }
//         }
//     }
// }

// // GEMM kernel v06.
// // Each thread in the block processes THREAD_TILE_SIZE_Y *
// // THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// // BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
// template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
//           size_t BLOCK_TILE_SIZE_K, size_t WARP_TILE_SIZE_X,
//           size_t WARP_TILE_SIZE_Y, size_t THREAD_TILE_SIZE_X,
//           size_t THREAD_TILE_SIZE_Y, size_t NUM_THREADS_PER_WARP_X,
//           size_t NUM_THREADS_PER_WARP_Y>
// __global__ void gemm_v06(size_t m, size_t n, size_t k, T alpha, T const* A,
//                          size_t lda, T const* B, size_t ldb, T beta, T* C,
//                          size_t ldc)
// {
//     static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
//     constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
//     static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
//     constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
//     static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);
//     constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{
//         WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)};
//     constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{
//         WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)};
//     static_assert(
//         WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
//     static_assert(
//         WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

//     constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
//     constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};
//     // Avoid using blockDim.x * blockDim.y as the number of threads per block.
//     // Because it is a runtime constant and the compiler cannot optimize the
//     // loop unrolling based on that.
//     // Use a compile time constant instead.
//     constexpr size_t NUM_THREADS{NUM_THREADS_X * NUM_THREADS_Y};

//     // Cache a tile of A and B in shared memory for data reuse.
//     __shared__ T
//         A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
//     __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

//     // A_vals is cached in the register.
//     T A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {
//         static_cast<T>(0)};
//     // B_vals is cached in the register.
//     T B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X] = {
//         static_cast<T>(0)};

//     size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
//     size_t const warp_linear_idx{thread_linear_idx / 32U};
//     size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
//     size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};
//     size_t const thread_linear_idx_in_warp{thread_linear_idx % 32U};
//     size_t const thread_linear_row_idx_in_warp{thread_linear_idx_in_warp /
//                                                NUM_THREADS_PER_WARP_X};
//     size_t const thread_linear_col_idx_in_warp{thread_linear_idx_in_warp %
//                                                NUM_THREADS_PER_WARP_X};

//     // Number of outer loops to perform the sum of inner products.
//     // C_thread_block_tile =
//     // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
//     // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
//     // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
//     size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
//                                         BLOCK_TILE_SIZE_K};
//     // Each thread in the block processes NUM_THREAD_TILES_PER_WARP_Y *
//     // NUM_THREAD_TILES_PER_WARP_X * THREAD_TILE_SIZE_Y *
//     // THREAD_TILE_SIZE_X output values.
//     T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
//                       [THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
//                           static_cast<T>(0)};

//     for (size_t thread_block_tile_idx{0U};
//          thread_block_tile_idx < num_thread_block_tiles;
//          ++thread_block_tile_idx)
//     {
//         load_data_from_global_memory_to_shared_memory_transposed<
//             T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
//             NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile_transposed,
//                          B_thread_block_tile, thread_block_tile_idx,
//                          thread_linear_idx, m, n, k);
//         __syncthreads();

// // Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// // thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// // shared memory as A_thread_block_tile and B_thread_block_tile,
// // respectively. This inner product is further decomposed to
// // BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// // B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// // A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// // both A_thread_block_tile and B_thread_block_tile can be cached in the
// // register.
// #pragma unroll
//         for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
//         {
//             // Load data from shared memory to register file for A.
//             load_data_from_shared_memory_to_register_file<
//                 T, BLOCK_TILE_SIZE_Y, WARP_TILE_SIZE_Y, NUM_THREADS_PER_WARP_Y,
//                 THREAD_TILE_SIZE_Y>(A_thread_block_tile_transposed[k_i], A_vals,
//                                     warp_row_idx,
//                                     thread_linear_row_idx_in_warp);
//             // Load data from shared memory to register file for B.
//             load_data_from_shared_memory_to_register_file<
//                 T, BLOCK_TILE_SIZE_X, WARP_TILE_SIZE_X, NUM_THREADS_PER_WARP_X,
//                 THREAD_TILE_SIZE_X>(B_thread_block_tile[k_i], B_vals,
//                                     warp_col_idx,
//                                     thread_linear_col_idx_in_warp);
//             // Compute NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X
//             // outer products.
//             compute_thread_tile_results<T, NUM_THREAD_TILES_PER_WARP_X,
//                                         NUM_THREAD_TILES_PER_WARP_Y,
//                                         THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>(
//                 A_vals, B_vals, C_thread_results);
//         }
//         __syncthreads();
//     }

//     // Write the results to DRAM.
//     write_results_from_register_file_to_global_memory<
//         T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, WARP_TILE_SIZE_X,
//         WARP_TILE_SIZE_Y, THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y,
//         NUM_THREAD_TILES_PER_WARP_X, NUM_THREAD_TILES_PER_WARP_Y>(
//         C_thread_results, alpha, beta, C, ldc, m, n, blockIdx.y, blockIdx.x,
//         warp_row_idx, warp_col_idx, thread_linear_row_idx_in_warp,
//         thread_linear_col_idx_in_warp);
// }

// template <typename T>
// void launch_gemm_kernel_v06(size_t m, size_t n, size_t k, T const* alpha,
//                             T const* A, size_t lda, T const* B, size_t ldb,
//                             T const* beta, T* C, size_t ldc,
//                             cudaStream_t stream)
// {
//     // Feel free to play with the block tile sizes.
//     // The algorithm correctness should always be guaranteed.
//     constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
//     constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
//     constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

//     constexpr unsigned int WARP_TILE_SIZE_X{32U};
//     constexpr unsigned int WARP_TILE_SIZE_Y{64U};
//     constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
//     constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
//     static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
//     static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

//     constexpr unsigned int THREAD_TILE_SIZE_X{8U};
//     constexpr unsigned int THREAD_TILE_SIZE_Y{8U};

//     constexpr unsigned int NUM_THREADS_PER_WARP_X{4U};
//     constexpr unsigned int NUM_THREADS_PER_WARP_Y{8U};
//     static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
//     static_assert(
//         WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
//     static_assert(
//         WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

//     constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
//     constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

//     constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};

//     dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
//     dim3 const grid_dim{
//         (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
//             BLOCK_TILE_SIZE_X,
//         (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
//             BLOCK_TILE_SIZE_Y,
//         1U};
//     gemm_v06<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
//              WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y, THREAD_TILE_SIZE_X,
//              THREAD_TILE_SIZE_Y, NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y>
//         <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
//                                               *beta, C, ldc);
//     CHECK_LAST_CUDA_ERROR();
// }

// // Explicit instantiation.
// template void launch_gemm_kernel_v06<float>(size_t m, size_t n, size_t k,
//                                             float const* alpha, float const* A,
//                                             size_t lda, float const* B,
//                                             size_t ldb, float const* beta,
//                                             float* C, size_t ldc,
//                                             cudaStream_t stream);
// template void launch_gemm_kernel_v06<double>(size_t m, size_t n, size_t k,
//                                              double const* alpha,
//                                              double const* A, size_t lda,
//                                              double const* B, size_t ldb,
//                                              double const* beta, double* C,
//                                              size_t ldc, cudaStream_t stream);
// template void launch_gemm_kernel_v06<__half>(size_t m, size_t n, size_t k,
//                                              __half const* alpha,
//                                              __half const* A, size_t lda,
//                                              __half const* B, size_t ldb,
//                                              __half const* beta, __half* C,
//                                              size_t ldc, cudaStream_t stream);