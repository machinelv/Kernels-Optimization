
#if defined(__HIP_PLATFORM_AMD__)
#define GPU_ARCH HIP
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

// Helper macro for HIP errors
#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(expression)                      \
    if(auto status = (expression); status != hipSuccess) \
    {                                                    \
        fprintf(stderr,                                  \
                "hip error: '%s'(%d) at %s:%d\n",        \
                hipGetErrorString(status),               \
                status,                                  \
                __FILE__,                                \
                __LINE__);                               \
        exit(EXIT_FAILURE);                              \
    }
#endif

#ifndef CHECK_HIPRTC_ERROR
#define CHECK_HIPRTC_ERROR(expression)                       \
    if(auto status = (expression); status != HIPRTC_SUCCESS) \
    {                                                        \
        fprintf(stderr,                                      \
                "hipRTC error: '%s'(%d) at %s:%d\n",         \
                hiprtcGetErrorString(status),                \
                status,                                      \
                __FILE__,                                    \
                __LINE__);                                   \
        exit(EXIT_FAILURE);                                  \
    }
#endif
#else
#define GPU_ARCH CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif



template <typename T_INPUT, typename T_OUTPUT, 
            size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
            size_t WMMA_TILE_SIZE_M, size_t WMMA_TILE_SIZE_N, size_t WMMA_TILE_SIZE_K,
            size_t WMMA_TILE_NUM_M, size_t WMMA_TILE_NUM_N,
            size_t NUM_THREADS, size_t WARP_SIZE, size_t STAGE_NUMS>
__device__ void load_data_from_global_memory_to_shared_memory(
    const T_INPUT* Matrix, T_OUTPUT* block_tile_matrix,
    const size_t M, const size_t N, const size_t K,
    const size_t block_row, const size_t block_col,
    const size_t block_row_offset, const size_t block_col_offset
    ) 
{
    // Load data from global memory to shared memory
    const size_t shared_mem_index = block_row * BLOCK_TILE_SIZE_M + block_col;
    block_tile_matrix[shared_mem_index] = Matrix[block_row_offset + block_col_offset];

    


}








template <typename T_INPUT, typename T_OUTPUT, 
            size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
            size_t WMMA_TILE_SIZE_M, size_t WMMA_TILE_SIZE_N, size_t WMMA_TILE_SIZE_K,
            size_t WMMA_TILE_NUM_M, size_t WMMA_TILE_NUM_N, 
            size_t NUM_THREADS, size_t WARP_SIZE>
__global__ void BF8_GEMM_kernel(const T_INPUT* A, const T_INPUT* B, const float* as, const float* bs, 
                    T_OUTPUT* c, const size_t M, const size_t N, const size_t K) 
{       





}


/*
Reference implementation of block-scale fp8 gemm
Args:
    data: Tuple that expands to:
        a: torch.Tensor[float8_e4m3fnuz] of shape [m, k],
        b: torch.Tensor[float8_e4m3fnuz] of shape [n, k],
        a_scale: torch.Tensor[float32] of shape [m, k // 128],
        b_scale: torch.Tensor[float32] of shape [n // 128, k // 128],
        c: torch.Tensor[bfloat16] of shape [m, n]
Returns:
    Tensor containing output in bf16
*/


void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) 
{
    const size_t M = a.size(0);
    const size_t N = b.size(0);
    const size_t K = a.size(1);



    


}