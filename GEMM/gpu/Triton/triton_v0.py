
from task import input_t, output_t
import torch

import triton
import triton.language as tl


DEVICE = "cuda"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_hip_9070xt():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == '?'

def is_hip_mi300():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx942'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Precious Recovery Scale
        as_ptr, bs_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bn, stride_bk,  #
        stride_cm, stride_cn,   
        stride_as0, stride_as1,    
        stride_bs0, stride_bs1,   
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        SCALE_SIZE_M: tl.constexpr, SCALE_SIZE_N: tl.constexpr, SCALE_SIZE_K: tl.constexpr,
        num_stages: tl.constexpr = 1,  #
        ACTIVATION: tl.constexpr = ""  #
):
    """
    Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (N, K) and C has shape (M, N)
    """
    tl.static_assert( (BLOCK_SIZE_M % SCALE_SIZE_M == 0) or (SCALE_SIZE_M % BLOCK_SIZE_M == 0))
    tl.static_assert( (BLOCK_SIZE_N % SCALE_SIZE_N == 0) or (SCALE_SIZE_N % BLOCK_SIZE_N == 0))
    # Make sure that each block of C has only one number in the K dimension.
    tl.static_assert(SCALE_SIZE_K % BLOCK_SIZE_K == 0)


    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    scale_num_m = tl.cdiv(M, SCALE_SIZE_M)
    scale_num_n = tl.cdiv(N, SCALE_SIZE_N)
    scale_num_k = tl.cdiv(K, SCALE_SIZE_K)

    block_scale_num_m: tl.constexpr = (BLOCK_SIZE_M + SCALE_SIZE_M - 1) // SCALE_SIZE_M
    block_scale_num_n: tl.constexpr = (BLOCK_SIZE_N + SCALE_SIZE_N - 1) // SCALE_SIZE_N
    block_scale_num_k: tl.constexpr = (BLOCK_SIZE_K + SCALE_SIZE_K - 1) // SCALE_SIZE_K

    assert block_scale_num_k == 1

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs_T = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    offs_sm = (pid_m * block_scale_num_m + tl.arange(0, block_scale_num_m)) % scale_num_m
    offs_sn = (pid_n * block_scale_num_n + tl.arange(0, block_scale_num_n)) % scale_num_n
    offs_sk = tl.arange(0, block_scale_num_k)

    scale_a_ptrs = as_ptr + (offs_sm[:, None] * stride_as0 + offs_sk[None, :] * stride_as1)
    scale_b_ptrs = bs_ptr + (offs_sn[:, None] * stride_bs0 + offs_sk[None, :] * stride_bs1)
    # scale_b_ptrs_T = bs_ptr + (offs_snT * stride_bs0 + offs_skT * stride_bs1)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of bfloat16 values for higher accuracy.
    # `accumulator` will be converted back to bfloat16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load FP8 A and B from global memory into shared memory.
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs_T, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N), other=0.0)
        
        # Tensor Core dot product.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

        # Load FP32 scale_A and scale_B from global memory into shared memory.
        # block_scale_id_k = (k * BLOCK_SIZE_K) // SCALE_SIZE_K
        # Load the scale factors for A and B.
        # scale_a = tl.load(scale_a_ptrs, mask=(offs_sm[:, None] < scale_num_m) & (offs_sk[None, :] < scale_num_k - block_scale_id_k), other=1.0)
        # scale_b = tl.load(scale_b_ptrs, mask=(offs_sn[:, None] < scale_num_n) & (offs_sk[None, :] < scale_num_k - block_scale_id_k), other=1.0)

        # CUDA Core computation
        # Dequantize A and B by multiplying accumulator with the scale factors.
        # Partition the accumulator into blocks of [BLOCK_SIZE_M // SCALE_SIZE_M, BLOCK_SIZE_N // SCALE_SIZE_K]
        # and each block will be multiplied by two corresponding scale factors (which is a scalar).
        # for i in range(block_scale_num_m):
        #     for j in range(block_scale_num_n):
        #         # sub = accumulator[i*SCALE_SIZE_M:(i + 1)* SCALE_SIZE_M, j*SCALE_SIZE_N:(j + 1)*SCALE_SIZE_N]
        #         scalar_scale_a:tl.float32 = scale_a[i, block_scale_id_k]
        #         scalar_scale_b:tl.float32 = scale_b[j, block_scale_id_k]
        #         # The scale factors are broadcasted to the size of the accumulator.
        #         accumulator[i*SCALE_SIZE_M:(i + 1)* SCALE_SIZE_M,
        #                     j*SCALE_SIZE_N:(j + 1)* SCALE_SIZE_N] = sub *  * 
        
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs_T += BLOCK_SIZE_K * stride_bk

        scale_a_ptrs += block_scale_num_k * stride_as1
        scale_b_ptrs += block_scale_num_k * stride_bs1
    
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in bfloat16!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
## triton_kernel API

def custom_kernel(data: input_t) -> output_t:
    """
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
    """
    # c: [m, n] is pre-allocated memory to avoid timing allocation overhead.
    a, b, a_scale, b_scale, c = data

    # Your implementation here
    a.contiguous()
    b.contiguous()
    a_scale.contiguous()
    b_scale.contiguous()
    c.contiguous()

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    # assert c.is_contiguous(), "Matrix C must be contiguous"

    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        a_scale, b_scale,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        a_scale.stride(0), a_scale.stride(1),  # a_scale strides
        b_scale.stride(0), b_scale.stride(1),  # b_scale strides
        SCALE_SIZE_M=1, SCALE_SIZE_N=128, SCALE_SIZE_K=128,  #
        ACTIVATION=""  #
    )
    print(f"{M=}, {N=}, {K=}")
    print(f"{a.shape=}, {b.shape=}, {a_scale.shape=}, {b_scale.shape=}, {c.shape=}")
    print(f"{a.stride(0)=}, {a.stride(1)=}, {b.stride(0)=}, {b.stride(1)=}, {c.stride(0)=}, {c.stride(1)=}")
    return c





# %%
# Test the kernel
import os

os.environ["TRITON_PRINT_PTX"] = "1"

if hasattr(matmul_kernel, 'asm'):
    ptx_code = matmul_kernel.asm['ptx']
    # print(ptx_code)
    if ".e4m3." in ptx_code:
        print("PTX code likely uses E4M3.")
    elif ".e5m2." in ptx_code:
        print("PTX code likely uses E5M2.")

