from task import input_t, output_t

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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def bf16_gemm_transpose_B_scaled(
    A_ptr, B_ptr, C_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_as0, stride_as1,    
    stride_bs0, stride_bs1,    
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                        # [BLOCK_K]

    # -- 1) Load A tile (FP8) and dequantize with a_scale --
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_fp8   = tl.load(a_ptrs,
                      mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                      other=0)
    # 对应的 a_scale 索引：[m, k//128]
    grp_k = offs_k[None, :] // 128                        # shape (1, BLOCK_K)
    scale_a_ptrs = a_scale_ptr + offs_m[:, None] * stride_as0 + grp_k * stride_as1
    scale_a = tl.load(scale_a_ptrs,
                      mask=(offs_m[:, None] < M) & (grp_k < (K // 128)),
                      other=1.0)
    a = tl.cast(a_fp8, tl.float32) * scale_a              # dequantize to fp32

    # -- 2) Load B^T tile (FP8) and dequantize with b_scale --
    # note: loading B^T by swapping k/n strides
    b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
    b_fp8  = tl.load(b_ptrs,
                     mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
                     other=0)
    # 对应的 b_scale 索引：[n//128, k//128]
    grp_n = offs_n[None, :] // 128                        # shape (1, BLOCK_N)
    grp_kT = offs_k[:, None] // 128                       # shape (BLOCK_K, 1)
    scale_b_ptrs = b_scale_ptr + grp_n * stride_bs0 + grp_kT * stride_bs1
    scale_b = tl.load(scale_b_ptrs,
                      mask=(grp_n < (N // 128)) & (grp_kT < (K // 128)),
                      other=1.0)
    b = tl.cast(b_fp8, tl.float32) * scale_b              # dequantize to fp32

    # -- 3) FP32 累加 --
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_ in range(BLOCK_K):
        acc += a[:, k_][:, None] * b[k_, :][None, :]

    # -- 4) Cast 回 bfloat16 并写回 --
    acc_bf16 = tl.cast(acc, tl.bfloat16)
    c_ptrs   = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc_bf16,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    

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
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    bf16_gemm_transpose_B_scaled[grid](
        a, b, c,  #
        a_scale, b_scale,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        a_scale.stride(0), a_scale.stride(1),  # a_scale strides
        b_scale.stride(0), b_scale.stride(1),  # b_scale strides
    )
    return c
