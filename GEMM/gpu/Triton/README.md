# FP8 GEMM

We need to compute a FP8 GEMM kernel. The kernel has two parts:
- A_ = A * A_scale & B_ = B * B_scale
- C = A_ * B_^T

The input parameters' data types are:
- a: torch.Tensor[float8_e4m3fnuz] of shape [m, k], 
- b: torch.Tensor[float8_e4m3fnuz] of shape [n, k], 
- c_scale: torch.Tensor[float32] of shape [m, k // 128], 
- b_scale: torch.Tensor[float32] of shape [n // 128, k // 128], 
- c: torch.Tensor[bfloat16] of shape [m, n]

To fully use the tensor core, we should partitionthe kernel into two parts:
- c_ = a @ b^T
- c = c_ * a_scale * b_scale (element-wise product)


## hardware configuration

AMD MI300:

Max FP8 performance: 2,614.9 TF 


# Optimization


## baseline:



## triton :

### v0
The FP8 GEMM has precious problem in RTX 4070super.




## hip 
- [ ] vector ALU
- [ ] tile partition
- [ ] vectorized fetching
- [ ] swizzled
- [ ] multistage

### v0

- use scalar ALU


### v1

