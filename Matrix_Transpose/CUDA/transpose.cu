#include <cuda_runtime.h>


template<typename T, int BLOCK_SIZE>
__global__ void transpose(T *in, T *out, int rows, int cols) {
    __shared__ T tile[32][32];
    
    
}



int main() {
    const size_t rows = 4096;
    const size_t cols = 4096;
    const size_t size = rows * cols;
    const size_t bytes = size * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);

    for (size_t i = 0; i < size; i++) {
        h_in[i] = i;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    transpose<float><<<grid, block>>>(d_in, d_out, rows, cols);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
