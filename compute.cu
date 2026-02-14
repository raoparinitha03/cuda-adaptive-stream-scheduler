#include <cuda_runtime.h>

__global__ void compute_kernel(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = data[idx] * 2.5f + 1.0f;
}

void launch_kernel(float* d_data, int size, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    compute_kernel<<<blocks, threads, 0, stream>>>(d_data, size);
}
