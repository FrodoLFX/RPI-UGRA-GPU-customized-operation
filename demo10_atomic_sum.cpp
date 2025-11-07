#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            std::cerr << "CUDA error " << cudaGetErrorString(err__)        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

__global__ void sumKernel(const float* data, float* result, int n)
{
    __shared__ float cache[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < n) val = data[i];

    cache[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            cache[tid] += cache[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, cache[0]);
}

int main()
{
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *h_data = new float[N];
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    sumKernel<<<gridSize, blockSize>>>(d_data, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "Sum = " << h_result << " (expected " << N << ")\n";

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    delete[] h_data;
    return 0;
}
