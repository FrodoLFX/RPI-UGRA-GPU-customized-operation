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

__global__ void vecAdd(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main()
{
    const int N = 1 << 22;
    const size_t bytes = N * sizeof(float);
    const int segments = 4;
    const int segSize = N / segments;

    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));

    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 0.1f;
        h_b[i] = 1.0f;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    cudaStream_t streams[segments];
    for (int i = 0; i < segments; ++i)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    int blockSize = 256;
    int gridSize = (segSize + blockSize - 1) / blockSize;

    for (int s = 0; s < segments; ++s) {
        int offset = s * segSize;

        CUDA_CHECK(cudaMemcpyAsync(d_a + offset, h_a + offset,
                                   segSize * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_b + offset, h_b + offset,
                                   segSize * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[s]));

        vecAdd<<<gridSize, blockSize, 0, streams[s]>>>(d_a + offset, d_b + offset,
                                                       d_c + offset, segSize);

        CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c + offset,
                                   segSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[s]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "h_c[0]   = " << h_c[0] << "\n";
    std::cout << "h_c[last]= " << h_c[N - 1] << "\n";

    for (int i = 0; i < segments; ++i)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));

    return 0;
}
