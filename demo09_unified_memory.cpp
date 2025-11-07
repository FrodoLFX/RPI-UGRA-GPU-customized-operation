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

__global__ void vecAddUM(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main()
{
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));

    for (int i = 0; i < N; ++i) {
        a[i] = i * 0.1f;
        b[i] = 1.0f;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, device));
    CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, device));
    CUDA_CHECK(cudaMemPrefetchAsync(c, bytes, device));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vecAddUM<<<gridSize, blockSize>>>(a, b, c, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "c[0] = " << c[0] << "\n";
    std::cout << "c[last] = " << c[N - 1] << "\n";

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    return 0;
}
