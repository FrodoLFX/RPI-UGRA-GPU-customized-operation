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

__global__ void helloKernel()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU thread %d\n", tid);
}

int main()
{
    int blocks = 2;
    int threadsPerBlock = 4;

    helloKernel<<<blocks, threadsPerBlock>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
