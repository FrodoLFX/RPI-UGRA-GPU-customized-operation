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

int main()
{
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);

    float *h_pageable = new float[N];
    float *h_pinned   = nullptr;
    float *d_data     = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    for (int i = 0; i < N; ++i) {
        h_pageable[i] = static_cast<float>(i);
        h_pinned[i]   = static_cast<float>(i);
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Pageable
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_pageable = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pageable, start, stop));

    // Pinned
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_pinned = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pinned, start, stop));

    std::cout << "Pageable memcpy H2D: " << ms_pageable << " ms\n";
    std::cout << "Pinned   memcpy H2D: " << ms_pinned   << " ms\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    delete[] h_pageable;

    return 0;
}
