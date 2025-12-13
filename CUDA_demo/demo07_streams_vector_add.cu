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

    // Helper: enqueue one full "streams pipeline" (H2D + kernel + D2H) for all segments
    auto enqueue_once = [&]() {
        for (int s = 0; s < segments; ++s) {
            int offset = s * segSize;

            CUDA_CHECK(cudaMemcpyAsync(d_a + offset, h_a + offset,
                segSize * sizeof(float),
                cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(d_b + offset, h_b + offset,
                segSize * sizeof(float),
                cudaMemcpyHostToDevice, streams[s]));

            vecAdd << <gridSize, blockSize, 0, streams[s] >> > (d_a + offset, d_b + offset,
                d_c + offset, segSize);
            CUDA_CHECK(cudaGetLastError()); // catch launch errors early

            CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c + offset,
                segSize * sizeof(float),
                cudaMemcpyDeviceToHost, streams[s]));
        }
        };

    // Warm-up (avoid first-run overhead skewing results)
    enqueue_once();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 200;
    CUDA_CHECK(cudaEventRecord(start));

    for (int it = 0; it < iters; ++it) {
        enqueue_once();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iters;

    // Effective transfer per iteration: H2D(A) + H2D(B) + D2H(C) = 3 * bytes
    double total_bytes = 3.0 * (double)bytes;
    double gbps = total_bytes / (avg_ms / 1e3) / 1e9;

    std::cout << "Avg end-to-end time (streams pipeline): " << avg_ms << " ms"
        << " | Effective bandwidth: " << gbps << " GB/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


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
