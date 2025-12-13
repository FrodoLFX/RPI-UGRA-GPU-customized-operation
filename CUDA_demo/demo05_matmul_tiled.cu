#include <iostream>
#include <vector>
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

const int TILE = 16;

__global__ void matMulTiled(const float* A, const float* B, float* C, int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        if (row < N && t * TILE + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

int main()
{
    const int N = 512;
    const size_t bytes = N * N * sizeof(float);

    std::vector<float> h_A(N * N, 1.0f);
    std::vector<float> h_B(N * N, 2.0f);
    std::vector<float> h_C(N * N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // --- Timing setup (CUDA events measure GPU time) ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up (helps avoid first-run overhead affecting timing)
    matMulTiled << <grid, block >> > (d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    const int iters = 200;  // you can change to 100/500
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iters; ++i) {
        matMulTiled << <grid, block >> > (d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iters;

    // Optional: compute throughput (approx. 2*N^3 FLOPs for GEMM)
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = flops / (avg_ms / 1e3) / 1e9;

    std::cout << "Kernel avg time: " << avg_ms << " ms"
        << " | Throughput: " << gflops << " GFLOP/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << "C[0] = " << h_C[0] << "\n";
    std::cout << "C[N*N-1] = " << h_C.back() << "\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
