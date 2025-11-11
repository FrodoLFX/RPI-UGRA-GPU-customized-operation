// demo04_cutlass_batched_gemm.cpp
#include <cutlass/gemm/device/gemm.h>
#include <iostream>

int main() {
    using T = float;
    const int batch_count = 4;
    const int M = 64, N = 64, K = 64;

    // Allocate host and device buffers for each matrix in the batch
    std::vector<std::vector<T>> host_A(batch_count), host_B(batch_count), host_C(batch_count);
    for (int b = 0; b < batch_count; ++b) {
        host_A[b].resize(M*K);
        host_B[b].resize(K*N);
        host_C[b].resize(M*N);
        // Initialize
        for (int i = 0; i < M*K; ++i) host_A[b][i] = 1.0f;
        for (int i = 0; i < K*N; ++i) host_B[b][i] = 0.5f;
    }

    using Gemm = cutlass::gemm::device::Gemm<
        T, cutlass::layout::RowMajor,
        T, cutlass::layout::RowMajor,
        T, cutlass::layout::RowMajor
    >;

    Gemm gemm;
    for (int b = 0; b < batch_count; ++b) {
        // allocate device memory per batch
        T *dA, *dB, *dC;
        cudaMalloc(&dA, M*K*sizeof(T));
        cudaMalloc(&dB, K*N*sizeof(T));
        cudaMalloc(&dC, M*N*sizeof(T));
        cudaMemcpy(dA, host_A[b].data(), M*K*sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, host_B[b].data(), K*N*sizeof(T), cudaMemcpyHostToDevice);

        typename Gemm::Arguments args({M,N,K}, {dA, K}, {dB, N}, {dC, N}, {dC, N}, {1.0f, 0.0f});
        gemm(args);
        cudaMemcpy(host_C[b].data(), dC, M*N*sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    std::cout << "Batched GEMM first result C[0] = " << host_C[0][0] << std::endl;
    return 0;
}
