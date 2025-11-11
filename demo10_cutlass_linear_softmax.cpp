// demo10_cutlass_linear_softmax.cpp
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cmath>
#include <iostream>

// CUDA kernel to apply softmax row‑wise
__global__ void row_softmax(float *C, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float max_val = -1e20f;
    for (int j = 0; j < cols; ++j) {
        float v = C[row * cols + j];
        max_val = max(max_val, v);
    }
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) {
        float e = expf(C[row * cols + j] - max_val);
        C[row * cols + j] = e;
        sum += e;
    }
    for (int j = 0; j < cols; ++j) {
        C[row * cols + j] /= sum;
    }
}

int main() {
    const int M = 64, K = 128, N = 32;
    std::vector<float> hA(M*K, 0.1f);
    std::vector<float> hB(K*N, 0.2f);
    std::vector<float> hBias(N, 0.05f);
    std::vector<float> hC(M*N, 0.0f);

    float *dA, *dB, *dC, *dBias;
    cudaMalloc(&dA, sizeof(float)*hA.size());
    cudaMalloc(&dB, sizeof(float)*hB.size());
    cudaMalloc(&dC, sizeof(float)*hC.size());
    cudaMalloc(&dBias, sizeof(float)*hBias.size());
    cudaMemcpy(dA, hA.data(), sizeof(float)*hA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeof(float)*hB.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dBias, hBias.data(), sizeof(float)*hBias.size(), cudaMemcpyHostToDevice);

    // LinearCombination epilogue (adds bias but no activation)
    using Epilogue = cutlass::epilogue::thread::LinearCombination<
        float, 1, float, float>;

    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,128,32>,
        cutlass::gemm::GemmShape<64,64,32>,
        cutlass::gemm::GemmShape<16,16,16>,
        Epilogue
    >;

    Gemm gemm;
    typename Gemm::Arguments args(
        {M,N,K}, {dA,K}, {dB,N}, {dC,N}, {dC,N}, {1.0f, 0.0f});
    args.epilogue.epilogue_op.params.bias = dBias;
    gemm(args);

    // Apply softmax row‑wise using a simple kernel
    int threads = 128;
    int blocks = (M + threads - 1) / threads;
    row_softmax<<<blocks, threads>>>(dC, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hC.data(), dC, sizeof(float)*hC.size(), cudaMemcpyDeviceToHost);
    std::cout << "Linear+Softmax output first row sum = " << std::accumulate(hC.begin(), hC.begin()+N, 0.0f) << std::endl;

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dBias);
    return 0;
}
