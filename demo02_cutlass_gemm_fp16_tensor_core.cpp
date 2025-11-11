// demo02_cutlass_gemm_fp16_tensor_core.cpp
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <iostream>

int main() {
    using ElementInput = cutlass::half_t; // 16‑bit floating point
    using ElementOutput = float;          // accumulate into FP32

    const int M = 256, N = 256, K = 128;
    std::vector<ElementInput> hA(M * K), hB(K * N);
    std::vector<ElementOutput> hC(M * N);

    // Initialize inputs
    for (int i = 0; i < M*K; ++i) hA[i] = static_cast<ElementInput>((i % 7) * 0.5f);
    for (int i = 0; i < K*N; ++i) hB[i] = static_cast<ElementInput>((i % 11) * 0.25f);

    // Allocate device memory
    ElementInput *dA, *dB; ElementOutput *dC;
    cudaMalloc(&dA, sizeof(ElementInput) * hA.size());
    cudaMalloc(&dB, sizeof(ElementInput) * hB.size());
    cudaMalloc(&dC, sizeof(ElementOutput) * hC.size());

    cudaMemcpy(dA, hA.data(), sizeof(ElementInput) * hA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeof(ElementInput) * hB.size(), cudaMemcpyHostToDevice);

    using GemmTensorOp = cutlass::gemm::device::Gemm<
        ElementInput, cutlass::layout::RowMajor,
        ElementInput, cutlass::layout::RowMajor,
        ElementOutput, cutlass::layout::RowMajor,
        ElementOutput,                               // accumulate type (FP32)
        cutlass::arch::OpClassTensorOp,              // use Tensor Cores
        cutlass::arch::Sm80                          // target architecture (Ampere or later)
    >;

    GemmTensorOp gemm;
    GemmTensorOp::Arguments arguments(
        {M, N, K},
        {dA, K}, {dB, N},
        {dC, N}, {dC, N},
        {1.0f, 0.0f});

    auto status = gemm(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Tensor Core GEMM failed\n";
        return -1;
    }
    cudaMemcpy(hC.data(), dC, sizeof(ElementOutput) * hC.size(), cudaMemcpyDeviceToHost);
    std::cout << "FP16×FP16→FP32 result C[0] = " << hC[0] << std::endl;

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
