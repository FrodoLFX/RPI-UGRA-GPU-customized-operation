// demo03_cutlass_gemm_int8.cpp
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <iostream>

int main() {
    using ElementInput = int8_t;
    using ElementOutput = int32_t;
    const int M = 128, N = 64, K = 256;

    std::vector<ElementInput> hA(M * K), hB(K * N);
    std::vector<ElementOutput> hC(M * N);

    // Fill with small integers
    for (int i = 0; i < M*K; ++i) hA[i] = (i % 7) - 3;
    for (int i = 0; i < K*N; ++i) hB[i] = (i % 5) - 2;

    ElementInput *dA, *dB; ElementOutput *dC;
    cudaMalloc(&dA, hA.size() * sizeof(ElementInput));
    cudaMalloc(&dB, hB.size() * sizeof(ElementInput));
    cudaMalloc(&dC, hC.size() * sizeof(ElementOutput));
    cudaMemcpy(dA, hA.data(), hA.size() * sizeof(ElementInput), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), hB.size() * sizeof(ElementInput), cudaMemcpyHostToDevice);

    using GemmInt8 = cutlass::gemm::device::Gemm<
        ElementInput, cutlass::layout::RowMajor,
        ElementInput, cutlass::layout::ColumnMajor,
        ElementOutput, cutlass::layout::RowMajor
    >;

    GemmInt8 gemm;
    GemmInt8::Arguments args({M,N,K}, {dA, K}, {dB, K}, {dC, N}, {dC, N}, {1, 0});
    auto status = gemm(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "INT8 GEMM failed\n";
    }
    cudaMemcpy(hC.data(), dC, hC.size() * sizeof(ElementOutput), cudaMemcpyDeviceToHost);
    std::cout << "INT8 GEMM result C[0] = " << hC[0] << std::endl;

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
