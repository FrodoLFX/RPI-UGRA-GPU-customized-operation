// demo08_cutlass_async_gemm.cpp
#include <cutlass/gemm/device/gemm_universal.h>
#include <iostream>

int main() {
    using Element = float;
    const int M = 512, N = 512, K = 256;
    std::vector<Element> hA(M*K, 1.0f), hB(K*N, 2.0f), hC(M*N, 0.0f);
    Element *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(Element)*hA.size());
    cudaMalloc(&dB, sizeof(Element)*hB.size());
    cudaMalloc(&dC, sizeof(Element)*hC.size());
    cudaMemcpy(dA, hA.data(), sizeof(Element)*hA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeof(Element)*hB.size(), cudaMemcpyHostToDevice);

    using GemmAsync = cutlass::gemm::device::GemmUniversal<
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::RowMajor,
        Element,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,128,32>,
        cutlass::gemm::GemmShape<64,64,32>,
        cutlass::gemm::GemmShape<16,16,16>
    >;

    GemmAsync gemm;
    // GemmUniversal uses a more general arguments structure that supports strided and batched modes
    typename GemmAsync::Arguments args;
    args.problem_size = {M, N, K};
    args.batch_count = 1;
    args.A = dA; args.lda = K;
    args.B = dB; args.ldb = N;
    args.C = dC; args.ldc = N;
    args.D = dC; args.ldd = N;
    args.alpha = 1.0f; args.beta = 0.0f;

    auto status = gemm(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Async GEMM failed\n";
    }
    cudaMemcpy(hC.data(), dC, sizeof(Element)*hC.size(), cudaMemcpyDeviceToHost);
    std::cout << "Async GEMM result C[0] = " << hC[0] << std::endl;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
