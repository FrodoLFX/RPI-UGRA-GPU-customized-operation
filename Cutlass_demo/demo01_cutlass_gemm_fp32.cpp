// demo01_cutlass_gemm_fp32.cpp
#include <cutlass/gemm/device/gemm.h>
#include <iostream>

int main() {
    using Element = float;
    static const int M = 128, N = 128, K = 64;
    // Host matrices
    std::vector<Element> host_A(M * K);
    std::vector<Element> host_B(K * N);
    std::vector<Element> host_C(M * N);

    // Initialize A and B with simple values
    for (int i = 0; i < M * K; ++i) host_A[i] = static_cast<Element>(i % 3 + 1);
    for (int i = 0; i < K * N; ++i) host_B[i] = static_cast<Element>((i % 5) - 2);

    // Allocate device memory
    Element *dev_A, *dev_B, *dev_C;
    cudaMalloc(&dev_A, sizeof(Element) * host_A.size());
    cudaMalloc(&dev_B, sizeof(Element) * host_B.size());
    cudaMalloc(&dev_C, sizeof(Element) * host_C.size());
    
    // Copy data to device
    cudaMemcpy(dev_A, host_A.data(), sizeof(Element) * host_A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B.data(), sizeof(Element) * host_B.size(), cudaMemcpyHostToDevice);

    // Define the GEMM type: C = alpha * A · B + beta * C
    using CutlassGemm = cutlass::gemm::device::Gemm<
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::RowMajor
    >;

    // Create GEMM arguments
    CutlassGemm::Arguments args(
        {M, N, K},        // problem size
        {dev_A, K},       // pointer to A and its leading dimension
        {dev_B, N},       // pointer to B and its leading dimension
        {dev_C, N},       // pointer to C and its leading dimension (input)
        {dev_C, N},       // pointer to C and its leading dimension (output)
        {1.0f, 0.0f});    // alpha and beta scalars

    // Launch the kernel
    CutlassGemm gemm_op;
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed\n";
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(host_C.data(), dev_C, sizeof(Element) * host_C.size(), cudaMemcpyDeviceToHost);
    std::cout << "Result element C[0] = " << host_C[0] << std::endl;

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    return 0;
}
