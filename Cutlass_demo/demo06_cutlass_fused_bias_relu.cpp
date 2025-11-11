// demo06_cutlass_fused_bias_relu.cpp
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/numeric_types.h>
#include <iostream>

int main() {
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementOutput = float;

    const int M = 128, N = 256, K = 64;

    // Host inputs
    std::vector<ElementAccumulator> hA(M*K, 1.0f);
    std::vector<ElementAccumulator> hB(K*N, 0.5f);
    std::vector<ElementOutput> hC(M*N, 0.0f);
    std::vector<ElementOutput> hBias(N, 0.1f);

    // Device allocations
    ElementAccumulator *dA, *dB;
    ElementOutput *dC, *dBias;
    cudaMalloc(&dA, sizeof(ElementAccumulator)*hA.size());
    cudaMalloc(&dB, sizeof(ElementAccumulator)*hB.size());
    cudaMalloc(&dC, sizeof(ElementOutput)*hC.size());
    cudaMalloc(&dBias, sizeof(ElementOutput)*hBias.size());
    cudaMemcpy(dA, hA.data(), sizeof(ElementAccumulator)*hA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeof(ElementAccumulator)*hB.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dBias, hBias.data(), sizeof(ElementOutput)*hBias.size(), cudaMemcpyHostToDevice);

    // Define a linear combination with ReLU epilogue: D = ReLU(alpha * accum + beta * C + bias)
    using Epilogue = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementOutput,                      // output element type
        1,                                  // number of elements per output
        ElementAccumulator,                 // accumulator type
        ElementCompute>;                    // compute type

    using Gemm = cutlass::gemm::device::Gemm<
        ElementAccumulator, cutlass::layout::RowMajor,
        ElementAccumulator, cutlass::layout::RowMajor,
        ElementOutput, cutlass::layout::RowMajor,
        ElementCompute,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,128,32>,
        cutlass::gemm::GemmShape<64,64,32>,
        cutlass::gemm::GemmShape<16,16,16>,
        Epilogue
    >;

    Gemm gemm;
    typename Gemm::Arguments args(
        {M, N, K}, {dA, K}, {dB, N}, {dC, N}, {dC, N},
        {1.0f, 0.0f});
    args.epilogue.epilogue_op.params.bias = dBias; // pass bias pointer
    auto status = gemm(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Fused GEMM failed\n";
    }

    cudaMemcpy(hC.data(), dC, sizeof(ElementOutput)*hC.size(), cudaMemcpyDeviceToHost);
    std::cout << "Fused Bias+ReLU result C[0] = " << hC[0] << std::endl;

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dBias);
    return 0;
}
