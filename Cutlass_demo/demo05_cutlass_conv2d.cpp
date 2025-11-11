// demo05_cutlass_conv2d.cpp
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/layout/tensor.h>
#include <iostream>

int main() {
    using Element = float;
    using LayoutInput = cutlass::layout::TensorNHWC;
    using LayoutFilter = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    // Problem: one image 32×32×1, one filter 3×3×1→1
    int N = 1, H = 32, W = 32, C = 1, K = 1; // N=batch, C=in_channels, K=out_channels
    int R = 3, S = 3; // filter dimensions
    int pad = 1;

    // Host tensors (NHWC)
    std::vector<Element> hInput(N * H * W * C, 1.0f);
    std::vector<Element> hFilter(K * R * S * C, 1.0f);
    std::vector<Element> hOutput(N * H * W * K);

    // Device tensors
    Element *dInput, *dFilter, *dOutput;
    cudaMalloc(&dInput, sizeof(Element)*hInput.size());
    cudaMalloc(&dFilter, sizeof(Element)*hFilter.size());
    cudaMalloc(&dOutput, sizeof(Element)*hOutput.size());
    cudaMemcpy(dInput, hInput.data(), sizeof(Element)*hInput.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dFilter, hFilter.data(), sizeof(Element)*hFilter.size(), cudaMemcpyHostToDevice);

    using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<
        Element, LayoutInput,
        Element, LayoutFilter,
        Element, LayoutOutput,
        Element,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64,64,16>,
        cutlass::gemm::GemmShape<32,32,16>,
        cutlass::gemm::GemmShape<1,1,1>
    >;

    Conv2dFprop conv;
    typename Conv2dFprop::Arguments args(
        {N, H, W, C}, {K, R, S, C}, {N, H, W, K},
        {1, 1}, {pad, pad}, {1, 1},
        dInput, dFilter, dOutput);

    cutlass::Status status = conv(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Convolution failed\n";
    }

    cudaMemcpy(hOutput.data(), dOutput, sizeof(Element)*hOutput.size(), cudaMemcpyDeviceToHost);
    std::cout << "Conv output[0] = " << hOutput[0] << std::endl;

    cudaFree(dInput); cudaFree(dFilter); cudaFree(dOutput);
    return 0;
}
