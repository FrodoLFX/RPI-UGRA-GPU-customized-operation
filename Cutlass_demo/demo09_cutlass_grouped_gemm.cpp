// demo09_cutlass_grouped_gemm.cpp
#include <cutlass/gemm/device/grouped_gemm.h>
#include <iostream>

int main() {
    using T = float;
    // Define two problems: C0 = A0×B0 and C1 = A1×B1 with different shapes
    cutlass::gemm::GemmCoord problem0(64, 128, 32);
    cutlass::gemm::GemmCoord problem1(128, 32, 64);

    // Allocate host and device memory for both groups
    std::vector<T> hA0(problem0.m() * problem0.k(), 1.0f);
    std::vector<T> hB0(problem0.k() * problem0.n(), 2.0f);
    std::vector<T> hC0(problem0.m() * problem0.n(), 0.0f);
    std::vector<T> hA1(problem1.m() * problem1.k(), 0.5f);
    std::vector<T> hB1(problem1.k() * problem1.n(), -1.0f);
    std::vector<T> hC1(problem1.m() * problem1.n(), 0.0f);

    // Device pointers for group 0
    T *dA0, *dB0, *dC0;
    cudaMalloc(&dA0, sizeof(T)*hA0.size());
    cudaMalloc(&dB0, sizeof(T)*hB0.size());
    cudaMalloc(&dC0, sizeof(T)*hC0.size());
    cudaMemcpy(dA0, hA0.data(), sizeof(T)*hA0.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB0, hB0.data(), sizeof(T)*hB0.size(), cudaMemcpyHostToDevice);

    // Device pointers for group 1
    T *dA1, *dB1, *dC1;
    cudaMalloc(&dA1, sizeof(T)*hA1.size());
    cudaMalloc(&dB1, sizeof(T)*hB1.size());
    cudaMalloc(&dC1, sizeof(T)*hC1.size());
    cudaMemcpy(dA1, hA1.data(), sizeof(T)*hA1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB1, hB1.data(), sizeof(T)*hB1.size(), cudaMemcpyHostToDevice);

    using GroupedGemm = cutlass::gemm::device::GroupedGemm<
        T, cutlass::layout::RowMajor,
        T, cutlass::layout::RowMajor,
        T, cutlass::layout::RowMajor
    >;

    GroupedGemm groupedGemm;
    // Prepare problem descriptors
    std::vector<GroupedGemm::Arguments> group_args;
    group_args.emplace_back(problem0, {dA0, problem0.k()}, {dB0, problem0.n()}, {dC0, problem0.n()}, {dC0, problem0.n()}, {1.0f, 0.0f});
    group_args.emplace_back(problem1, {dA1, problem1.k()}, {dB1, problem1.n()}, {dC1, problem1.n()}, {dC1, problem1.n()}, {1.0f, 0.0f});

    auto status = groupedGemm(group_args.data(), group_args.size());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Grouped GEMM failed\n";
    }

    cudaMemcpy(hC0.data(), dC0, sizeof(T)*hC0.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC1.data(), dC1, sizeof(T)*hC1.size(), cudaMemcpyDeviceToHost);
    std::cout << "Group0 C[0] = " << hC0[0] << ", Group1 C[0] = " << hC1[0] << std::endl;

    cudaFree(dA0); cudaFree(dB0); cudaFree(dC0);
    cudaFree(dA1); cudaFree(dB1); cudaFree(dC1);
    return 0;
}
