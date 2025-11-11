// demo07_cutlass_complex_gemm.cpp
#include <cutlass/gemm/device/gemm_complex.h>
#include <cutlass/complex.h>
#include <iostream>

int main() {
    using Complex = cutlass::complex<float>;
    const int M = 64, N = 64, K = 32;

    std::vector<Complex> hA(M*K), hB(K*N), hC(M*N);
    for (int i = 0; i < M*K; ++i) hA[i] = Complex(1.0f, 0.5f);
    for (int i = 0; i < K*N; ++i) hB[i] = Complex(0.25f, -0.75f);

    Complex *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(Complex)*hA.size());
    cudaMalloc(&dB, sizeof(Complex)*hB.size());
    cudaMalloc(&dC, sizeof(Complex)*hC.size());
    cudaMemcpy(dA, hA.data(), sizeof(Complex)*hA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeof(Complex)*hB.size(), cudaMemcpyHostToDevice);

    using GemmComplex = cutlass::gemm::device::GemmComplex<
        Complex, cutlass::layout::RowMajor,
        Complex, cutlass::layout::RowMajor,
        Complex, cutlass::layout::RowMajor
    >;

    GemmComplex gemm;
    GemmComplex::Arguments args({M,N,K}, {dA, K}, {dB, N}, {dC, N}, {dC, N}, {Complex(1.0f), Complex(0.0f)});
    gemm(args);
    cudaMemcpy(hC.data(), dC, sizeof(Complex)*hC.size(), cudaMemcpyDeviceToHost);
    std::cout << "Complex GEMM C[0] = (" << hC[0].real() << ", " << hC[0].imag() << ")\n";
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
