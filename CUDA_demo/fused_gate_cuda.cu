#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_gate_kernel(const float* __restrict__ x,
    const float* __restrict__ v,
    float* __restrict__ y,
    int M, int N,
    float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int j = idx % N;
    float gj = v[j];
    float out = (fabsf(gj) >= threshold) ? (x[idx] * gj) : 0.0f;
    y[idx] = out;
}

at::Tensor fused_gate_forward_cuda(at::Tensor x, at::Tensor v, double threshold) {
    const int M = (int)x.size(0);
    const int N = (int)x.size(1);

    auto y = at::empty_like(x);

    const int threads = 256;
    const int blocks = (M * N + threads - 1) / threads;

    fused_gate_kernel << <blocks, threads >> > (
        x.data_ptr<float>(),
        v.data_ptr<float>(),
        y.data_ptr<float>(),
        M, N, (float)threshold
        );

    return y;
}
