#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef TILE
#define TILE 16
#endif

// gate kernel (baseline step 2) 
__global__ void gate_kernel(const float* __restrict__ C,
    const float* __restrict__ v,
    float* __restrict__ Out,
    int M, int N, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int j = idx % N;
    float gj = v[j];
    Out[idx] = (fabsf(gj) >= threshold) ? (C[idx] * gj) : 0.0f;
}

at::Tensor gate_forward_cuda(at::Tensor C, at::Tensor v, double threshold) {
    int M = (int)C.size(0), N = (int)C.size(1);
    auto Out = at::empty_like(C);

    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;

    gate_kernel << <blocks, threads >> > (C.data_ptr<float>(), v.data_ptr<float>(),
        Out.data_ptr<float>(), M, N, (float)threshold);
    return Out;
}

//tiled matmul (baseline step 1)
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

at::Tensor matmul_forward_cuda(at::Tensor A, at::Tensor B) {
    int M = (int)A.size(0), K = (int)A.size(1), N = (int)B.size(1);
    auto C = at::empty({ M, N }, A.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_tiled_kernel << <grid, block >> > (A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

// fused matmul + gate (CATS-style)
__global__ void fused_matmul_gate_kernel(const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ v,
    float* __restrict__ Out,
    int M, int N, int K,
    float threshold) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float gj = v[col];
        Out[row * N + col] = (fabsf(gj) >= threshold) ? (acc * gj) : 0.0f;
    }
}

at::Tensor fused_forward_cuda(at::Tensor A, at::Tensor B, at::Tensor v, double threshold) {
    int M = (int)A.size(0), K = (int)A.size(1), N = (int)B.size(1);
    auto Out = at::empty({ M, N }, A.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    fused_matmul_gate_kernel << <grid, block >> > (
        A.data_ptr<float>(), B.data_ptr<float>(), v.data_ptr<float>(), Out.data_ptr<float>(),
        M, N, K, (float)threshold
        );
    return Out;
}
