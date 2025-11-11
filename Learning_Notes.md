# GPU Custom Operations — Learning Notes (CUDA · CUTLASS · PyTorch)

> Short, practical notes to design, implement, and ship fast custom GPU ops. Keep it hands‑on, profile‑driven, and fused where it counts.

---

## 0) Big picture (what a “custom op” really is)

- **Math spec** → clear definition + small **reference** (NumPy/PyTorch) to check correctness.
- **Parallel algorithm & layout** → how to map the math onto threads/warps/blocks and memory.
- **Implementation** → CUDA kernel **or** a library like **CUTLASS** (for GEMM/Conv dialects).
- **Framework binding** → PyTorch extension, shape/dtype checks, stream handling, autograd.
- **Validation & performance** → correctness tests, micro‑benchmarks, Nsight profiling, iterate.

---

## 1) GPU mental model (minimal model to reason about speed)

- **SMs**: kernels launch a grid of blocks; each block runs on one SM.
- **Warps**: 32 threads lockstep (SIMT). Divergence serializes.
- **Latency hiding**: keep many warps ready (occupancy) to cover memory stalls.
- **Memory hierarchy** (fast → slow): registers → **shared/L1** → L2 → global (HBM/DRAM).
- Most ops are **bandwidth‑bound** → bytes matter as much as FLOPs.

---

## 2) CUDA essentials (stuff you use every time)

- **Thread indexing**: `blockIdx`, `threadIdx`, grid‑stride loops.
- **Coalesced access**: neighbors read neighbors. Lay out tensors accordingly.
- **Shared memory**: tile, reuse, avoid bank conflicts.
- **Sync**: `__syncthreads()` within block; prefer warp shuffles for small reductions.
- **Asynchrony**: streams to overlap copy/compute; SM80+ has `cp.async` for staged loads.

```cpp
// Grid‑stride skeleton (elementwise baseline)
__global__ void scale_kernel(float* y, const float* x, float a, int64_t n) {
  for (int64_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    y[i] = a * x[i];
  }
}
```

---

## 3) Roofline‑lite (know your bottleneck)

- **Arithmetic intensity** = FLOPs / Bytes moved (from global memory).
- Low intensity ⇒ bandwidth‑bound → **layout/tiling/vectorized loads/fusion**.
- High intensity ⇒ compute‑bound → **Tensor Cores / occupancy / unroll**.

---

## 4) Design workflow (from whiteboard to kernel)

1. **Math & reference**: tiny, correct PyTorch/NumPy function.
2. **Data layout**: NCHW vs NHWC; row/col major; contiguous vs strided. Decide early.
3. **Parallelization**: pick block/warp/thread tiles; ensure reuse within a block.
4. **Tiling pipeline**: global → shared (maybe with `cp.async`) → registers → MMA → epilogue writeback.
5. **Fuse epilogue**: bias, activation, scaling, clamp — **one write** to global memory.
6. **Numerics**: accumulation type, deterministic vs fast, eps for comparisons.
7. **Implement & profile**: measure achieved BW/FLOPs, load efficiency, occupancy, warp stalls.
8. **Iterate**: adjust tiles, vectorization width, staging, epilogue fusion.

```cpp
// Conceptual tiled matmul (teaching sketch; production uses MMA/Tensor Cores)
template<int BM, int BN, int BK>
__global__ void matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K) {
  __shared__ float As[BM][BK], Bs[BK][BN];
  int row = blockIdx.y * BM + threadIdx.y;
  int col = blockIdx.x * BN + threadIdx.x;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += BK) {
    if (row < M && k0 + threadIdx.x < K) As[threadIdx.y][threadIdx.x] = A[row*K + (k0 + threadIdx.x)];
    if (col < N && k0 + threadIdx.y < K) Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y)*N + col];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row*N + col] = acc; // ← fuse bias/activation here
}
```

---

## 5) CUTLASS in practice (when GEMM/Conv is your op)

- **What it gives**: proven tiling, async staging (`cp.async`), Tensor Core MMA, fused epilogues.
- **When to use**: if your op *is* GEMM/Conv or can be lowered to it (linear layers, 1×1 conv, attention blocks).
- **When to hand‑write CUDA**: elementwise, stencil, custom reductions, irregular memory patterns.
- **Epilogue fusion**: choose epilogues like `LinearCombinationRelu` to save a round trip to memory.

*Rule of thumb*: If you see `A @ B (+ bias) (+ activation)`, CUTLASS is a strong default.

---

## 6) PyTorch integration (extension & streams)

- **Binding** (C++): validate dtype/device/layout, allocate outputs, launch on **current stream**:
  `at::cuda::getCurrentCUDAStream()`.
- **API**: small, explicit: e.g., `cutlass_ext.linear(A, B, bias=None, relu=False)`.
- **Autograd**: start with Python `torch.autograd.Function` for clarity, then move backward to CUDA/CUTLASS.

```cpp
// Minimal CUDA‑backed PyTorch op pattern
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor myop_forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous());
  auto y = torch::empty_like(x);
  // launch <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>
  return y;
}

TORCH_LIBRARY(myops, m) { m.def("forward(Tensor x) -> Tensor"); }
TORCH_LIBRARY_IMPL(myops, CUDA, m) { m.impl("forward", myop_forward); }
```

---

## 7) Autograd strategy (making it differentiable)

- **Python autograd first**: wrap forward in `torch.autograd.Function` and use known formulas in Python for `backward` (great for correctness bring‑up).
- **Custom backward kernels** later for speed:
  - For GEMM‑like ops: `dA = dC @ B^T`, `dB = A^T @ dC` (watch transposes & layouts).
  - Consider fusing grad epilogues if bandwidth‑bound.
- **Numerics**: accumulate in FP32 for FP16/BF16 inputs; be mindful of determinism.

---

## 8) Streams, events, and synchronization

- Launch on **PyTorch’s current stream**; avoid global `cudaDeviceSynchronize()`.
- Use **events** if you must order work across streams.
- When calling cuBLAS/CUTLASS, pass the **same stream** to avoid hidden syncs.

---

## 9) Testing & benchmarking (tight feedback loops)

- **Correctness**: compare against reference: `torch.allclose(out, ref, rtol, atol)` across shapes/dtypes/layouts.
- **Micro‑benchmarks**: warmup → repeat → sync → time. Record shapes that matter to your app.
- **Nsight Systems**: look for gaps/overlap; **Nsight Compute**: load/store efficiency, occupancy, warp stall reasons.
  - Low L2 hit or many replays → reconsider tiling & coalescing.
  - Low occupancy (reg/SMEM bound) → adjust tile sizes, unroll, vectorization.

---

## 10) Pattern playbook (common structures)

- **Elementwise + reduction**: grid‑stride over elems, warp‑shuffle reduce to block partials, atomicAdd to global.
- **Stencil**: shared‑memory tiles with halos; minimize redundant global loads.
- **GEMM/Conv**: 3‑level tiling (block/warp/thread), async staging, Tensor Cores, **fused epilogue**.

---

## 11) Fusion strategy (where perf often comes from)

- Push all post‑ops (bias, activation, clamp, scaling) into the **epilogue**.
- Goal: **read once, write once** to global memory.
- Usually outperforms micro‑tweaks in mainloop when memory‑bound.

---

## 12) Multi‑precision & Tensor Cores

- FP16/BF16 inputs with **FP32 accumulation** → Tensor Cores (SM70+).
- TF32 (Ampere) accelerates FP32 GEMM with small precision tradeoff.
- Choose precision based on accuracy budget; consider loss scaling for FP16 training.

---

## 13) Shipping & maintenance (make it easy to use again)

- **Stable API** (clear shapes/dtypes/broadcast rules).
- **Helpful errors** (turn CUDA/CUTLASS statuses into actionable messages).
- **Portability** (compile `-gencode` for your SMs; test on target GPUs).
- **Packaging**: JIT via `torch.utils.cpp_extension.load` or a `setup.py` wheel.
- **CI**: correctness + smoke perf tests; CPU‑only fallback where possible.

---

## 14) My study plan (quick personal checklist)

- [ ] Write tiny **reference** implementations for each new op.
- [ ] Practice **layout decisions** and measure coalescing effects.
- [ ] Build one **CUDA elementwise**, one **CUDA reduction**, one **tiled matmul**.
- [ ] Re‑implement the matmul with **CUTLASS**, add **fused epilogue**.
- [ ] Wrap both in **PyTorch**; add a **Python autograd backward** first.
- [ ] Profile with Nsight; adjust tiles, vectorization, staging, fusion.
- [ ] Upgrade to **FP16/BF16 Tensor Cores**; compare accuracy/speed.
- [ ] Replace Python backward with **custom CUDA/CUTLASS** backward.
- [ ] Document results and lessons learned per shape/dtype.

---

## 15) Takeaways

- Most wins = **moving fewer bytes**: better layout, more reuse, **fused epilogues**.
- **CUTLASS** is the fastest path for GEMM/Conv‑like ops; write CUDA for the rest.
- Respect **streams/autograd/layouts** when binding to PyTorch.
- Always **measure**; let profiles, not vibes, drive the next optimization.
