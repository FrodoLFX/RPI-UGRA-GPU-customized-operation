# Advanced Demo: CUTLASS + CUDA + PyTorch Custom Op

This demo shows how to **use CUTLASS from a PyTorch extension**. We expose a `cutlass_linear(A, B, bias=None, relu=False)`
function that computes `C = A @ B (+ bias) (then ReLU)` using **CUTLASS GEMM** under the hood.

> **Why this is cool**
> - You write high-level training code in **PyTorch**.
> - Your hot GEMM path runs through **CUTLASS** (CUDA C++ templates) for performance.
> - You can fuse bias/activation in Python today or move fusion into CUTLASS epilogues later.

## Layout

```
cutlass_cuda_pytorch_demo/
├─ build_and_test.py          # JIT-compiles the extension and benchmarks vs torch.mm / nn.Linear
├─ ext/
│  ├─ cutlass_linear.cpp      # PyTorch binding & safety checks
│  └─ cutlass_linear_kernel.cu# CUTLASS GEMM launcher (FP32 RowMajor)
└─ scripts/
   └─ quick_bench.py          # Small benchmark harness you can customize
```

## Requirements

- CUDA-capable NVIDIA GPU
- CUDA Toolkit installed
- **PyTorch** (matching your CUDA)
- **CUTLASS** headers (set `CUTLASS_DIR` env var to the CUTLASS repo root)
  - e.g.
    ```bash
    git clone https://github.com/NVIDIA/cutlass.git
    export CUTLASS_DIR=/path/to/cutlass
    ```

## Build & Run (JIT)

```bash
# 1) Set CUTLASS_DIR to your local clone
export CUTLASS_DIR=/path/to/cutlass

# 2) Create and activate a Python env, then:
pip install torch torchvision torchaudio  # pick proper CUDA/CPU build from pytorch.org
pip install numpy

# 3) JIT-build and test
python build_and_test.py
```

This script:
- compiles the extension with `torch.utils.cpp_extension.load`
- runs a quick correctness check
- benchmarks vs PyTorch `torch.mm` and `nn.Linear`

## Notes

- The kernel here targets **FP32 RowMajor** for robustness across CUTLASS versions.
- To use **Tensor Cores** with FP16/BF16, switch the CUTLASS GEMM template to those types and set `-gencode` for your SM arch.
- To fuse bias/activation inside CUTLASS, use an epilogue like `LinearCombinationRelu` (API names vary by CUTLASS version).
