# RPI‑UGRA – GPU Customized Operation

This repository tracks my undergraduate research journey into **GPU‑accelerated computation** and **custom CUDA operations**.  It began with simple CUDA C/C++ programs and has progressed into exploring higher‑level libraries such as NVIDIA CUTLASS.  The goal is to move from **basic CUDA programming concepts** to **implementing custom GPU kernels** that can later be integrated into higher‑level frameworks (e.g., PyTorch custom ops, CUTLASS‑style kernels, etc.) and eventually contribute novel operations.

---

## Repository Structure

> (Folder and filenames here assume the current layout with CUDA and CUTLASS demos; adjust if you rename things.)

- `cuda_demos/` – step‑by‑step CUDA learning demos (`demo01_*.cpp` … `demo10_*.cpp`)
- `cutlass_demos/` – C++ demos illustrating how to use the NVIDIA **CUTLASS** library.  These build upon the CUDA basics to show how matrix multiply‑accumulate and related operations are implemented at a higher level.
- (future) `custom_ops/` – more advanced, research‑oriented custom kernels
- (future) `notes/` – PDF/markdown notes, experiment logs, and design sketches

---

## Learning Path & CUDA Demos

The main CUDA learning track lives in `cuda_demos/`.  Each demo is intentionally small and focused on **one idea at a time**.

| Demo | File | Concept |
| --- | --- | --- |
| 1 | `demo01_hello_gpu.cpp` | Launching a kernel & printing from GPU threads |
| 2 | `demo02_vector_add.cpp` | 1D grid, thread indexing, vector addition |
| 3 | `demo03_matrix_add_2d.cpp` | 2D grids/blocks, row–column indexing |
| 4 | `demo04_matmul_naive.cpp` | Naive matrix multiplication on GPU |
| 5 | `demo05_matmul_tiled.cpp` | Shared memory tiling for matmul |
| 6 | `demo06_pinned_memory.cpp` | Pinned vs pageable host memory & transfer timing |
| 7 | `demo07_streams_vector_add.cpp` | CUDA streams & overlapping copy/compute |
| 8 | `demo08_constant_memory_scale.cpp` | Using constant memory (`__constant__`) |
| 9 | `demo09_unified_memory.cpp` | Unified / managed memory + prefetch |
| 10 | `demo10_atomic_sum.cpp` | Atomic operations and parallel reduction |

You can read the demos in order to get a **narrative** of CUDA:

1. Start from “hello world” and a single simple kernel.
2. Learn how threads and blocks map to data structures (vectors, matrices).
3. Explore the **memory hierarchy** (global, shared, pinned, unified, constant).
4. Introduce **asynchrony** and **streams** to overlap transfers with compute.
5. Finish with **atomic operations** and reductions, which appear in many real workloads.

---

## CUTLASS Learning Demos

Parallel to the CUDA basics, I began exploring the **CUTLASS (CUDA Templates for Linear Algebra Subroutines)** library.  CUTLASS decomposes matrix multiplication into reusable components and supports mixed‑precision, fused epilogues, and advanced scheduling【228626371764721†L50-L66】.  The `cutlass_demos/` directory contains a series of ten C++ programs that demonstrate progressively more sophisticated uses of CUTLASS.  These demos are written in plain C++ with CUDA extensions and assume that CUTLASS is installed on your system.

| Demo | File | Concept |
| --- | --- | --- |
| 1 | `demo01_cutlass_gemm_fp32.cpp` | Basic FP32 GEMM using CUTLASS |
| 2 | `demo02_cutlass_gemm_fp16_tensor_core.cpp` | Half‑precision GEMM on Tensor Cores (FP16×FP16→FP32) |
| 3 | `demo03_cutlass_gemm_int8.cpp` | Quantized INT8 GEMM with INT32 accumulation |
| 4 | `demo04_cutlass_batched_gemm.cpp` | Batched GEMM – executing multiple matrix multiplications in a loop |
| 5 | `demo05_cutlass_conv2d.cpp` | 2‑D convolution via CUTLASS’s implicit GEMM convolution |
| 6 | `demo06_cutlass_fused_bias_relu.cpp` | Fusing bias addition and ReLU activation into the GEMM epilogue |
| 7 | `demo07_cutlass_complex_gemm.cpp` | Complex number GEMM (`std::complex<float>`) |
| 8 | `demo08_cutlass_async_gemm.cpp` | GEMM using CUTLASS 3.x kernels with asynchronous `cp.async` copies |
| 9 | `demo09_cutlass_grouped_gemm.cpp` | Grouped GEMM for executing multiple problems of different sizes in one launch |
| 10 | `demo10_cutlass_linear_softmax.cpp` | Linear layer with bias and a row‑wise softmax function |

To build the CUTLASS demos, configure your project similarly to the CUDA demos but link against the CUTLASS headers and ensure your `CUDACXX` environment variable points to `nvcc`.  Each demo in this directory illustrates a new feature and provides comments explaining the key concepts.

---

## How to Build & Run (Visual Studio + CUDA/CUTLASS on Windows)

These demos are written as `.cpp` files but use CUDA extensions (`__global__`, `<<< >>>`, etc.), so **they must be compiled as CUDA C/C++**, not plain MSVC C++.  CUTLASS is header‑only and requires C++17.

1. Install **NVIDIA CUDA Toolkit** (see links in the “Learning Resources” section) and clone the [CUTLASS repository](https://github.com/NVIDIA/cutlass) into a location referenced by your include path.
2. In Visual Studio:
   1. `File → New → Project → CUDA 12.x Runtime` (or similar template).
   2. Add the `cuda_demos/*.cpp` and `cutlass_demos/*.cpp` files to the project.
   3. Set each file’s **Item Type** to **`CUDA C/C++`** under *Properties → General*.  Choose a demo as the **Startup Item**.
3. Ensure your project’s **Additional Include Directories** include the path to the CUTLASS headers (e.g., `cutlass/include`).
4. Build and run.  For example, running `demo02_cutlass_gemm_fp16_tensor_core.cpp` should print the first element of the result matrix and demonstrate the speed of Tensor Cores.

If you prefer CMake, this repo can also be turned into a CUDA + CUTLASS CMake project (`project(... LANGUAGES CXX CUDA)`).  See the `examples/` in the CUTLASS repository for inspiration.

---

## Conceptual Roadmap / Process

This repo is meant to document a **learning + research pipeline**, not just a pile of code.

### Phase 1 – Get Comfortable with CUDA Basics
- Understand how a **kernel launch** maps to hardware:
  - Threads, blocks, and grids.
  - Using `blockIdx`, `threadIdx`, `blockDim` to compute global indices.
- Practice with simple compute patterns:
  - Elementwise ops (vector add, scaling).
  - 2‑D indexing (matrix add).

### Phase 2 – Respect the GPU Memory Hierarchy
- Explore **global memory** vs **shared memory**:
  - Naive matmul (`demo04`) vs shared‑memory tiled matmul (`demo05`).
- Experiment with host–device transfers:
  - Compare pageable vs **pinned host memory** in `demo06_pinned_memory.cpp`.
- Get a feel for **latency hiding**:
  - Use **streams** (`demo07`) to overlap copies and compute.
  - See how large problem sizes benefit more from asynchrony.

### Phase 3 – Advanced CUDA Features
- **Constant memory** (`demo08`) for small read‑only parameters shared by all threads.
- **Unified (managed) memory** (`demo09`) to simplify allocations and gradually learn about page migration and prefetching.
- **Atomic operations** & reductions (`demo10`):
  - Build intuition for race conditions and when atomics are necessary.
  - Use shared memory for intra‑block reduction + `atomicAdd()` for global accumulation.

### Phase 4 – Towards Custom GPU Operations
Once the core CUDA concepts are solid:

- Start designing **problem‑specific kernels**:
  - E.g., custom stencil operations, fused elementwise ops, simple convolutions, etc.
- Integrate with **host libraries / frameworks**:
  - Wrap kernels in C++ APIs that can later be called from Python or integrated into frameworks (e.g., PyTorch custom ops).
- Benchmark and profile:
  - Use Nsight tools and CUDA events to measure kernel runtime, occupancy, and bandwidth utilization.

### Phase 5 – High‑Level Libraries and CUTLASS
After understanding low‑level CUDA, the next phase explores libraries built on top of CUDA that encapsulate best practices.

- Learn about **CUTLASS** and how it decomposes GEMM into configurable components【228626371764721†L50-L66】.
- Implement GEMM, convolution, and fused operations using CUTLASS.
- Compare CUTLASS kernels against your own implementations to understand the trade‑offs (e.g., tile sizes, Tensor Cores).
- Prototype deep‑learning layers (e.g., linear + softmax) to see how CUTLASS fits into larger GPU workloads.

---

## Learning Resources

These are core references used while building the demos and understanding the hardware/software stack:

- **Basics on NVIDIA GPU Hardware Architecture – NASA HECC**  – A very clear introduction to NVIDIA GPU hardware, including the GPU vs CPU mental model, streaming multiprocessors, tensor cores, thread/block/grid structures and the memory hierarchy.  <https://www.nas.nasa.gov/hecc/support/kb/basics-on-nvidia-gpu-hardware-architecture_704.html>
- **NVIDIA CUDA Toolkit (official)** – The official toolkit and documentation: compiler (`nvcc`), core libraries (cuBLAS, cuDNN, etc.), debugging & profiling tools, and the reference documentation for CUDA C/C++.  Use this for installation, version checks and reading the Programming Guide and Best Practices Guide.  <https://developer.nvidia.com/cuda-toolkit>
- **CUTLASS technical blog** – Introduces CUTLASS, its decomposition of GEMM and support for mixed precision.  It explains that CUTLASS can fuse element‑wise operations (e.g., activation functions) into GEMM【228626371764721†L50-L66】 and provides example code for bias + ReLU epilogues【228626371764721†L687-L776】.  <https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/>
- (future) Additional links:
  - CUDA Programming Guide sections or blog posts that were useful.
  - Papers or notes related to the custom operation you end up implementing.

---

## Future Work / TODO

- [x] Complete the initial CUDA learning demos (`demo01`–`demo10`).
- [x] Create a set of **CUTLASS demos** showcasing GEMM, convolution, fused epilogues, complex numbers, grouped problems and a simple neural‑network layer.
- [ ] Add a `CMakeLists.txt` for easy cross‑platform builds of both CUDA and CUTLASS demos.
- [ ] Add profiling scripts / instructions (Nsight Systems / Nsight Compute) to compare naive CUDA kernels with CUTLASS kernels.
- [ ] Implement the first “real” custom op kernel (e.g., fused activation, small matmul, or custom reduction) and document the design decisions.
- [ ] Connect kernels to a higher‑level framework (e.g., PyTorch custom op) and benchmark end‑to‑end performance.
- [ ] Expand the notes directory with experiment logs and design sketches.

---

## License

This project is currently for **research and educational purposes** as part of an undergraduate research project at RPI.  Choose a license (e.g., MIT) once you are ready to share code more broadly.
