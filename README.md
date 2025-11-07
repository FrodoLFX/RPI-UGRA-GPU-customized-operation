# RPI-UGRA – GPU Customized Operation

This repository tracks my undergraduate research journey into **GPU-accelerated computation** and **custom CUDA operations**.  

The goal is to move from **basic CUDA programming concepts** to **implementing custom GPU kernels** that can later be integrated into higher-level frameworks (e.g., PyTorch custom ops, CUTLASS-style kernels, etc.).

---

## Repository Structure

> (Folder and filenames here assume the current layout with CUDA demos;
> adjust if you rename things.)

- `cuda_demos/` – step-by-step CUDA learning demos (`demo01_*.cpp` … `demo10_*.cpp`)
- (future) `custom_ops/` – more advanced, research-oriented custom kernels
- (future) `notes/` – PDF/markdown notes, experiment logs, and design sketches

---

## Learning Path & Demos

The main learning track lives in `cuda_demos/`.  
Each demo is intentionally small and focused on **one idea at a time**.

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

## How to Build & Run (Visual Studio + CUDA on Windows)

These demos are written as `.cpp` files but use CUDA extensions (`__global__`, `<<< >>>`, etc.), so **they must be compiled as CUDA C/C++**, not plain MSVC C++.

### 1. Create a CUDA Runtime Project

1. Install **NVIDIA CUDA Toolkit** (see links in the “Learning Resources” section).
2. In Visual Studio:  
   `File → New → Project → CUDA 12.x Runtime` (or similar template).
3. Add the `cuda_demos/*.cpp` files to the project.

### 2. Mark Files as CUDA C/C++

For each demo file:

1. Right-click the file → **Properties**.
2. Under **General → Item Type**, set to **`CUDA C/C++`**.
3. Choose one demo as the **Startup Item** (e.g., `demo02_vector_add.cpp`).

### 3. Run a Demo

Example (running the vector add demo):

1. Set `demo02_vector_add.cpp` as startup.
2. Build the project (Ctrl+Shift+B).
3. Run (F5).  
   You should see output confirming the result is `OK`.

> If you prefer CMake, this repo can also be turned into a CUDA CMake project later (`project(... LANGUAGES CXX CUDA)` etc.).

---

## Conceptual Roadmap / Process

This repo is meant to document a **learning + research pipeline**, not just a pile of code.

### Phase 1 – Get Comfortable with CUDA Basics

- Understand how a **kernel launch** maps to hardware:
  - Threads, blocks, and grids.
  - Using `blockIdx`, `threadIdx`, `blockDim` to compute global indices.
- Practice with simple compute patterns:
  - Elementwise ops (vector add, scaling).
  - 2D indexing (matrix add).

### Phase 2 – Respect the GPU Memory Hierarchy

- Explore **global memory** vs **shared memory**:
  - Naive matmul (`demo04`) vs shared-memory tiled matmul (`demo05`).
- Experiment with host–device transfers:
  - Compare pageable vs **pinned host memory** in `demo06_pinned_memory.cpp`.
- Get a feel for **latency hiding**:
  - Use **streams** (`demo07`) to overlap copies and compute.
  - See how large problem sizes benefit more from asynchrony.

### Phase 3 – Advanced Features

- **Constant memory** (`demo08`) for small read-only parameters shared by all threads.
- **Unified (managed) memory** (`demo09`) to simplify allocations and gradually learn about page migration and prefetching.
- **Atomic operations** & reductions (`demo10`):
  - Build intuition for race conditions and when atomics are necessary.
  - Use shared memory for intra-block reduction + atomicAdd() for global accumulation.

### Phase 4 – Towards Custom GPU Operations

Once the core CUDA concepts are solid:

- Start designing **problem-specific kernels**:
  - E.g., custom stencil operations, fused elementwise ops, simple convolutions, etc.
- Integrate with **host libraries / frameworks**:
  - Wrap kernels in C++ APIs that can later be called from Python or integrated into frameworks (e.g., PyTorch custom ops).
- Benchmark and profile:
  - Use Nsight tools and CUDA events to measure kernel runtime, occupancy, and bandwidth utilization.

The long-term plan is for this repo to evolve from “learning demos” into a **playground for experimental custom operations**, with a trail of notes explaining what worked and what didn’t.

---

## Learning Resources

These are core references used while building the demos and understanding the hardware/software stack:

- **Basics on NVIDIA GPU Hardware Architecture – NASA HECC**  
  A very clear introduction to NVIDIA GPU hardware, including:
  - GPU vs CPU mental model
  - Streaming Multiprocessors (SMs), CUDA cores, Tensor cores
  - Threads, warps, blocks, and grids
  - The GPU memory hierarchy (registers, shared/L1, L2, global, constant, etc.)  
  <https://www.nas.nasa.gov/hecc/support/kb/basics-on-nvidia-gpu-hardware-architecture_704.html>

- **NVIDIA CUDA Toolkit (official)**  
  The official toolkit and docs: compiler (`nvcc`), core libraries (cuBLAS, cuDNN, etc.), debugging & profiling tools, and the reference documentation for CUDA C/C++.  
  Use this for:
  - Installation on Windows/Linux
  - Checking the right version for your GPU and driver
  - Reading the Programming Guide and Best Practices Guide  
  <https://developer.nvidia.com/cuda-toolkit>

- (future) Additional links:
  - CUDA Programming Guide sections or blog posts that were useful.
  - Papers or notes related to the custom operation you end up implementing.

---

## Future Work / TODO

- [ ] Add a `CMakeLists.txt` for easy cross-platform builds.
- [ ] Add profiling scripts / instructions (Nsight Systems / Nsight Compute).
- [ ] Implement first “real” custom op kernel (e.g., fused activation, small matmul, or custom reduction) and document the design decisions.
- [ ] Connect kernels to a higher-level framework (e.g., PyTorch custom op).

---

## License

This project is currently for **research and educational purposes** as part of an undergraduate research project at RPI.  
Choose a license (e.g., MIT) once you are ready to share code more broadly.
