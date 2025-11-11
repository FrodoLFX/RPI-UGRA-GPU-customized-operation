# RPI‑UGRA – GPU Customized Operation

This repository tracks my undergraduate research journey into **GPU‑accelerated computation** and **custom CUDA operations**. It began with simple CUDA C/C++ programs and progressed into exploring higher‑level libraries such as **NVIDIA CUTLASS**. **Now it also includes a full PyTorch learning track** that mirrors the same easy→hard progression and prepares for framework‑level research (custom ops, deployment, benchmarking).

---

## Repository Structure

> (Folder and filenames assume the current layout with CUDA, CUTLASS, and PyTorch demos; adjust if you rename things.)

- `cuda_demos/` – step‑by‑step CUDA learning demos (`demo01_*.cpp` … `demo10_*.cpp`)
- `cutlass_demos/` – C++ demos illustrating how to use **CUTLASS** (GEMM FP32/FP16/INT8, grouped GEMM, fused epilogues, conv2d, etc.)
- `pytorch_demos/` – **NEW**: ten PyTorch demos from tensors & autograd to CNNs, transfer learning, custom autograd, and TorchScript
- (future) `custom_ops/` – more advanced, research‑oriented custom kernels and PyTorch bindings
- (future) `notes/` – PDF/markdown notes, experiment logs, and design sketches

---

## Learning Path & CUDA Demos

The main CUDA learning track lives in `cuda_demos/`. Each demo is intentionally small and focused on **one idea at a time**.

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

Parallel to the CUDA basics, I began exploring the **CUTLASS (CUDA Templates for Linear Algebra Subroutines)** library. CUTLASS decomposes matrix multiplication into reusable components and supports mixed‑precision, fused epilogues, and advanced scheduling. The `cutlass_demos/` directory contains a series of ten C++ programs that demonstrate progressively more sophisticated uses of CUTLASS.

| Demo | File | Concept |
| --- | --- | --- |
| 1 | `demo01_cutlass_gemm_fp32.cpp` | Basic FP32 GEMM using CUTLASS |
| 2 | `demo02_cutlass_gemm_fp16_tensor_core.cpp` | Half‑precision GEMM on Tensor Cores (FP16×FP16→FP32) |
| 3 | `demo03_cutlass_gemm_int8.cpp` | Quantized INT8 GEMM with INT32 accumulation |
| 4 | `demo04_cutlass_batched_gemm.cpp` | Batched GEMM – multiple matrix multiplies |
| 5 | `demo05_cutlass_conv2d.cpp` | 2‑D convolution via implicit GEMM |
| 6 | `demo06_cutlass_fused_bias_relu.cpp` | GEMM epilogue fused bias + ReLU |
| 7 | `demo07_cutlass_complex_gemm.cpp` | Complex number GEMM |
| 8 | `demo08_cutlass_async_gemm.cpp` | Async mainloop (e.g., `cp.async`) |
| 9 | `demo09_cutlass_grouped_gemm.cpp` | Grouped GEMM (varying sizes) |
| 10 | `demo10_cutlass_linear_softmax.cpp` | Linear layer with row‑wise softmax |

---

## NEW: PyTorch Learning Demos

Ten runnable Python scripts live in `pytorch_demos/` and mirror an **easy → hard** progression. Each file is self‑contained and commented to show the learning process.

| # | File | Title & Concept |
|---|---|---|
| 1 | `demo01_Pytorch_Tensor_Basics.py` | **Tensor Basics** – creation, dtype/device, view/reshape, broadcasting, indexing |
| 2 | `demo02_Pytorch_Autograd_Essentials.py` | **Autograd Essentials** – `requires_grad`, `backward()`, gradient tape introspection |
| 3 | `demo03_Pytorch_Linear_Regression_From_Scratch.py` | **Linear Regression from Scratch** – manual training loop with MSE + SGD |
| 4 | `demo04_Pytorch_Simple_Neural_Networks.py` | **MLP Classifier** – `nn.Module`, layers, activations, training/val loop |
| 5 | `demo05_Pytorch_Data_Handling.py` | **Data Handling** – `Dataset`, `DataLoader`, transforms, augmentation |
| 6 | `demo06_Pytorch_GPU_Acceleration.py` | **GPU Acceleration** – `.to(device)`, timing CPU vs GPU, reproducibility |
| 7 | `demo07_Pytorch_Model_Persistence.py` | **Model Persistence** – `state_dict` save/load, resume training, checkpoints |
| 8 | `demo08_Pytorch_Image_Classification.py` | **CNN on MNIST/CIFAR** – small ConvNet, accuracy tracking, confusion matrix |
| 9 | `demo09_Pytorch_Transfer_Learning.py` | **Transfer Learning** – fine‑tune ResNet‑18 from `torchvision.models` |
| 10 | `demo10_Pytorch_Advanced_Techniques.py` | **Advanced** – custom autograd function + TorchScript export (`torch.jit.trace/script`) |

### PyTorch Environment & Running

1. **Create env (conda or venv)**
   ```bash
   conda create -n ugra-gpu python=3.10 -y
   conda activate ugra-gpu
   ```
2. **Install PyTorch & extras** (choose the correct CUDA build from the PyTorch website if using GPU):
   ```bash
   pip install torch torchvision torchaudio   # CPU or CUDA build as needed
   pip install matplotlib scikit-learn tqdm
   ```
3. **Run a demo**
   ```bash
   python pytorch_demos/demo01_Pytorch_Tensor_Basics.py
   ```
4. **GPU check**
   ```python
   import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
   ```

> Note: `demo09_Pytorch_Transfer_Learning.py` may download pretrained weights on first run (internet required).

---

## Conceptual Roadmap / Process

This repo documents a **learning + research pipeline**, from low‑level kernels to high‑level frameworks.

### Phase 1 – CUDA Basics
- Kernels, indexing, grids/blocks; vector & matrix ops.

### Phase 2 – Memory Hierarchy & Asynchrony
- Global/shared/pinned/unified memory; streams & overlap.

### Phase 3 – Advanced CUDA Features
- Constant memory; reductions/atomics; profiling.

### Phase 4 – Towards Custom GPU Operations
- Design problem‑specific kernels; fuse elementwise ops; prepare to wrap as custom ops.

### Phase 5 – High‑Level Libraries (CUTLASS)
- Decompose GEMM; batched/grouped GEMM; fused epilogues; conv2d via implicit GEMM.

### **Phase 6 – PyTorch (NEW)**
- Tensors → autograd → modules → data → GPU → CNN/transfer learning → custom autograd → TorchScript.  
- Bridge to **custom ops**: prototype in PyTorch first, then replace hot paths with CUDA/CUTLASS implementations.

---

## How to Build / Run (summary)

- **CUDA/CUTLASS**: Visual Studio CUDA Runtime project or CMake (`project(... LANGUAGES CXX CUDA)`), include CUTLASS headers, compile `.cpp` as CUDA C/C++.
- **PyTorch**: standard Python environment; install `torch/torchvision/torchaudio`; run the scripts directly.

---

## Learning Resources

- NVIDIA CUDA Toolkit (installation, `nvcc`, Nsight tools) – <https://developer.nvidia.com/cuda-toolkit>  
- CUTLASS on GitHub – <https://github.com/NVIDIA/cutlass>  
- PyTorch Tutorials – <https://pytorch.org/tutorials/>  

---

## Future Work / TODO

- [x] Complete CUDA learning demos (`demo01`–`demo10`).
- [x] Create **CUTLASS** demo set (GEMM/conv/fused epilogues/grouped).
- [x] Add **PyTorch** demo set (tensors → TorchScript) with clear progression.
- [ ] Add `CMakeLists.txt` for CUDA/CUTLASS; simple `Makefile` for Linux.
- [ ] Benchmark: compare CUDA vs CUTLASS vs PyTorch on identical shapes (Nsight + `torch.utils.benchmark`).
- [ ] First end‑to‑end custom op: PyTorch module calling a bespoke CUDA/CUTLASS kernel.
- [ ] Expand `notes/` with experiment logs and design sketches.

---

## License

This project is for **research and educational** purposes as part of an undergraduate research project at RPI. Choose a license (e.g., MIT) once ready to share more broadly.
