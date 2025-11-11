import os, time, torch, math
from torch.utils.cpp_extension import load

CUTLASS_DIR = os.environ.get("CUTLASS_DIR", None)
assert CUTLASS_DIR is not None, "Please set CUTLASS_DIR to your local CUTLASS repo root"

this_dir = os.path.dirname(__file__)
ext_dir  = os.path.join(this_dir, "ext")

extra_include_paths = [os.path.join(CUTLASS_DIR, "include")]
extra_cuda_cflags = [
    "-O3",
    # Adjust arch to your GPU; add additional -gencode as needed.
    "-gencode=arch=compute_80,code=sm_80",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math"
]

cutlass_mod = load(
    name="cutlass_linear_ext",
    sources=[os.path.join(ext_dir, "cutlass_linear.cpp"),
             os.path.join(ext_dir, "cutlass_linear_kernel.cu")],
    extra_include_paths=extra_include_paths,
    extra_cuda_cflags=extra_cuda_cflags,
    with_cuda=True, verbose=True
)

def cutlass_linear(A, B, bias=None, relu=False):
    return torch.ops.cutlass_ext.linear(A, B, bias, relu)

def check_correctness(M=512, K=1024, N=256, device="cuda"):
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)

    ref = (A @ B) + bias
    out = cutlass_linear(A, B, bias=bias, relu=False)
    max_abs = (ref - out).abs().max().item()
    print(f"[check] max |ref - out| = {max_abs:.3e}")

def bench(M=4096, K=4096, N=4096, iters=20, device='cuda'):
    torch.cuda.synchronize()
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        _ = cutlass_linear(A, B, bias=bias, relu=True)
    torch.cuda.synchronize()

    # CUTLASS
    t0 = time.time()
    for _ in range(iters):
        _ = cutlass_linear(A, B, bias=bias, relu=True)
    torch.cuda.synchronize()
    t1 = time.time()
    cutlass_ms = (t1 - t0) * 1000 / iters

    # PyTorch mm + bias + relu
    for _ in range(5):
        _ = torch.relu(A @ B + bias)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = torch.relu(A @ B + bias)
    torch.cuda.synchronize()
    t1 = time.time()
    torch_ms = (t1 - t0) * 1000 / iters

    print(f"[bench] CUTLASS GEMM+bias+ReLU: {cutlass_ms:.2f} ms   |   torch.mm+bias+relu: {torch_ms:.2f} ms")

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU is required"
    print("Building extension and running quick checks...")
    check_correctness()
    bench()
    print("Done.")
