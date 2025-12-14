import time
import torch
from torch.utils.cpp_extension import load

ext = load(
    name="cats_fused_ext_v1",
    sources=["cats_ext.cpp", "cats_ext_cuda.cu"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["/O2"],
    verbose=False,   #turn off build spam for clean screenshots
)


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / iters

def main():
    torch.manual_seed(0)
    device = "cuda"

    # Choose sizes where the extra C read/write can matter (smaller K makes fusion more visible)
    M, K, N = 1024, 128, 1024
    threshold = 0.2

    A = torch.randn(M, K, device=device, dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device=device, dtype=torch.float32).contiguous()
    v = torch.randn(N, device=device, dtype=torch.float32).contiguous()

    # Baseline (two kernels): matmul -> gate
    def baseline():
        C = ext.matmul(A, B)
        return ext.gate(C, v, threshold)

    # Fused (one kernel): matmul+gate
    def fused():
        return ext.fused(A, B, v, threshold)

    y0 = baseline()
    y1 = fused()
    max_err = (y0 - y1).abs().max().item()
    print("max_err:", max_err)

    t_base = bench(baseline)
    t_fuse = bench(fused)

    speedup = t_base / t_fuse
    pct = (1.0 - t_fuse / t_base) * 100.0

    print("\n" + "=" * 78)
    print("Phase 2 Evidence — CATS-style Operator Fusion (PyTorch CUDA Extension)")
    print("-" * 78)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Env: torch={torch.__version__} | torch.cuda={torch.version.cuda}")
    print(f"Workload: A[{M},{K}] @ B[{K},{N}] -> Out[{M},{N}] (fp32)")
    print(f"Gate: column-wise v[N], threshold={threshold}")
    print("-" * 78)
    print(f"Correctness: max_err = {max_err:.6g}")
    print(f"Baseline: matmul + gate (2 kernels)  = {t_base:.4f} ms/iter")
    print(f"Fused:    matmul⨉gate epilogue (1k) = {t_fuse:.4f} ms/iter")
    print(f"Result:   speedup = {speedup:.2f}x  ({pct:.1f}% faster)")
    print("-" * 78)
    print("Interpretation: Fusion removes one kernel boundary and avoids an extra")
    print("global read/write of the intermediate output tensor.")
    print("=" * 78 + "\n")



if __name__ == "__main__":
    main()
