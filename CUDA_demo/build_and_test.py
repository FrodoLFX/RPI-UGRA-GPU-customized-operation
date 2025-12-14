import time
import torch
from torch.utils.cpp_extension import load
from torch.profiler import profile, ProfilerActivity

# Build extension (first time will compile; later runs use cache)
ext = load(
    name="fused_gate_ext_v2",
    sources=["fused_gate.cpp", "fused_gate_cuda.cu"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["/O2"],
    verbose=True,
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

def run_prof(label, fn, iters=200, warmup=50):
    # Warm-up
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    print("\n==== PROFILER:", label, "====")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))

    trace_path = f"{label}_trace.json"
    prof.export_chrome_trace(trace_path)
    print("chrome trace saved:", trace_path)

def main():
    device = "cuda"
    M, N = 4096, 2048
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    v = torch.randn(N, device=device, dtype=torch.float32)
    threshold = 0.2

    # Baseline (PyTorch)
    y_ref = x * torch.where(v.abs() >= threshold, v, torch.zeros_like(v))

    # Extension
    y_ext = ext.forward(x, v, threshold)

    max_err = (y_ref - y_ext).abs().max().item()
    print("max_err:", max_err)

    baseline_fn = lambda: x * torch.where(v.abs() >= threshold, v, torch.zeros_like(v))
    ext_fn = lambda: ext.forward(x, v, threshold)

    t_ref = bench(baseline_fn)
    t_ext = bench(ext_fn)

    print(f"baseline(ms): {t_ref:.4f}")
    print(f"ext(ms):      {t_ext:.4f}")
    print(f"speedup:      {t_ref / t_ext:.2f}x")

    # Profiler evidence (kernel count / boundary overhead)
    run_prof("baseline", baseline_fn)
    run_prof("extension", ext_fn)

if __name__ == "__main__":
    main()
