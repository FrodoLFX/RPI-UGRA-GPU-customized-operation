import torch, time
from build_and_test import cutlass_linear

def run(M=2048, K=2048, N=2048, iters=50, device="cuda"):
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)

    # Warm-up
    for _ in range(10):
        _ = cutlass_linear(A, B, bias=bias, relu=True)
    torch.cuda.synchronize()

    # Timed
    t0 = time.time()
    for _ in range(iters):
        _ = cutlass_linear(A, B, bias=bias, relu=True)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"CUTLASS linear avg: {(t1 - t0)*1000/iters:.2f} ms over {iters} iters")

if __name__ == "__main__":
    assert torch.cuda.is_available()
    run()
