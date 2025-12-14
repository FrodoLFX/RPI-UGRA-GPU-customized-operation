import time
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Utilities
# -----------------------------
def device_info():
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"torch_cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")


class Net(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def run_benchmark(
    device: torch.device,
    batch: int,
    data_dim: int,
    hidden_dim: int,
    num_classes: int,
    iters: int = 200,
    warmup: int = 20,
) -> tuple[float, float]:
    """
    Returns: (final_loss, avg_ms_per_iter)
    Measures forward + backward + optimizer.step time.
    """
    torch.manual_seed(0)

    model = Net(data_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(batch, data_dim, device=device)
    y = torch.randint(0, num_classes, (batch,), dtype=torch.long, device=device)

    # Warm-up
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / iters
    return float(loss.item()), avg_ms


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    device_info()

    # (A) Keep the original "one update" behavior (small network)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dim = 100
    hidden_dim = 50
    num_classes = 10

    model = Net(data_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    X = torch.randn(1024, data_dim, device=device)
    y = torch.randint(0, num_classes, (1024,), dtype=torch.long, device=device)

    logits = model(X)
    loss = criterion(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"Loss after one update: {loss.item():.4f}")

    # (B) Benchmark CPU vs GPU (bigger workload so GPU advantage shows up)
    bench_batch = 4096
    bench_data_dim = 1024
    bench_hidden_dim = 2048
    bench_classes = 10
    iters = 200
    warmup = 20

    loss_cpu, ms_cpu = run_benchmark(
        torch.device("cpu"),
        bench_batch,
        bench_data_dim,
        bench_hidden_dim,
        bench_classes,
        iters=iters,
        warmup=warmup,
    )
    print(f"CPU benchmark: loss={loss_cpu:.4f}, time={ms_cpu:.3f} ms/iter")

    if torch.cuda.is_available():
        loss_gpu, ms_gpu = run_benchmark(
            torch.device("cuda"),
            bench_batch,
            bench_data_dim,
            bench_hidden_dim,
            bench_classes,
            iters=iters,
            warmup=warmup,
        )
        print(f"GPU benchmark: loss={loss_gpu:.4f}, time={ms_gpu:.3f} ms/iter")
        print(f"Speedup: {ms_cpu / ms_gpu:.2f}x")
    else:
        print("GPU benchmark skipped (CUDA not available).")
