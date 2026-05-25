"""Microbench: f32 matmul at our actual decode-step shapes.

Tests M=1 (single batch) vs M=2 (CFG-doubled) for K=N=1024 and K=N=4096.
If M=1 hits a much faster path, we should split CFG batching.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chatterbox_mojo"))

import torch
import numpy as np

# Just use torch for the timing since it's the most direct comparison.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

def bench(M, K, N, iters=200, warmup=20):
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)
    for _ in range(warmup):
        C = A @ B.T
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        C = A @ B.T
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = (time.perf_counter() - t0) / iters * 1e6   # microseconds
    flops = 2 * M * K * N
    gflops = flops / dt / 1e3
    return dt, gflops

for K, N in [(1024, 1024), (1024, 4096), (4096, 1024)]:
    for M in [1, 2]:
        dt, gflops = bench(M, K, N)
        print(f"M={M} K={K} N={N}: {dt:>7.1f} us  {gflops:>6.1f} GFLOP/s")
    print()
