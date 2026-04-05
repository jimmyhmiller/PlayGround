#!/usr/bin/env python3
"""GPT-2 forward pass benchmark using PyTorch (CPU, single-threaded).

Usage: python3 bench/gpt2_pytorch.py [seq_len] [warmup] [iters]
"""

import sys
import time

import torch

# Force single-threaded for fair comparison
torch.set_num_threads(1)

def main():
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    print(f"PyTorch {torch.__version__}, threads={torch.get_num_threads()}", file=sys.stderr)
    print(f"T={seq_len}, warmup={warmup}, iters={iters}", file=sys.stderr)

    from transformers import GPT2LMHeadModel
    print("Loading GPT-2...", file=sys.stderr)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # Same dummy tokens as C/WASM benchmark
    tokens = torch.tensor([[464 + i for i in range(seq_len)]], dtype=torch.long)
    print(f"Tokens: {tokens[0].tolist()}", file=sys.stderr)

    # Warmup
    print("Warming up...", file=sys.stderr)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(tokens)

    # Benchmark
    print("Benchmarking...", file=sys.stderr)
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            out = model(tokens)
            times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)

    print(f"PyTorch CPU:  avg={avg:8.1f}ms  min={mn:8.1f}ms  max={mx:8.1f}ms")

    # Print top-5
    logits = out.logits[0, -1]
    top5 = torch.topk(logits, 5)
    print("Top-5:", file=sys.stderr)
    for v, i in zip(top5.values, top5.indices):
        print(f"  idx={i.item()} logit={v.item():.4f}", file=sys.stderr)

if __name__ == "__main__":
    main()
