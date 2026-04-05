#!/usr/bin/env python3
"""Compare C native logits vs reference logits and (optionally) WASM logits."""

import struct
import sys
import os
import subprocess
import numpy as np

def load_f32_bin(path):
    data = open(path, "rb").read()
    return np.frombuffer(data, dtype=np.float32)

def compare(name_a, a, name_b, b):
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    diff = np.abs(a - b)
    mae = diff.mean()
    max_diff = diff.max()

    # Top-5 comparison for last position
    n = len(a)
    vocab = 50257
    last_a = a[-vocab:]
    last_b = b[-vocab:]
    top5_a = np.argsort(-last_a)[:5]
    top5_b = np.argsort(-last_b)[:5]

    print(f"\n  {name_a} vs {name_b}:")
    print(f"    MAE:      {mae:.6f}")
    print(f"    Max diff: {max_diff:.6f}")
    print(f"    Top-1 match: {top5_a[0] == top5_b[0]}  (A={top5_a[0]}, B={top5_b[0]})")
    print(f"    Top-5 A: {top5_a.tolist()}")
    print(f"    Top-5 B: {top5_b.tolist()}")
    overlap = len(set(top5_a) & set(top5_b))
    print(f"    Top-5 overlap: {overlap}/5")

    # Sample of actual logit values at last position
    print(f"    Sample logits (last pos, first 5 vocab):")
    print(f"      A: {last_a[:5]}")
    print(f"      B: {last_b[:5]}")

def main():
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    print(f"=== Verification: T={seq_len} ===")

    # 1. Run C code and dump logits
    print("\nRunning C native...")
    subprocess.run([
        "cc", "-O3", "-march=native", "-o", "bench/gpt2_cpu", "bench/gpt2_cpu.c", "-lm"
    ], check=True)
    subprocess.run([
        "./bench/gpt2_cpu", str(seq_len), "0", "1", "--dump", "bench/c_logits.bin"
    ], check=True)

    c_logits = load_f32_bin("bench/c_logits.bin")
    print(f"  C logits: {c_logits.shape[0]} floats, range [{c_logits.min():.4f}, {c_logits.max():.4f}]")

    # 2. Compare with reference output (if seq_len matches export)
    ref_path = "gpt2_weights/reference_output.bin"
    if os.path.exists(ref_path):
        ref_logits = load_f32_bin(ref_path)
        ref_seq = ref_logits.shape[0] // 50257
        if ref_seq == seq_len:
            compare("C native", c_logits, "PyTorch ref", ref_logits)
        else:
            print(f"\n  (Reference is T={ref_seq}, skipping comparison)")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
