#!/usr/bin/env python3
"""Benchmark: clox vs our JIT."""

import os
import subprocess
import sys
import statistics

CLOX = "/tmp/craftinginterpreters/build/clox"
JIT = os.path.join(os.path.dirname(__file__), "..", "..", "target", "release", "lox")
BENCHMARKS_DIR = "/tmp/craftinginterpreters/test/benchmark"
SKIP = {"zoo_batch.lox", "trees.lox", "binary_trees.lox"}
RUNS = 5


def extract_time(output):
    for line in reversed(output.strip().splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None


def run(binary, args, filepath, timeout=120):
    cmd = [binary] + args + [filepath]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None
        return extract_time(r.stdout)
    except subprocess.TimeoutExpired:
        return None


def fmt(t):
    if t is None: return "FAIL"
    if t < 0.001: return f"{t*1e6:.0f}us"
    if t < 1.0: return f"{t*1000:.1f}ms"
    return f"{t:.3f}s"


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else RUNS
    benchmarks = sorted(f for f in os.listdir(BENCHMARKS_DIR) if f.endswith(".lox") and f not in SKIP)

    print(f"clox vs JIT  ({runs} runs each)")
    print()

    w = max(len(b.replace(".lox", "")) for b in benchmarks) + 2
    print(f"{'Benchmark':<{w}} {'clox':>12} {'JIT':>12} {'JIT/clox':>10}")
    print("-" * (w + 12 + 12 + 10 + 6))

    speedups = []
    for bench in benchmarks:
        path = os.path.join(BENCHMARKS_DIR, bench)
        name = bench.replace(".lox", "")

        clox_times = []
        jit_times = []
        for _ in range(runs):
            c = run(CLOX, [], path)
            j = run(JIT, ["--jit"], path)
            if c is not None: clox_times.append(c)
            if j is not None: jit_times.append(j)

        if not clox_times or not jit_times:
            print(f"{name:<{w}} {fmt(clox_times[0] if clox_times else None):>12} {fmt(jit_times[0] if jit_times else None):>12} {'N/A':>10}")
            continue

        cm = statistics.median(clox_times)
        jm = statistics.median(jit_times)
        ratio = jm / cm if cm > 0 else float('inf')
        speedups.append(ratio)

        print(f"{name:<{w}} {fmt(cm):>12} {fmt(jm):>12} {ratio:>9.2f}x")

    print("-" * (w + 12 + 12 + 10 + 6))
    if speedups:
        geo = statistics.geometric_mean(speedups)
        print(f"{'GEO MEAN':<{w}} {'':>12} {'':>12} {geo:>9.2f}x")
        print()
        if geo < 1:
            print(f"JIT is {1/geo:.1f}x FASTER than clox (geo mean)")
        else:
            print(f"JIT is {geo:.1f}x SLOWER than clox (geo mean)")


if __name__ == "__main__":
    main()
