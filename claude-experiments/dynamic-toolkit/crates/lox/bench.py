#!/usr/bin/env python3
"""Benchmark runner: compares interpreter vs JIT across the Crafting Interpreters benchmark suite."""

import os
import re
import subprocess
import sys
import time
import statistics

LOX_BINARY = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "target", "release", "lox",
)

BENCHMARKS_DIR = "/tmp/craftinginterpreters/test/benchmark"

# zoo_batch runs for a fixed 10s wall-clock, not useful for time comparison
# trees segfaults (deep recursion GC issue), binary_trees gives wrong results (correctness bug)
SKIP = {"zoo_batch.lox", "trees.lox", "binary_trees.lox"}

RUNS = 3  # number of iterations per benchmark


def extract_time(output: str, name: str) -> float | None:
    """Extract the last floating-point number printed (the elapsed time)."""
    lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
    # Most benchmarks print the time as the last numeric line.
    # binary_trees prints "elapsed:" then the number.
    for i in range(len(lines) - 1, -1, -1):
        try:
            return float(lines[i])
        except ValueError:
            continue
    return None


def run_benchmark(filepath: str, jit: bool, timeout: float = 120) -> float | None:
    """Run a benchmark and return the self-reported elapsed time in seconds."""
    cmd = [LOX_BINARY]
    if jit:
        cmd.append("--jit")
    cmd.append(filepath)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    return extract_time(result.stdout, os.path.basename(filepath))


def run_wall_clock(filepath: str, jit: bool, timeout: float = 120) -> float | None:
    """Run a benchmark and return wall-clock time."""
    cmd = [LOX_BINARY]
    if jit:
        cmd.append("--jit")
    cmd.append(filepath)

    try:
        start = time.perf_counter()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.perf_counter() - start
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    return elapsed


def fmt_time(t: float | None) -> str:
    if t is None:
        return "FAIL"
    if t < 0.001:
        return f"{t*1_000_000:.0f}µs"
    if t < 1.0:
        return f"{t*1000:.1f}ms"
    return f"{t:.3f}s"


def main():
    runs = RUNS
    if len(sys.argv) > 1:
        runs = int(sys.argv[1])

    # Ensure binary exists
    if not os.path.isfile(LOX_BINARY):
        print(f"Binary not found: {LOX_BINARY}")
        print("Run: cargo build --release -p lox")
        sys.exit(1)

    benchmarks = sorted(
        f for f in os.listdir(BENCHMARKS_DIR)
        if f.endswith(".lox") and f not in SKIP
    )

    print(f"Running {len(benchmarks)} benchmarks, {runs} iterations each")
    print(f"Binary: {LOX_BINARY}")
    print()

    # Header
    name_w = max(len(b.replace(".lox", "")) for b in benchmarks) + 2
    print(f"{'Benchmark':<{name_w}} {'Interp (med)':>12} {'JIT (med)':>12} {'Speedup':>10} {'Interp (σ)':>12} {'JIT (σ)':>12}")
    print("─" * (name_w + 12 + 12 + 10 + 12 + 12 + 10))

    total_interp = 0.0
    total_jit = 0.0
    speedups = []

    for bench_file in benchmarks:
        filepath = os.path.join(BENCHMARKS_DIR, bench_file)
        name = bench_file.replace(".lox", "")

        interp_times = []
        jit_times = []

        for i in range(runs):
            # Alternate to reduce thermal bias
            it = run_benchmark(filepath, jit=False)
            jt = run_benchmark(filepath, jit=True)
            if it is not None:
                interp_times.append(it)
            if jt is not None:
                jit_times.append(jt)

        if not interp_times or not jit_times:
            interp_str = fmt_time(interp_times[0] if interp_times else None)
            jit_str = fmt_time(jit_times[0] if jit_times else None)
            print(f"{name:<{name_w}} {interp_str:>12} {jit_str:>12} {'N/A':>10}")
            continue

        i_med = statistics.median(interp_times)
        j_med = statistics.median(jit_times)
        i_std = statistics.stdev(interp_times) if len(interp_times) > 1 else 0
        j_std = statistics.stdev(jit_times) if len(jit_times) > 1 else 0

        if j_med > 0:
            speedup = i_med / j_med
            speedups.append(speedup)
        else:
            speedup = float('inf')

        total_interp += i_med
        total_jit += j_med

        speedup_str = f"{speedup:.2f}x"
        print(f"{name:<{name_w}} {fmt_time(i_med):>12} {fmt_time(j_med):>12} {speedup_str:>10} {fmt_time(i_std):>12} {fmt_time(j_std):>12}")

    print("─" * (name_w + 12 + 12 + 10 + 12 + 12 + 10))

    if speedups:
        geo_mean = statistics.geometric_mean(speedups)
        total_speedup = total_interp / total_jit if total_jit > 0 else 0
        print(f"{'TOTAL':<{name_w}} {fmt_time(total_interp):>12} {fmt_time(total_jit):>12} {total_speedup:.2f}x")
        print(f"{'GEO MEAN':<{name_w}} {'':>12} {'':>12} {geo_mean:.2f}x")
        print()
        print(f"Geometric mean speedup: {geo_mean:.2f}x")


if __name__ == "__main__":
    main()
