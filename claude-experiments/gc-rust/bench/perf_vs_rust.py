#!/usr/bin/env python3
"""Perf: gc-rust (AOT) vs equivalent hand-written Rust (-O).

Quantifies the "perf OK (slower-than-Rust is fine)" goal with real numbers, so
it stops being an assertion. Methodology:

  * gc-rust is built AHEAD-OF-TIME (`gcr build`) and the native binary is timed —
    NOT `gcr run`, which includes LLVM JIT-compile time and would pollute the
    measurement. Fair vs a compiled Rust binary.
  * Rust equivalents (bench/*.rs) are compiled with `rustc -O` (same LLVM backend
    gc-rust uses), faithful line-for-line translations of the examples.
  * Each binary is run best-of-N; we report the MIN wall time (least noise).
  * Correctness is checked separately via `gcr run` (which prints the full i64
    result) and the Rust stdout, both against the known-good expected value.

Compute-only benches (fib/nbody/mandelbrot) test whether monomorphized LLVM-O2
code reaches Rust speed. binary_trees is the allocation/tracing head-to-head:
gc-rust's GC vs Rust's Box (malloc/free) — the number that decides whether inline
bump allocation is urgent.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GCR = os.path.join(ROOT, "target", "release", "gcr")
TMP = "/tmp/gcr_bench"
RUNS = 9

# name -> expected i64 result (from scripts/run_examples.sh, known-good)
BENCHES = {
    "fib": (2178309, "compute"),
    "nbody": (921463, "compute"),
    "mandelbrot": (86906, "compute"),
    "small_alloc": (25000000000000, "alloc-small"),
    "binary_trees": (5242840, "alloc-large"),
}


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def best_time(cmd, runs=RUNS):
    best = None
    for _ in range(runs):
        t0 = time.perf_counter()
        subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
    return best


def main():
    os.makedirs(TMP, exist_ok=True)
    if not os.path.exists(GCR):
        print(f"building release gcr ...", flush=True)
        r = sh(["cargo", "build", "--release", "-q"])
        if r.returncode != 0:
            print(r.stderr)
            sys.exit(1)

    rows = []
    for name, (expected, kind) in BENCHES.items():
        gcr_src = os.path.join("examples", f"{name}.gcr")
        rust_src = os.path.join("bench", f"{name}.rs")
        gcr_bin = os.path.join(TMP, f"gcr_{name}")
        rust_bin = os.path.join(TMP, f"rust_{name}")

        # Build both.
        rb = sh([GCR, "build", gcr_src, "-o", gcr_bin])
        if rb.returncode != 0:
            print(f"[{name}] gcr build failed:\n{rb.stderr}")
            continue
        rr = sh(["rustc", "-O", rust_src, "-o", rust_bin])
        if rr.returncode != 0:
            print(f"[{name}] rustc failed:\n{rr.stderr}")
            continue

        # Correctness: gc-rust full result via JIT (prints), Rust via stdout.
        gcr_run = sh([GCR, "run", gcr_src])
        gcr_res = gcr_run.stdout.strip().splitlines()[-1] if gcr_run.stdout.strip() else "?"
        rust_res = sh([rust_bin]).stdout.strip()
        ok = (gcr_res == str(expected) and rust_res == str(expected))

        # Timing (AOT binary vs Rust binary), best-of-N.
        g = best_time([gcr_bin]) * 1000.0
        r = best_time([rust_bin]) * 1000.0
        ratio = g / r if r > 0 else float("inf")
        rows.append((name, kind, g, r, ratio, ok, gcr_res, rust_res, expected))

    # GC stats for the alloc benchmark (context for the ratio).
    gc_stats = ""
    bt = sh([os.path.join(TMP, "gcr_binary_trees")], env={**os.environ, "GCR_GC_STATS": "1"})
    for line in bt.stderr.splitlines():
        if "pause" in line or "collections" in line:
            gc_stats = line.strip()

    print()
    print("gc-rust (AOT) vs Rust (rustc -O) — best of %d runs, min wall time" % RUNS)
    print("=" * 78)
    print(f"{'benchmark':<16}{'kind':<10}{'gcr ms':>10}{'rust ms':>10}{'gcr/rust':>10}  {'result':>8}")
    print("-" * 78)
    for name, kind, g, r, ratio, ok, gres, rres, exp in rows:
        flag = "" if ok else f"  !! MISMATCH gcr={gres} rust={rres} exp={exp}"
        print(f"{name:<16}{kind:<10}{g:>10.2f}{r:>10.2f}{ratio:>9.2f}x  {'ok' if ok else 'BAD':>8}{flag}")
    print("-" * 78)
    if gc_stats:
        print(f"binary_trees GC: {gc_stats}")
    print()
    print("Note: compute benches gauge monomorphized-LLVM vs Rust; binary_trees")
    print("gauges GC-alloc (out-of-line ai_gc_alloc_* call, no inline bump) vs Box.")


if __name__ == "__main__":
    main()
