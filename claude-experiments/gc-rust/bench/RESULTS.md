# Perf: gc-rust vs Rust — first measured baseline

Run: `python3 bench/perf_vs_rust.py` (gc-rust AOT via `gcr build` vs `rustc -O`,
best of 9, min wall time). This converts the "perf OK (slower-than-Rust is fine)"
goal from an assertion into a measurement.

| benchmark | kind | gcr ms | rust ms | gcr/rust | result |
|-----------|------|-------:|--------:|---------:|:------:|
| fib | compute | 9.5 | 11.7 | **0.8–1.1×** | ok |
| nbody | compute | 8.8 | 5.6 | **1.0–1.6×** | ok |
| mandelbrot | compute | 5.0 | 3.0 | **1.5–1.7×** | ok |
| small_alloc | alloc-small | 367 | 74 | **~5×** | ok |
| binary_trees | alloc-large | 3118 | 116 | **~27–48×** | ok |

(Compute ratios vary run-to-run by codegen/FP-contraction + machine noise; the
alloc-large ratio is GC-pause-dominated and noisy — two runs gave 27× and 48×.)

`binary_trees` GC: 200 minor collections, pause p50 ~20ms / **p99 ~300–400ms**,
**reclaimed 0 B, promoted 200 MB**.

## Verdict

- **Compute is at parity with Rust (~0.8×–1.7×).** Monomorphized LLVM-O2 reaches
  Rust speed on non-allocating code — the core codegen is sound and the "perf OK"
  goal is *met* for compute.
- **Allocation has a wide range, bracketed by the two alloc benches:**
  - **small, short-lived objects → ~5×** (small_alloc: 5M Pairs that die in the
    nursery, reclaimed by cheap minor GCs — the case generational GC is built
    for). This ~5× is the clean measure of the **out-of-line `ai_gc_alloc_*` call
    overhead** — exactly what **inline bump allocation** would target. "OK-slower,"
    not alarming.
  - **large / promoted objects → ~27–48× (alarming worst case).** The cause is
    multi-pronged, NOT just "no inline bump":
  1. **Out-of-line `ai_gc_alloc_*` call per object** (5.24M calls) — the known
     inline-bump gap (FUTURE_WORK P2).
  2. **AOT nursery is 1 MB** (and inconsistent with the JIT's 16 MB — see below),
     while one depth-16 tree is **5.24 MB** (131071 nodes × 40 B) — far bigger
     than the nursery. So every tree is **promoted wholesale during
     construction**: the 200 minor GCs copy live partial-tree data and reclaim
     **nothing** (reclaimed 0 B, promoted 200 MB).
  3. **No major GC fires** (the 256 MB tenured threshold isn't hit in-run), so the
     40 dead trees accumulate in tenured and are never reclaimed — effectively a
     within-run leak. A major-GC trigger on tenured pressure would fix the
     reclaim-0 behavior.
  4. p99 pause **283 ms** — dominated by promotion copying.

## Findings worth acting on

- **AOT/JIT heap-size inconsistency:** `gcr_runtime_main` (AOT) uses a **1 MB**
  nursery; `jit_run_i64_mode` (JIT) uses **16 MB**. Same program, different GC
  behavior and perf depending on run mode. These should agree (or be configurable).
- The 48× says allocation perf is **alarming in the worst case** → warrants
  priority, but the remedy is **multi-pronged** (inline bump **+** nursery sizing /
  AOT-JIT consistency **+** a major-GC-on-tenured-pressure trigger), not inline
  bump alone.

## Bracketing

The two alloc benches bracket the range: **~5× (small_alloc, the typical
short-lived case) to ~27–48× (binary_trees, the large-promoted worst case).**
Don't over-generalize the worst case — for objects that die young, gc-rust is
~5× off Rust's malloc/free, and that gap is squarely the inline-bump target.

## Prioritization read

Compute is done (parity). Allocation is the perf frontier: **inline bump** closes
the ~5× typical case; **nursery sizing / AOT-JIT consistency + a major-GC-on-
tenured-pressure trigger** are needed to fix the large-object cliff. The compute
parity means none of this is a codegen problem — it's all in the GC/alloc path,
which is the right place for it to be.
