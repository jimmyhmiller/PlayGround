# aot-bench — statically-compiled SIMD JS stage-1 (no JIT)

Proves the `.simd` SIMD JS stage-1 kernel can ship **without a JIT** and
benchmarks it against the JIT version.

## How it works

- `simd-lang compile examples/js_stage1.simd -o aot-bench/generated` lowers the
  `.simd` source through MLIR→LLVM at **-O3, host CPU** and dumps a relocatable
  native-NEON object, archived into `generated/libjs_stage1.a` with Rust bindings
  in `generated/js_stage1.rs`.
- This crate links that static lib and calls `js_stage1` directly. It has **no
  dependency on melior / MLIR / LLVM** — its only dependency is `simd-lang` built
  with `default-features = false` (the pure-Rust lexer, used as the correctness
  oracle).

## Run

```sh
# from the simd-lang root — (re)generate the AOT artifact whenever js_stage1.simd changes
simd-lang compile examples/js_stage1.simd -o aot-bench/generated   # or: ./target/debug/simd-lang …

# AOT (this crate, zero-MLIR):
cargo run --release --manifest-path aot-bench/Cargo.toml -- <file.js> [iters]

# JIT version, same kernel, same MB/s formula:
./target/debug/simd-lang jit-stage1 <file.js> [iters]
```

`aot-bench` asserts its two output bitmaps are byte-identical to the pure-Rust
reference before timing, so a perf number always implies a correct kernel.

## Result (Apple Silicon, M-series)

Same MLIR→LLVM pipeline at O3 for both, so steady-state throughput is identical
(~8 GB/s, byte-bound). The difference is **startup**:

| | stage-1 throughput | compile cost |
|---|---|---|
| **AOT** (`libjs_stage1.a`) | ~7.9–8.0 GB/s | **0 ms** — compiled once, linked in |
| **JIT** (ExecutionEngine) | ~7.9–8.0 GB/s | **~42 ms every launch** |
| pure-Rust hand-written NEON (ref) | ~4.5 GB/s | n/a |

Takeaways: AOT matches JIT throughput exactly while eliminating the per-launch
MLIR compile, and runs with no LLVM in the process. (The MLIR-compiled kernel
also beats the hand-written-Rust NEON reference ~1.75×.)
