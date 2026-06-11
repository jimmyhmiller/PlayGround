# ai-lang vs Rust vs Go benchmarks

Classic benchmark programs implemented three times with identical
algorithms: `ail/*.ail` (ai-lang), `rust/src/bin/*.rs` (Rust, `-O3 +
lto`), and `go/<name>/main.go` (Go).

Each program self-times its core workload and prints
`RESULT <name> <ms> ms checksum=<n>`, so JIT/compile/startup time is
excluded in every language. The harness rejects a run if any language
disagrees on the checksum — every benchmark is also a correctness test
(nbody's checksum is the total energy after 500k steps scaled to an
integer, and it matches across all three languages bit-for-bit because
every version uses the same op order).

## Run

```
benchmarks/run.sh [runs]    # default 3, best-of reported
```

## Benchmarks

| name | what it measures |
|---|---|
| `fib` | naive recursive fib(32) — function-call overhead |
| `loop_mix` | 500M-iteration add/xor/shift loop — tail recursion as the loop construct vs a native loop; the mix defeats closed-form folding |
| `mandelbrot` | 1000×1000 grid, 100 max iters — f64 arithmetic in hot inner recursion |
| `nbody` | 5 bodies, 500k steps — f64 + `Array<Float>` columns + `float_sqrt`; arrays use the UNBOXED `PrimArray` representation, and every access goes through the CHECKED `array_get(..)?` / `array_set(..)?` API (an out-of-bounds would flow out as `Result::Err(IndexError)`) |
| `binary_trees` | 40 complete trees of depth 16 (~5.2M nodes) — GC allocation/tracing vs malloc/free (Rust) vs generational GC (Go) |

## Results (2026-06-11, Apple Silicon, best of 3)

| benchmark | ai-lang (ms) | Rust (ms) | Go (ms) | vs Rust | vs Go |
|---|---|---|---|---|---|
| fib | 9 | 6 | 6 | 1.5× | 1.5× |
| loop_mix | 433 | 412 | 418 | 1.05× | 1.04× |
| mandelbrot | 63 | 59 | 59 | 1.07× | 1.07× |
| nbody | 329 | 25 | 21 | 13× | 16× |
| binary_trees | 103 | 88 | 47 | 1.17× | 2.2× |

Notes:

- loop_mix and mandelbrot are at PARITY with Rust/Go, and fib is 1.5×.
  Three changes got us here: (1) the per-call pending-panic checks are
  gone (errors are `Result` values; a contract violation hard-aborts);
  (2) provably scalar-only functions skip the GC frame entirely — no
  alloca, no memset, no chain link (`scalar_only_body` in codegen.rs);
  (3) the JIT now runs LLVM's `default<O2>` pipeline — previously NO IR
  passes ran at all (the engine's OptimizationLevel only drives
  instruction selection), so nothing inlined and no alloca was ever
  promoted. The safepoint poll's state load is volatile so LICM can't
  hoist it.
- Scalar arrays (`Array<Int>` / `Array<Float>` / `Array<Bool>`) are
  UNBOXED when the creation site's context pins the element type (def
  return type, struct field, call argument): raw 8-byte slots, untraced
  by the GC, zero allocation per `array_set`. Arrays created in generic
  code keep the boxed representation; every accessor branches on the
  shape at runtime, so the two representations are interchangeable
  (including `value_eq`/`value_hash`, which are representation-blind).
  This took nbody from 26× Rust (boxed, ~37M allocations) to 15×.
- Indexing is CHECKED and errors are values: the public `array_get` /
  `array_set` / `bytes_get` / `bytes_set` return
  `Result<T, IndexError>`, and `?` is FUSED — the happy path compiles
  to an inline shape check + bounds check + raw load/store with no
  Result ever built; only an actual out-of-bounds constructs (and
  early-returns) the `Err(OutOfBounds(OobInfo { index, len }))`. The
  greppable `*_trusted` tier skips the Result protocol and ABORTS on a
  violation, for code with proven indices (the stdlib's internals).
  With trusted accessors nbody runs 159ms (6.6×); the table reports
  the checked version because that's what user code writes.
- Frames are EXACT: no conservative pre-scan anywhere — the alloca,
  the zeroing memset, and the GC's scanned-slot count all equal the
  body's real slot high-water mark (materialized after compilation via
  replace-all-uses). nbody's remaining gap is per-use volatile reloads
  of pointer params (required around potential GC points) which block
  CSE of the per-access checks; the next lever is call-boundary-aware
  reloads (stack-map-lite).
- `binary_trees`: Go's generational GC wins on pure allocation churn
  (46ms vs Rust's 87ms of malloc/free); ai-lang's copying GC lands at
  1.9× Rust / 3.5× Go.
