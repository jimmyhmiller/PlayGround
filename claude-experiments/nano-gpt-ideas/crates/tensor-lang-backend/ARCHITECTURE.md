# Tensor Lang Backend Architecture

This document describes how the backend compiles a tensor dataflow graph into
executable WASM, and the key design decisions that got us from 17.6 seconds of
GPT-2 inference down to 4.5 seconds — while also eliminating per-token
recompilation entirely.

## The Pipeline

```
DSL source → Parser → Graph → Loop IR → AssemblyScript → WASM
```

Each stage narrows the representation. The DSL is high-level (`matmul(a, b)`).
The graph decomposes that into primitives (`reshape → expand → mul → reducesum
→ reshape`). The loop IR fuses chains of operations into single loops. The AS
codegen turns those loops into executable code. `asc` compiles that to WASM.

## Stage 1: Graph (`tensor-lang-graph`)

The graph is a list of nodes. Each node has an operation, a list of inputs
(references to other nodes), and a shape:

```rust
struct Node {
    op: Op,
    inputs: Vec<NodeId>,
    shape: Vec<Dim>,
}
```

`Dim` is the key type that makes the whole system work:

```rust
enum Dim {
    Lit(usize),       // compile-time known: 768, 12, 50257
    Param(String),    // runtime parameter: "T" for sequence length
    Add(Box<Dim>, Box<Dim>),
    Mul(Box<Dim>, Box<Dim>),
    Div(Box<Dim>, Box<Dim>),
    Sub(Box<Dim>, Box<Dim>),
}
```

When you write `load([1, T, 768])` in the DSL (after declaring `dim T`),
the graph stores the shape as `[Lit(1), Param("T"), Lit(768)]`. Shape
inference propagates symbolic dims through operations. A `neg` of
`[1, T, 768]` produces `[1, T, 768]`. A matmul of `[1, T, 768]` and
`[768, 2304]` produces `[1, T, 2304]`. The output has `T` in the right
place without anyone having to know what `T` actually is.

### What used to happen

Shapes were `Vec<usize>`. When seq_len changed from 3 to 4, the entire
graph had to be rebuilt because every shape containing seq_len was
different. This forced full recompilation for every generated token.

### What happens now

Shapes are `Vec<Dim>`. The graph is built once with `Param("T")` for
seq_len. The graph structure doesn't depend on what `T` is — only on the
model architecture (which dims are size 1, how many dims each tensor has,
which axes get reduced). So the graph is compiled once, regardless of
sequence length.

## Stage 2: Loop IR (`loop_ir.rs`)

The loop IR takes the graph and produces a flat list of statements:

```rust
enum Stmt {
    Alloc { buf: usize, size: Dim },
    Fill { buf: usize, value: f64 },
    FillArange { buf: usize, size: Dim },
    Loop { buf, shape: Vec<Dim>, reduce, body: Vec<Inst>, result, tile },
    Pad { buf, input_buf, output_shape, input_shape, padding },
}
```

### Fusion

The main job of `lower()` is deciding which graph nodes get their own loop
(materialized) and which get inlined into another node's loop body (fused).

A node is materialized if:
- It's a source (Input, Constant, Arange)
- It has multiple consumers (its value is read more than once)
- It's a reduce output (downstream has a different iteration shape)
- It's a reshape that changes dimensionality (not just inserting/removing 1s)
- It's the final output

Everything else gets inlined. For a matmul decomposition
`reshape → expand → mul → reducesum → reshape`, the reshape, expand, and
mul are all single-consumer and get fused into the reducesum's loop body.
The result is one loop that does the entire matmul.

The fused body is a flat instruction list:

```rust
enum Inst {
    Load { buf: usize, index: Index },
    Const(f64),
    DimVar(usize),
    Neg(usize), Recip(usize), Exp2(usize), Log2(usize), Sqrt(usize),
    Add(usize, usize), Mul(usize, usize), Max(usize, usize), CmpLt(usize, usize),
}
```

Each instruction produces a value referenced by its index in the list.
`Mul(0, 1)` means "multiply the results of instructions 0 and 1."

### Index computation

`Index` describes how to compute a buffer offset from loop dimension
variables:

```rust
enum Index {
    Flat,                                    // offset = oi (flat loop index)
    Strided { parts: Vec<(usize, Dim)>, offset: Dim },  // offset = sum(d[dim] * stride) + offset
}
```

The `parts` list says "dimension variable d0 times stride 768, plus
dimension variable d3 times stride 1." This is how the fused loop body
knows which element of A to load when computing a matmul — even though
A, B, and the output all have different shapes and layouts.

The strides are `Dim` values, so they can be symbolic. A stride of
`Mul(Param("T"), Lit(768))` means "T times 768" — computed at runtime.

### Tiling

`tile_reduce_loops()` tags reduce loops with a `TileConfig` that says
how to block the iteration. For a matmul with output shape `[1, T, 2304]`
and reduce K=768, it sets tile sizes `[1, T, 32, 32]` — tile the last
two output dims (or the full dim if it's symbolic) and the reduce dim by
blocks of 32.

Tiling decisions use `.as_usize()` to check if a dim is large enough to
bother tiling. Symbolic dims (which could be anything at runtime) are
always tiled — the tiled code correctly handles any size via remainder
paths.

## Stage 3: AssemblyScript Codegen (`assemblyscript.rs`)

This is where the loop IR becomes executable code. The big design decisions
here are kernel extraction, tiled loop emission, and symbolic dim handling.

### Kernel extraction

This is the most important architectural decision. Without it, GPT-2 generates
25,000+ lines of AssemblyScript (the entire 12-layer forward pass inlined into
one function). With it: ~3,900 lines.

The observation: GPT-2 has ~84 matmuls but only ~15 unique loop structures.
Layer 0's QKV projection and layer 5's QKV projection have identical loop
bodies, shapes, strides, and tile configs — they differ only in which buffers
they read from and write to.

The codegen does two passes:

**Pass 1: Signature computation.** For each `Stmt::Loop`, normalize the body
by replacing buffer IDs with ordinals (first unique buffer → 0, second → 1).
The signature is the tuple of (shape, reduce config, normalized body, tile
config). Loops with identical signatures share a kernel.

**Pass 2: Emission.** Each unique kernel becomes a standalone function:

```typescript
function kernel_3(T: i32, _out: Float32Array, _in0: Float32Array, _in1: Float32Array): void {
    // the tiled loop body, emitted ONCE
}
```

The `execute()` function just allocates buffers and calls kernels:

```typescript
export function execute(T: i32, input_0: Float32Array, ...): Float32Array {
    const buf0 = input_0;
    const buf5 = new Float32Array((T * 2304));
    kernel_3(T, buf5, buf0, buf2);    // layer 0 QKV — reuses kernel_3
    ...
    kernel_3(T, buf17, buf12, buf14); // layer 1 QKV — same kernel_3!
    ...
}
```

### What used to happen

Every loop was emitted inline in `execute()`. The same matmul code appeared
12 times (once per layer), with the only difference being buffer names.
25,000 lines of AssemblyScript took `asc` 10+ seconds to compile.

### What happens now

~15 kernel functions × ~100-200 lines each = ~2,000 lines of kernels, plus
~200 lines of `execute()` body. Total ~3,900 lines. `asc` compiles this in
~1.6 seconds.

### Tiled loop emission

For reduce loops tagged with a `TileConfig`, the codegen emits a blocked
loop nest with three key optimizations:

**1. N-unrolling with scalar accumulators.** Instead of looping over the
N dimension, the codegen unrolls it into explicit `let _acc0, _acc1, ...,
_acc31` variables. These map to WASM registers — a register add is much
faster than a memory read-modify-write through an array.

**2. A-load hoisting.** The codegen analyzes which body instructions
depend on the N dimension (`compute_n_dependence`). Instructions that
don't (like loading a value from matrix A in a matmul) are emitted once
before the unrolled blocks. Each A value is loaded once and reused across
all 32 N iterations.

**3. Block-count optimization.** When a dimension is concrete (`Dim::Lit`),
the block count (`ceil(N/32)`) is pre-computed in Rust and stamped as a
literal. When symbolic (`Dim::Param`), it's emitted as a runtime expression
`((T + 31) / 32)`.

### Symbolic dimension rendering

Every `Dim` value that appears in loop bounds, buffer sizes, or stride
computations is rendered via `dim.to_code()`:

- `Lit(768)` → `"768"`
- `Param("T")` → `"T"`
- `Mul(Param("T"), Lit(768))` → `"(T * 768)"`

The generated `execute()` function takes symbolic dims as `i32` parameters:

```typescript
export function execute(T: i32, input_0: Float32Array, ...): Float32Array {
```

The dim params are collected by scanning all graph node shapes for
`Dim::Param` values. At runtime, the host passes the actual sequence
length as the first argument.

## Stage 4: WASM Execution

`asc` compiles the AssemblyScript to WASM. The Node.js test runner
(`test_runner_bin.mjs`) loads the WASM module, creates Float32Array inputs,
and calls `execute()`.

The manifest format supports dimension parameters:

```json
{
    "dim_params": [3],
    "inputs": [{"n_elements": 3}, {"n_elements": 38597376}, ...]
}
```

`dim_params` values are passed as the first arguments to `execute()` before
the Float32Array inputs. The old format (a plain array of input descriptors)
still works for backward compatibility.

## Performance Summary

All numbers are for full GPT-2 124M with seq_len=3:

| Metric | Before (start of session) | After |
|--------|--------------------------|-------|
| WASM execution | 17.6s | 4.5s |
| AS lines | 8,075 | 3,914 |
| ASC compile | 5.9s | 1.6s |
| Recompilation per token | yes (full rebuild) | no (symbolic T) |

The execution speedup comes from matmul tiling (A-load hoisting +
register accumulators). The compile speedup comes from kernel extraction.
The recompilation elimination comes from symbolic dimensions.

## File Map

```
crates/tensor-lang-graph/src/
    dim.rs          — Dim type: Lit/Param/Add/Mul/Div/Sub with eval, simplify, to_code
    lib.rs          — Graph, Node, Op, shape inference, DSL compiler, broadcast_shapes
    nanogpt.rs      — GPT-2 program generator (concrete and symbolic variants)

crates/tensor-lang-backend/src/
    loop_ir.rs      — Stmt, Inst, Index, ReduceDesc, TileConfig
                      lower() — fusion pass
                      tile_reduce_loops() — tiling transform
    assemblyscript.rs — AS codegen
                      emit_fused_inner() — kernel extraction + code emission
                      emit_loop() — untiled loop codegen
                      emit_tiled_loop() — tiled loop with unrolling + hoisting
                      compute_n_dependence() — instruction dependency analysis
                      normalize_body() / compute_loop_signature() — kernel dedup

crates/tensor-lang-cli/src/
    main.rs         — GPT-2 CLI: compile once with symbolic T, run with actual seq_len

test_runner_bin.mjs — Node.js WASM runner with dim_params support
```
