# How We Made GPT-2 3.7x Faster in WASM by Tiling Matmul

We have a small tensor compiler that lowers a dataflow graph to
AssemblyScript (compiled to WASM). It runs full GPT-2 inference — but at
17.6 seconds, it's painfully slow. The bottleneck is matmul. This is the
story of how we got it down to 4.7 seconds, and why the thing that actually
mattered wasn't what we expected.

## What we're starting with

Our compiler takes a graph of tensor operations (add, mul, reduce, reshape,
etc.) and fuses chains of single-consumer ops into single loops. A matmul
`A[M,K] @ B[K,N]` doesn't exist as a primitive — it decomposes into
`reshape → expand → mul → reducesum → reshape`. After fusion, all of that
collapses into one `Stmt::Loop` node in our IR:

```rust
Stmt::Loop {
    buf: 42,                      // output buffer id
    shape: vec![M, 1, N],         // output shape (reduced axis kept as 1)
    reduce: Some(ReduceDesc {
        axis: 1, size: K,         // sum over the K dimension
        op: ReduceOp::Sum,
    }),
    body: vec![                   // the fused computation
        Inst::Load { buf: 0, index: ... },  // load from A
        Inst::Load { buf: 1, index: ... },  // load from B
        Inst::Mul(0, 1),                    // multiply them
    ],
    result: 2,                    // instruction 2 produces the output
}
```

The `body` is a flat list of instructions. Each instruction can reference
earlier instructions by index — `Mul(0, 1)` means "multiply the result
of instruction 0 and instruction 1." The `result` field says which
instruction's value gets written to the output buffer.

The `index` on each Load tells the codegen how to compute a buffer offset
from the loop's dimension variables. There are two kinds:

- **`Index::Flat`** — the buffer offset equals the loop's flat iteration
  index. Only used when the buffer shape exactly matches the loop shape.
- **`Index::Strided`** — the offset is `sum(d[dim] * stride) + offset`.
  Each entry maps a loop dimension variable to a stride in the target buffer.

For example, loading from A with shape `[M, K]` in a loop with output shape
`[M, 1, N]` plus a reduce dimension K would use:

```rust
Index::Strided {
    parts: vec![(0, K), (3, 1)],  // d0 * K + d3 * 1
    offset: 0,
}
```

Here `d0` is the M dimension (stride K in A's row-major layout) and `d3` is
the reduce dimension (stride 1 — moving along a row of A). The N dimension
(`d2`) doesn't appear because A doesn't depend on which output column we're
computing.

The codegen turns this into a flat loop:

```typescript
for (let oi: i32 = 0; oi < M * N; oi++) {
  let acc: f32 = f32(0.0);
  const d0: i32 = (oi / N);    // which row of the output (M dimension)
  const d1: i32 = 0;           // reduced axis, always 0
  const d2: i32 = oi % N;      // which column of the output (N dimension)
  for (let d3: i32 = 0; d3 < K; d3++) {
    const t0: f32 = buf0[d0 * K + d3];    // A[row, k]
    const t1: f32 = buf1[d3 * N + d2];    // B[k, col]
    const t2: f32 = (t0 * t1);
    acc = acc + t2;
  }
  buf42[oi] = acc;
}
```

The dimension variables `d0, d1, d2` are computed by decomposing the flat
index `oi` back into coordinates — integer division and modulo. `d3` is the
reduce loop variable. The body instructions don't know anything about the
loop structure; they just reference `d0`, `d1`, `d2`, `d3` and let the
codegen define those however it wants.

This is important because it means **we can change how the loops are
structured without touching the body at all**. The body instructions and
their index expressions stay exactly the same. We just change how `d0`,
`d1`, `d2`, `d3` get their values.

## The access pattern problem

Look at how B is accessed in that inner loop: `buf1[d3 * N + d2]`. The
variable `d2` is fixed for a given `oi` (it's the output column). The
variable `d3` increments by 1 each iteration. So the B index jumps by
`N` each step — we're reading down a column of B.

If N is 2304 (a real GPT-2 dimension), each step jumps 2304 floats =
9,216 bytes. A CPU cache line is 64 bytes. Every single B load is a
cache miss — we're touching a new cache line each time, and by the time
we circle back, the old ones are evicted.

Meanwhile, A is accessed as `buf0[d0 * K + d3]`. Here `d0` is fixed and
`d3` increments by 1, so we're reading along a row of A — perfectly
sequential. No problem there.

But there's a subtler waste: the value `A[row, k]` is the same for every
output column in that row. We need `A[0, 5]` when computing output
column 0, and we need the exact same `A[0, 5]` when computing output
column 1, 2, 3, ... N-1. But because the outer loop iterates over `oi`
(one output element at a time), we load each A value, use it once, throw
it away, then load it again for the next column.

## Step 1: Add tiling metadata to the IR

Tiling means breaking a big loop into smaller blocks. Instead of iterating
`n` from 0 to 2304, you iterate `n_blk` from 0 to 72 (blocks of 32), and
within each block, `ni` from 0 to 32. Same values of `n`, just visited in
a different order that's friendlier to the cache.

The first code change is boring but necessary — give the IR a place to
store tiling decisions. We add a `TileConfig` struct:

```rust
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Tile size for each output dimension, plus one for the reduce dimension.
    pub tiles: Vec<usize>,
}
```

And an optional `tile` field on `Stmt::Loop`:

```rust
Loop {
    buf: usize,
    shape: Vec<usize>,
    reduce: Option<ReduceDesc>,
    body: Vec<Inst>,
    result: InstRef,
    tile: Option<TileConfig>,  // NEW — None means untiled
},
```

We set `tile: None` in `lower()`, so nothing changes yet. Every existing
test passes. This is just plumbing.

## Step 2: Decide which loops to tile

We add a transform pass that runs on the lowered IR and stamps `TileConfig`
onto loops that would benefit from tiling:

```rust
pub fn tile_reduce_loops(stmts: &mut Vec<Stmt>) {
    const TILE_M: usize = 8;
    const TILE_N: usize = 32;
    const TILE_K: usize = 32;
    const MIN_DIM: usize = 16;

    for stmt in stmts.iter_mut() {
        if let Stmt::Loop { shape, reduce: Some(reduce), tile, .. } = stmt {
            if shape.len() < 2 { continue; }

            let ndim = shape.len();
            let m = shape[ndim - 2];
            let n = shape[ndim - 1];
            let k = reduce.size;

            // Don't bother tiling tiny loops — the overhead would dominate
            if m < MIN_DIM && n < MIN_DIM && k < MIN_DIM { continue; }

            let mut tiles = Vec::with_capacity(ndim + 1);
            // Leading dims become simple batch loops (tile = full dim)
            for d in 0..ndim - 2 {
                tiles.push(shape[d]);
            }
            // Tile the last two output dims and the reduce dim
            tiles.push(TILE_M.min(m));
            tiles.push(TILE_N.min(n));
            tiles.push(TILE_K.min(k));

            *tile = Some(TileConfig { tiles });
        }
    }
}
```

A concrete example: GPT-2's first linear layer computes
`[1, 3, 768] @ [768, 2304]`. After the matmul decomposition and fusion,
the reduce loop has output shape `[1, 3, 1, 2304]` with K=768. Here:

- `shape[0] = 1` and `shape[1] = 3` are batch dimensions — tile = full size
- `shape[2] = 1` is the reduced axis output — tile = min(8, 1) = 1
- `shape[3] = 2304` is the N dimension — tile = min(32, 2304) = 32
- The reduce K = 768 — tile = min(32, 768) = 32

The tile sizes are clamped with `min` so we never try to tile bigger than
the actual dimension. This handles the M=1 case naturally — that dimension
just has one block with one iteration.

## Step 3: Wire the tiling into codegen

Two small changes. First, apply the transform after lowering:

```rust
fn emit_fused_inner(&self, graph: &Graph, debug_bounds: bool) -> String {
    let mut stmts = loop_ir::lower(graph);
    loop_ir::tile_reduce_loops(&mut stmts);  // NEW
    // ... rest of codegen
}
```

Then dispatch tiled loops to a new emitter:

```rust
Stmt::Loop { buf, shape, reduce, body, result, tile } => {
    if let (Some(reduce), Some(tile_cfg)) = (reduce.as_ref(), tile.as_ref()) {
        emit_tiled_loop(out, *buf, shape, reduce, body, *result, tile_cfg, debug);
    } else {
        emit_loop(out, *buf, shape, reduce.as_ref(), body, *result, debug);
    }
}
```

If a loop doesn't have tiling (either because it's not a reduce loop, or it
was too small to tile), it falls through to the original codegen unchanged.

## Step 4: The naive tiled loop — and why it barely helps

The basic idea of tiling: instead of one flat loop over all N output
columns, break it into blocks of 32. Within each block, sweep through all
K values. This means B accesses become `B[k, n_blk*32 + 0], B[k, n_blk*32 + 1],
..., B[k, n_blk*32 + 31]` — 32 consecutive floats per K step, fitting in
two cache lines.

The naive implementation generates:

```typescript
// Batch loops: one per leading dimension
for (let d0: i32 = 0; d0 < 1; d0++) {
for (let d1: i32 = 0; d1 < 3; d1++) {

  // Block loops over M and N
  for (let m_blk: i32 = 0; m_blk < 1; m_blk++) {  // ceil(1/8) = 1
  for (let n_blk: i32 = 0; n_blk < 72; n_blk++) { // ceil(2304/32) = 72

    // K block loop
    for (let k_blk: i32 = 0; k_blk < 24; k_blk++) { // ceil(768/32) = 24
      for (let mi: i32 = 0; mi < 1; mi++) {
        for (let ki: i32 = 0; ki < 32; ki++) {
          const d2: i32 = m_blk * 8 + mi;      // = 0 always
          const d4: i32 = k_blk * 32 + ki;     // reduce dim
          for (let ni: i32 = 0; ni < 32; ni++) {
            const d3: i32 = n_blk * 32 + ni;   // output column

            // Body — IDENTICAL to the untiled version
            const t0: f32 = buf0[d1 * 768 + d4];   // A load
            const t1: f32 = buf1[d4 * 2304 + d3];  // B load
            const t2: f32 = (t0 * t1);

            // Accumulate into output buffer directly
            buf42[_row_base + d3] = buf42[_row_base + d3] + t2;
          }
        }
      }
    }
  }}
}}
```

Notice how the body instructions are byte-for-byte the same as the untiled
version. The only difference is how `d2`, `d3`, `d4` get computed — from
`block * tile_size + offset` instead of dividing/modding a flat index.

**Result: 17.6s → ~15s.** Disappointing.

Why? Look at the accumulate line:
`buf42[_row_base + d3] = buf42[_row_base + d3] + t2`

That's a memory read, an add, and a memory write — on every single inner
iteration. In the untiled version, `acc` was a local variable that the
WASM JIT puts in a CPU register. Register add is ~1 cycle. Memory
read-modify-write, even for L1-cached data, is several cycles with
potential pipeline stalls.

We could use a temporary array for the accumulators, but array access in
WASM still goes through memory — it's not a register.

## Step 5: Figure out what depends on N

Here's an observation about the matmul body:

```
t0 = buf0[d1 * 768 + d4]    // A load — uses d1 (batch) and d4 (reduce)
t1 = buf1[d4 * 2304 + d3]   // B load — uses d4 (reduce) and d3 (N)
t2 = t0 * t1                // depends on t1, so depends on d3
```

`t0` doesn't reference `d3` at all. It only depends on the batch dimension
and the reduce dimension. Within the inner `ni` loop, `d1` and `d4` are
fixed — only `d3` changes. So `t0` is the same for all 32 iterations of
`ni`. We're loading it 32 times and getting the same value every time.

If we could compute `t0` once and reuse it across all 32 `ni` steps,
that's 32x fewer A loads.

We need a way to tell which instructions care about `d3` and which don't.
It's a simple forward dataflow analysis:

```rust
fn compute_n_dependence(body: &[Inst], n_dim: usize) -> Vec<bool> {
    let mut depends_on_n = vec![false; body.len()];
    for (j, inst) in body.iter().enumerate() {
        depends_on_n[j] = match inst {
            // A Load depends on N if its index formula references d_n
            Inst::Load { index, .. } => match index {
                Index::Strided { parts, .. } =>
                    parts.iter().any(|(dim, _)| *dim == n_dim),
                Index::Flat => true, // conservative — might depend on anything
            },
            // A DimVar is N-dependent if it IS the N dimension
            Inst::DimVar(d) => *d == n_dim,
            // Constants never depend on N
            Inst::Const(_) => false,
            // Unary ops inherit from their input
            Inst::Neg(a) | Inst::Recip(a) | Inst::Exp2(a)
            | Inst::Log2(a) | Inst::Sqrt(a) => depends_on_n[*a],
            // Binary ops depend on N if either input does
            Inst::Add(a, b) | Inst::Mul(a, b)
            | Inst::Max(a, b) | Inst::CmpLt(a, b) =>
                depends_on_n[*a] || depends_on_n[*b],
        };
    }
    depends_on_n
}
```

For our matmul body, this produces `[false, true, true]` — instruction 0
(the A load) is N-invariant; instructions 1 and 2 depend on N.

This analysis is general. It doesn't know it's looking at a matmul. If the
fused body were something like `(A * B) + C` where C is broadcast over N,
the C load would also be flagged as N-invariant and get hoisted. The
analysis works on any fused body, not just simple matmul.

## Step 6: Unroll and hoist — where the speed actually comes from

Now we combine three ideas:

1. **Unroll** the `ni` loop into explicit copies — instead of a loop over
   `ni = 0..32`, emit the body 32 times with `d3 = ni_base + 0`,
   `d3 = ni_base + 1`, etc.

2. **Use scalar accumulators** — declare `let _acc0, _acc1, ..., _acc31`
   as local variables. WASM locals live in registers, not memory. Each
   unrolled copy accumulates into its own scalar.

3. **Hoist N-invariant instructions** — emit them once before the unrolled
   copies. Each unrolled copy only contains the N-dependent instructions.

The generated code:

```typescript
// Declare 32 scalar accumulators — these become WASM registers
let _acc0: f32 = f32(0.0);
let _acc1: f32 = f32(0.0);
// ... _acc2 through _acc31 ...

for (let k_blk: i32 = 0; k_blk < 24; k_blk++) {
  for (let ki: i32 = 0; ki < 32; ki++) {
    const d4: i32 = k_blk * 32 + ki;

    // HOISTED — emitted once, reused 32 times
    const t0: f32 = unchecked(buf0[d1 * 768 + d4]);  // A load

    // UNROLLED — only the N-dependent instructions, 32 copies
    {
      const d3: i32 = ni_base + 0;
      const t1: f32 = unchecked(buf1[d4 * 2304 + d3]);
      const t2: f32 = (t0 * t1);
      _acc0 = _acc0 + t2;        // register add, not memory!
    }
    {
      const d3: i32 = ni_base + 1;
      const t1: f32 = unchecked(buf1[d4 * 2304 + d3]);
      const t2: f32 = (t0 * t1);
      _acc1 = _acc1 + t2;
    }
    // ... 30 more blocks ...
  }
}

// Write registers to memory once at the end
unchecked(buf42[_row_base + ni_base + 0] = _acc0);
unchecked(buf42[_row_base + ni_base + 1] = _acc1);
// ... 30 more ...
```

Each `{ ... }` block creates a new scope so the `d3`, `t1`, `t2` variable
names don't conflict between copies. But `t0` and `_acc0` live in the
outer scope — `t0` is readable from inside the blocks, and `_acc0` is a
`let` binding that gets mutated.

What happens when N isn't divisible by 32? (For example, a matmul with
N=37.) The unrolled portion handles the first `floor(N/32) * 32 = 32`
elements in groups of 32. A scalar fallback loop handles the remaining
5 elements one at a time:

```typescript
for (let ni: i32 = 32; ni < 37; ni++) {
  const d3: i32 = n_blk * 32 + ni;
  let _acc_r: f32 = f32(0.0);
  for (let k_blk: i32 = 0; k_blk < 24; k_blk++) {
    for (let ki: i32 = 0; ki < 32; ki++) {
      const d4: i32 = k_blk * 32 + ki;
      const t0: f32 = unchecked(buf0[d1 * 768 + d4]);
      const t1: f32 = unchecked(buf1[d4 * 2304 + d3]);
      const t2: f32 = (t0 * t1);
      _acc_r = _acc_r + t2;
    }
  }
  unchecked(buf42[_row_base + n_blk * 32 + ni] = _acc_r);
}
```

This remainder loop is slower (no hoisting, processes one element at a
time) but only runs for at most 31 elements per tile, which is negligible.

## Step 7: Drop the bounds checks

One last thing: AssemblyScript normally inserts a bounds check on every
array access — it verifies that the index is >= 0 and < array.length before
reading or writing. That's an extra comparison and branch per access.

In our generated code, the indices are guaranteed in-bounds by construction
(they come from loop bounds that match the buffer sizes). So we wrap
everything in `unchecked()`:

```typescript
// Without: two branches per access (index >= 0, index < length)
const t0: f32 = buf0[d1 * 768 + d4];

// With: direct memory access, no branches
const t0: f32 = unchecked(buf0[d1 * 768 + d4]);
```

We apply this to both tiled and non-tiled loops (when not in debug mode).

## What actually mattered

Here's the performance timeline as we added each piece:

| What changed | WASM runtime | vs baseline |
|---|---|---|
| Baseline (flat loops) | 17.6s | 1.0x |
| Blocked loops only (naive tiling) | ~15.0s | 1.2x |
| + unchecked() everywhere | ~15.0s | 1.2x |
| + Scalar unroll ×4 (no hoisting) | 14.4s | 1.2x |
| + A-load hoisting (×4 unroll) | 7.7s | 2.3x |
| Hoisting + ×8 unroll | 6.1s | 2.9x |
| Hoisting + ×16 unroll | 5.3s | 3.3x |
| Hoisting + ×32 unroll | 4.6s | 3.8x |

The blocked loop structure on its own was worth almost nothing. The
`unchecked()` optimization made no measurable difference on top of that.
Even the unrolling didn't help much without hoisting — the ×4 unroll
duplicated the body (including the redundant A loads), bloating the WASM
binary and confusing the JIT optimizer.

**Hoisting was the single biggest win.** It turned each inner-loop
iteration from "2 loads + 1 multiply + 1 accumulate" into "1 load + 1
multiply + 1 accumulate" — cutting the load traffic nearly in half.
The V8 JIT couldn't do this optimization itself because the body copies
were in separate block scopes with `const` declarations.

The unroll factor then scaled the hoisting benefit: with ×4 unroll, each
A value is reused 4 times. With ×32, it's reused 32 times. The
diminishing returns above ×32 suggest that 32 WASM locals for the
accumulators is near the limit of what V8 keeps in physical registers
before spilling to the stack.

## Why these three things are inseparable

You can't get the hoisting benefit without tiling. In the untiled loop,
each output element independently sweeps all K values. There is no inner
N loop — `d_n` is computed once from `oi % N` and stays fixed. The A load
`buf0[d0*K + d3]` changes every iteration (because `d3` changes), so
there's nothing to hoist.

Tiling creates the loop nest `for ki { for ni { ... } }` that makes the
A load invariant across `ni`. But without unrolling, you'd accumulate
into an array (because you have 32 in-flight sums), which is still a
memory access per iteration. And without hoisting, each of the 32
unrolled copies would redundantly re-load A.

All three pieces have to be there. Tiling creates the structure. Hoisting
exploits the structure to eliminate redundant loads. Unrolling exploits
the structure to keep accumulators in registers.

## The files

- **`loop_ir.rs`**: `TileConfig` struct, `tile` field on `Stmt::Loop`,
  `tile_reduce_loops()` transform pass
- **`assemblyscript.rs`**: `emit_tiled_loop()`, `compute_n_dependence()`,
  `emit_body_unchecked()`, dispatch logic in `emit_fused_inner()`
- **`fused_oracle.rs`**: `test_fused_matmul_non_divisible_tile` — test
  with [7,41]@[41,37] where no dimension divides the tile sizes evenly
