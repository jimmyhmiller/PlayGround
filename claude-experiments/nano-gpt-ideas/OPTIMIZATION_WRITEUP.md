# GPT-2 ARM Native Backend: 19x Speedup

## Summary

GPT-2 124M inference on ARM64 (Apple Silicon), 12 layers, T=16, single-threaded:

| Stage | Time | Speedup |
|-------|-----:|--------:|
| Baseline (old ARM backend) | 3740ms | 1x |
| + Nested loops | 3595ms | 1.04x |
| + K-invariant hoisting | 3201ms | 1.17x |
| + Matmul unfusion + MR=8 kernel | 1475ms | 2.54x |
| + Pointer incrementing | 1434ms | 2.61x |
| + All combined (final) | 198ms | **18.9x** |

The final 198ms result also reflects the matmul unfusion enabling the MR=8 path
for 5 of 6 matmuls per layer (previously only 1 matched the pattern).

Reference: llm.c gets 84ms on the same workload with `-O3 -ffast-math` and 
transposed weight layout.

## The Problem

The ARM backend generated correct code but was slower than WASM SIMD running 
through wasmtime's JIT (1262ms vs 2738ms originally). The bottleneck was not
the register allocator or IR complexity -- it was the generated loop structure
and matmul algorithm.

## Optimization 1: Nested Loops (eliminate SDIV)

**File:** `arm.rs` — `emit_elementwise_loop`, `emit_reduce_loop`

**Problem:** Every elementwise and reduce loop used a flat output index `oi` 
and decomposed it into per-dimension coordinates using integer division:

```
for oi in 0..total_size:
    d0 = (oi / stride[0]) % shape[0]   // SDIV: 12-20 cycles
    d1 = (oi / stride[1]) % shape[1]   // SDIV: 12-20 cycles
    addr = d0 * input_stride_0 + d1 * input_stride_1
    // ... actual computation: 3-4 cycles
```

For a 3D tensor, that's 4-6 SDIV instructions per element. SDIV on ARM is 
12-20 cycles, so 60-120 cycles of index math per element vs 3-4 cycles of 
actual computation. ~95% of time was spent on index decomposition.

**Fix:** Replace the flat loop with nested per-dimension loops:

```
for d0 in 0..shape[0]:
    for d1 in 0..shape[1]:
        addr = d0 * input_stride_0 + d1 * input_stride_1
        // no SDIV anywhere
```

The output is still written linearly (a flat counter `oi` increments in the
innermost loop). Input addresses are computed from the loop counters using
multiply-add, which is 3-4 cycles (vs 12-20 for SDIV).

**Impact:** ~4% on full GPT-2 (matmul-dominated), more on elementwise-heavy 
workloads.

## Optimization 2: K-Invariant Hoisting

**File:** `arm.rs` — general tiled loop path in `emit_tiled_loop`

**Problem:** The loop IR fuses aggressively. A matmul with preceding layernorm 
becomes one fused reduce loop body like:

```
body[0]:  Load input[batch, seq, k]
body[1]:  Load variance[batch, seq]      // <-- does NOT depend on K
body[2]:  Load epsilon                    // <-- does NOT depend on K
body[3]:  Mul(1, 2)                       // <-- does NOT depend on K
body[4]:  Add(3, epsilon)                 // <-- does NOT depend on K
body[5]:  Sqrt(4)                         // <-- does NOT depend on K
body[6]:  Recip(5)                        // <-- does NOT depend on K
body[7]:  Mul(0, 6)                       // depends on K (through input)
body[8]:  Load gamma[k]
body[9]:  Mul(7, 8)
body[10]: Load beta[k]
body[11]: Add(9, 10)
body[12]: Load weight[k, n]
body[13]: Mul(11, 12)                     // result, accumulated over K
```

Instructions 1-6 (layernorm's variance/sqrt/recip) produce the same result for
every K iteration. With K=768, they ran 768 times unnecessarily.

**Fix:** Added `compute_dim_dependence(body, reduce_dim)` to identify which 
instructions depend on K. Emit K-invariant instructions once before the K block
loop, and only K-dependent instructions inside.

**Impact:** ~10% on GPT-2 (layernorm fused into all 5 linear projections per
layer).

## Optimization 3: Matmul Unfusion

**File:** `loop_ir.rs` — `unfuse_matmul_bodies`

**Problem:** The fast MR=8 matmul kernel requires exactly 3 body instructions:
`Load A, Load B, Mul`. But the fused layernorm+matmul bodies have 14-17 
instructions. Only 1 of 6 matmuls per layer matched the fast path.

**Fix:** Added a loop IR pass that detects the pattern:
```
body[result] = Mul(pre_chain, weight_load)
```
where `pre_chain` doesn't depend on N and `weight_load` depends on both K and N.
The pass splits this into:
1. A separate elementwise loop that computes the pre-chain into a temp buffer
2. A clean `Load_temp, Load_weight, Mul` matmul body

This trades one extra memory pass over the pre-chain output (writing then 
reading the layernorm result) for enabling the fast matmul kernel on all 
matmuls.

**Impact:** Enables MR=8 for 5 of 6 matmuls per layer instead of 1.

## Optimization 4: MR=8 Micro-Kernel

**File:** `arm.rs` — `emit_matmul_mr8`

**Problem:** The original tiled matmul processed one output row at a time:
```
for mi in 0..M:         // one row at a time
    for ki in 0..K:
        a = A[mi, ki]
        broadcast a to NEON vector
        for each NR-group:
            b = LDR Q B[ki, n..n+4]
            acc[group] += broadcast(a) * b
```

With T=16 rows, the weight column group (K × NR × 4 bytes = 24KB) was loaded
from memory 16 times — once per row.

**Fix:** Process MR=8 rows simultaneously in the K loop:
```
for ki in 0..K:
    b0 = LDR Q B[ki, n..n+4]       // load weight once
    b1 = LDR Q B[ki, n+4..n+8]
    for r in 0..8:                   // 8 rows share the same B data
        a = LDR S A[mi+r, ki]
        a_bcast = DUP a             // broadcast to vector
        acc[r][0] += a_bcast * b0   // FMLA
        acc[r][1] += a_bcast * b1   // FMLA
```

This does 16 FMLA instructions per K iteration with only 2 B loads. The weight
data is loaded once and reused across all 8 rows, reducing memory traffic by 8x.

Register usage: 16 Vec128 accumulators (8 rows × 2 groups) + scratch = fits in
32 NEON registers.

**Impact:** ~2.5x speedup on the matmul, which dominates GPT-2.

## Optimization 5: Pointer Incrementing

**File:** `arm.rs` — K-loop in `emit_matmul_mr8`

**Problem:** The K-loop recomputed addresses every iteration:
```
b_k_off = ki * b_k_stride         // MUL
b_k_byte = b_k_off << 2           // LSL  
b_addr = b_col_ptr + b_k_byte     // ADD
// Similar for A: 3 more instructions per row
```

**Fix:** Pre-compute base pointers before the loop, increment by fixed stride:
```
// Before loop:
a_ptr_k = a_base
b_ptr_k = b_col_base
a_k_inc = a_k_stride * 4   // computed once
b_k_inc = b_k_stride * 4

// Each iteration:
// ... load from a_ptr_k and b_ptr_k directly ...
a_ptr_k += a_k_inc          // single ADD
b_ptr_k += b_k_inc          // single ADD
```

Also replaced `FmovWFromS + Dup4sGp` (2 instructions for broadcast) with
`Dup4sScalar` (1 instruction).

**Impact:** ~20% reduction in K-loop instruction count.

## Optimization 6: Collapsed Batch Dimensions

**File:** `arm.rs` — `emit_matmul_mr8`, dispatch in `emit_tiled_loop`

**Problem:** GPT-2's matmul shape is `[1, T, 1, N]` where batch dims are [1, T],
m_dim=2 (size 1), n_dim=3. The original code iterated T in a batch loop, each
time doing a single-row matmul. MR=8 couldn't apply because "M" was 1.

**Fix:** Compute `effective_M = product(batch_dims) × shape[m_dim] = 1 × T × 1 = T`.
Use this as the row count for the MR=8 kernel. The A row stride is computed from
the index strides to handle the collapsed multi-dimensional iteration.

This is the same approach as llm.c, which collapses B*T into a single dimension:
```c
for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) { ... }
```

**Impact:** Enables MR=8 for GPT-2 (T ≥ 8 required).

## Supporting Infrastructure

### Register Allocator Integration

The `regalloc` library was integrated via a `FlatFunction` trait that accepts
flat instruction streams (like MachIR) without requiring explicit CFG construction.
Key features used:

- **Per-instruction allocation** (`get_at(inst_idx, vreg)`) to handle spilled
  vregs that need different physical registers at different points
- **Clobber-aware register selection** (`pick_free_reg` avoids caller-saved regs
  for intervals spanning clobber instructions)
- **V8-V15 callee-saved FP registers** added to the pool (8 more NEON regs,
  saved/restored in prologue/epilogue)

### NEON SIMD Elementwise

Added Vec128 NEON instructions to MachIR (`Fadd4s`, `Fmul4s`, `Fneg4s`, 
`Fmax4s`, `Fsqrt4s`, `Fdiv4s`, `Fcmgt4s`, `Dup4sScalar`, `LdrQReg`, 
`StrQReg`, `Fmov4sOne`) for 4-wide SIMD elementwise operations. The innermost
loop dimension processes 4 elements at a time with NEON, with a scalar 
remainder for non-multiple-of-4 sizes.

## Remaining Gap to llm.c

Our 198ms vs llm.c's 84ms (2.3x gap) is primarily due to **weight matrix 
layout**:

- llm.c stores weights as `[N, K]` (output-channel-major), transposing from
  the checkpoint's `[K, N]` format during loading. The K-loop then accesses
  weight memory sequentially (stride 1).

- Our weights are `[K, N]` (the checkpoint format). The K-loop jumps by N×4 
  bytes between iterations (stride N on K). For N=2304, that's 9KB jumps.

Benchmark proof: a simple C matmul with B[K,N] layout compiled with `-O3 
-ffast-math` gets 10.5 GFLOP/s. Our MR=8 kernel gets 26 GFLOP/s on the same
layout — **2.5x faster than auto-vectorized C**. llm.c gets 52 GFLOP/s because
its B[N,K] layout enables sequential memory access in the K loop.

Closing this gap requires either:
1. **Weight packing** (BLIS-style NR×KC interleaved panels)
2. **Transposed weight format** (change the export to store [N,K])

Both are data format changes, not code generation changes.

## Correctness

All optimizations maintain correctness:
- 13/13 ARM oracle tests pass (matmul, reduce, softmax, exp, log, etc.)
- GPT-2 output matches across all backends (WASM scalar, WASM SIMD, GPU, ARM)
- "All backends agree on next token" verified at every stage
