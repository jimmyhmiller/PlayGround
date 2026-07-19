# Memory model: what a billion nodes actually costs

This is the exact per-part byte accounting for nebula's data, an assessment of
whether each part is optimal, and the achievable floor.

**Convention.** `N` = number of nodes, `d` = average undirected degree, so the
edge count is `E = N·d/2` and the CSR adjacency has `2E` directed entries. The
concrete column is **N = 1e9, d = 4** (so `E = 2e9`, `2E = 4e9`).

Types are taken from the code: GPU buffers in `crates/nebula-render/src/scene.rs`,
CPU buffers in `crates/nebula-core/src/graph.rs`.

## GPU buffers (unified memory)

| Part | Element | Count | B/elem | Formula | @1e9, d=4 | Optimal? |
|------|---------|-------|--------|---------|-----------|----------|
| positions   | `vec2<f32>` | N     | 8 | 8N  | **8 GB**  | ✅ f32 needed for sim stability at large world coords |
| velocities  | `vec2<f32>` | N     | 8 | 8N  | **8 GB**  | ❌ fp16 → 4N; or free once settled → 0 |
| colors      | `u32` rgba8 | N     | 4 | 4N  | **4 GB**  | ❌ derived from an algorithm → compute in-shader → 0, or `u8` index → 1N |
| sizes       | `f32`       | N     | 4 | 4N  | **4 GB**  | ❌ size is a function of degree (`offsets[i+1]-offsets[i]`) → derive → 0 |
| csr_offsets | `u32`       | N+1   | 4 | 4N  | **4 GB**  | ⚠️ `u32` only reaches d≈4 (2E < 4.29e9); d≥5 needs `u64` → 8N |
| edges (flat)| `u32`       | 2E    | 4 | 8E  | **16 GB** | ❌❌ 100% redundant with CSR — render from CSR → 0 |
| csr_targets | `u32`       | 2E    | 4 | 8E  | **16 GB** | ~ floor for exact random adjacency (needs ~30 bits/target); delta+varint → ~6E |
| grid_counts | `u32`       | gridDim² (≤2048²) | 4 | fixed | 0.016 GB | ✅ fixed, tiny |
| node_order  | `u32`       | N     | 4 | 4N  | **4 GB**  | ⚠️ node ids counting-sorted by cell — this is what makes `forces` cache-coherent (3.3x at 5M). Scales with N, unlike the fixed capped list it replaced: cheaper below ~130M nodes, dearer above |
| cell_starts | `u32`       | gridDim² (≤2048²) | 4 | fixed | 0.016 GB | ✅ fixed, tiny |
| cell_cursor | `u32`       | gridDim² | 4 | fixed | 0.016 GB | ⚠️ scatter scratch; could alias cell_starts if the near field derived `start = cursor - count` |
| scan_sums   | `u32`       | ~gridDim²/256 | 4 | fixed | 0.00007 GB | ✅ two levels of block totals for the prefix sum |
| pyr (com+mass) | `vec2<u32>` | ~1.33·gridDim² | 8 | fixed | 0.045 GB | ✅ fixed; COM packed cell-relative (16-bit/axis) + mass as an exact count, so 8 B/cell not 12 |

**GPU total ≈ 32 B/node + 16E = (32 + 8d) B/node → 64 GB at d=4.**

The counting sort traded a *fixed* 0.5 GB `grid_items` (`gridDim²·cap`, mostly empty
at real occupancies) for a *per-node* 4N `node_order`. Below ~130M nodes that is a
straight win — 537 MB → 20 MB at 5M — and it buys a 3.3x layout speedup plus an exact
per-cell run instead of a 32-node sample. Past ~130M nodes it costs more than it saved,
which is the honest price of the speedup at the billion-node target. The old capped list
was not really an alternative there anyway: at 1e9 nodes it held 32 of ~238 nodes per
cell and leaned entirely on the true/sampled scaling to stay stable.

## CPU buffers (nebula-core `Graph`, resident at the same time)

The `App` keeps the `Graph` alive so it can recolor (algorithms need CSR on the
CPU today), so this is **not** freed after upload.

| Part | Element | Count | B/elem | @1e9, d=4 |
|------|---------|-------|--------|-----------|
| edges       | `[u32;2]` | E   | 8 | 16 GB |
| csr.offsets | **`u64`** | N+1 | 8 | 8 GB  |
| csr.targets | `u32`     | 2E  | 4 | 16 GB |

**CPU total ≈ 40 GB.**

## The real current number: ~104 GB at d=4

Steady state = **64 GB GPU + 40 GB CPU ≈ 104 GB**. The frequently-quoted "64 GB"
is GPU-only and understates it. The current layout is **not optimal**; the two
dominant redundancies are:

1. **The flat edge list (16 GB GPU) duplicates the CSR.** Render each edge by
   having its source node emit its CSR neighbors → the edge buffer disappears.
2. **The CPU `Graph` (40 GB) is a second full copy.** Move coloring to GPU
   compute (the algorithms are already data-parallel) or mmap CSR from disk →
   the CPU copy disappears.

Plus the cheap derived-data wins: `sizes` and `colors` are computed quantities,
and `velocities` only matter mid-simulation.

## The optimal floor

**Settled, exact rendering, GPU-side algorithms:**

```
positions  8N
csr_offsets 4N
csr_targets 8E
-----------------------------
12N + 8E = (12 + 4d) B/node  →  28 GB at d=4
```

**During simulation**, add fp16 velocities `4N` and the sort's `node_order` `4N`
→ **36 GB at d=4**. (`node_order` is not free, but it buys 3.3x on the step and
the grid it replaced cost 0.5 GB fixed regardless.)

So the honest target is **~28–36 GB optimal vs. ~104 GB today** — roughly a 3× cut,
all mechanical:

| Win | Saves @1e9, d=4 |
|-----|-----------------|
| Drop flat edge list, render from CSR | −16 GB |
| Drop CPU graph (GPU algorithms / mmap CSR) | −40 GB |
| Derive `sizes` from degree | −4 GB |
| Derive / index `colors` | −3 GB |
| fp16 or free `velocities` | −4 to −8 GB |

## Beyond d = 4

Past average degree ~4, `csr_targets` (`4d` B/node) dominates everything else and
sets the scaling wall:

| d | targets B/node | total (optimal, settled) @1e9 |
|---|----------------|-------------------------------|
| 1 | 4  | 16 GB |
| 4 | 16 | 28 GB |
| 10 | 40 | 52 GB |
| 20 | 80 | 92 GB |

At high degree the only ways forward are **adjacency compression** (delta-coded,
varint neighbor lists — roughly halves `csr_targets`) and **out-of-core tiling**:
keep only the visible region's edges resident and stream the rest. Positions
(`8N`) and offsets (`4N`) stay resident; targets page in per view. That is the
path to a genuinely dense billion-node graph on a 64 GB machine.

## Rendering, separately

Rendering cost is not the same as storage. With **screen-space aggregation** (the
density-heatmap LOD), *drawing* a graph is O(screen tiles), independent of N — a
single O(N) compute pass bins nodes into tiles and the heatmap renders in fixed
time. So "can it be shown" and "can it be held in memory" are two different
ceilings; aggregation solves the first, the table above governs the second.
