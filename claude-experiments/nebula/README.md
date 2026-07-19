# nebula

A GPU-accelerated, native graph viewer written in Rust + [wgpu](https://wgpu.rs/).
Built to lay out and render *very* large graphs interactively — the layout physics
and the rendering both run on the GPU, and node positions never leave GPU memory
between simulating and drawing.

On an Apple M2 Max it lays out **5,000,000 nodes / 50,000,000 edges with full
force-directed physics at ~40 simulation steps per second** (24.8 ms/step, GPU
otherwise idle), and renders **10,000,000 nodes at 30–40 fps**. It aims at the
billion-node regime by architecture (see [Scaling](#scaling-to-a-billion-nodes)).

Two caveats on that number, because a step rate is easy to quote misleadingly.
Step cost rises about **1.4x** once a layout contracts into clusters (dense cells
mean more near-field neighbours per node) — the figure above is for the evenly
spread state. And *settling* is not one step: a graph this size is given ~3,660
steps (see [cooling](#converging-and-stopping)), so a full 5M layout takes minutes,
not seconds. It is interactive throughout, and `space` stops it whenever it looks
right.

```
cargo run --release -p nebula-app -- --blocks 40000 6 300000 --color communities
```

## What it does

- **GPU force-directed layout that converges and stops.** A Fruchterman–Reingold
  simulation runs entirely in compute shaders. Repulsion is approximated with a
  uniform spatial grid for the near field plus a center-of-mass pyramid
  (fast-multipole-style interaction lists) for the far field — this is what lets a
  grid unfold into a flat sheet and lets communities separate, instead of
  everything collapsing into a ball. Nodes are counting-sorted into their grid
  cells every step so the force pass walks them in spatial order: the gathers hit
  cache instead of stalling, which is worth ~3x at 5M nodes. A global **alpha**
  cooling factor scales the forces down to zero (d3-force style) so the layout
  settles and auto-pauses; `space` reheats it, `R`/`G`/`O` restart from a fresh
  seed.
- **GPU rendering.** Nodes are instanced SDF circles (crisp at any zoom); edges
  are additively-blended lines so dense regions read as luminous bundles.
  Positions are shared by *binding* between the compute and render passes — zero
  CPU round-trips.
- **Click a node to inspect it.** GPU id-picking selects the node under the
  cursor; an in-engine panel shows its label, degree, neighbors, algorithm value,
  and position, and its connections are drawn as highlighted links to ringed
  neighbors. Node size is adjustable at runtime (`+`/`-`, or `--node-size`) so
  nodes are easy to hit.
- **Aggregation for graphs too big to draw.** A density heatmap LOD (`A`) bins
  every node into screen tiles in one O(N) compute pass and renders the *tiles*,
  not the nodes — so rendering is O(screen), independent of N. A 20M-node graph
  draws at ~210 fps aggregated (vs ~64 fps node-by-node). Auto-enables past 2M
  nodes.
- **Color by graph algorithm.** Connected components, degree, PageRank, greedy
  proper coloring, and label-propagation communities — all computed on the CPU in
  parallel (rayon) and mapped to perceptual palettes (turbo for scalars, golden-
  angle hues for categories).
- **Loads many formats.** Edge lists, CSV, Matrix Market (`.mtx`), DIMACS
  (`.gr`), GML, DOT/Graphviz, JSON (d3-style `{nodes, links}` or `{edges}`), and
  adjacency lists — auto-detected by extension and content.
- **Typed edges.** JSON may declare multiple named, colored edge sets via
  `{nodes, edge_types:[{name, color, edges}]}` (e.g. `children`/`deps`). Each
  set toggles and colors independently, and a "compute over" selector chooses
  which set the coloring/layout algorithms use. `scripts/tp-to-nebula.py`
  converts a Turbopack task-graph dump into this format.
- **Pluggable layouts.** A `Layout` trait with random / circle / grid / force-
  directed implementations; re-seed at runtime with `R` / `G` / `O`.
- **In-engine HUD.** Stats, controls, a color legend, and the inspection panel are
  drawn with an embedded 8×8 bitmap font — no UI toolkit, one wgpu version, ~zero
  extra dependencies.

## Install / run

Requires a recent Rust toolchain and a GPU (Metal / Vulkan / DX12).

```bash
cargo run --release -p nebula-app -- [GENERATOR] [OPTIONS]
```

### Generators

| Flag | Meaning |
|------|---------|
| `--grid W H` | 2D lattice, `W*H` nodes |
| `--random N M` | Erdős–Rényi: N nodes, M random edges |
| `--ba N M` | Barabási–Albert scale-free: N nodes, m=M attachments (hubs) |
| `--blocks N K M` | Stochastic block model: N nodes, K communities, M edges |
| `--geo N R` | Random geometric: N nodes, connect within radius R∈(0,1) |
| `--file PATH` | Load a graph file (format auto-detected) |

Default (no generator): `--ba 50000 3`.

### Options

`--color MODE` (`uniform`/`components`/`degree`/`pagerank`/`coloring`/`communities`),
`--k FLOAT` (optimal edge length), `--seed N`, `--dt F`, `--substeps N`
(sim steps per frame), `--paused`, `--no-edges`, `--no-nodes`,
`--frames N` + `--screenshot PATH` (headless capture), `--select INDEX`,
`--help-overlay`.

### Controls

```
drag            pan                 scroll          zoom
space           pause/resume        F               fit view
click a node    inspect it          double-click    focus its neighborhood
C               clear selection     A               aggregate (density LOD)
1..6            color: uniform / components / degree / pagerank / coloring / communities
R / G / O       re-seed random / grid / circle
E / N           toggle edges / nodes            L    labels        S   save screenshot
+ / -           node size           [ / ]           edge brightness
H               help overlay        Tab             toggle HUD          Esc  quit
```

## Examples

```bash
# Six communities that separate into distinct clusters (SBM + label propagation)
cargo run --release -p nebula-app -- --blocks 40000 6 300000 --color communities

# Scale-free hubs, sized and colored by PageRank
cargo run --release -p nebula-app -- --ba 40000 2 --color pagerank

# A 200x200 grid unfolding into a flat sheet
cargo run --release -p nebula-app -- --grid 200 200 --color degree

# A random geometric graph recovering its planar mesh
cargo run --release -p nebula-app -- --geo 30000 0.015

# Load a real graph (SNAP/Konect edge lists, .mtx, .gml, .dot, .json, ...)
cargo run --release -p nebula-app -- --file mygraph.mtx --color components

# Headless render to a PNG
cargo run --release -p nebula-app -- --ba 100000 3 --frames 500 --screenshot out.png
```

## Architecture

A cargo workspace of four crates:

- **`nebula-core`** — graph representation and algorithms. The canonical form is a
  flat undirected edge list (exactly what the GPU consumes); a CSR adjacency is
  built in parallel on demand. Node ids are `u32` (up to 4.29B); edge counts are
  `u64`. Contains the generators, the multi-format loaders, and the algorithms
  (components via union-find, BFS, PageRank, greedy coloring, label propagation).
- **`nebula-layout`** — the `Layout` trait and CPU layouts, including a parallel,
  grid-accelerated force-directed reference implementation.
- **`nebula-render`** — everything GPU: device bring-up, the compute layout, the
  node/edge/pick pipelines, the overlay (bitmap-font HUD), camera, and the winit
  app loop.
- **`nebula-app`** — the CLI.

### The layout compute pipeline

Each simulation step is a chain of compute passes, each in its own pass so wgpu
inserts the right memory barriers:

1. `clear_grid` / `count_grid` — count how many nodes fall in each uniform
   spatial cell.
2. **The counting sort** — `scan_cells`, `scan_sums0`, `scan_sums1`, `add_sums0`,
   `add_cells` prefix-sum those counts (a three-stage scan, since 2048² cells is
   far past one workgroup), `init_cursor` and `scatter_nodes` then place every
   node id into its cell's run in `node_order`.
3. `build_pyr_l0` / `reduce_pyr` (per level) — build the center-of-mass pyramid
   from the cell counts. Level 0 uses cell centers, so no float atomics are needed;
   each level above merges 2×2 children.
4. `forces` — read positions (read-only), accumulate near-field repulsion (3×3
   cell neighborhood), far-field repulsion (fast-multipole-style interaction lists
   down the pyramid, ~27 cells per level), edge attraction (CSR neighbors), and
   gravity; write velocities.
5. `integrate` — advance positions from velocities.

Splitting force accumulation (positions read-only) from integration (positions
write) makes the whole step race-free without double-buffering. Dispatches are
tiled across a 2D workgroup grid so the ~4M workgroups a billion nodes need fit
under the 65,535-per-dimension limit; the shader reconstructs a linear index.

**Why sort every step.** `forces` is latency-bound, not bandwidth-bound: it moves
only a few hundred MB per step, but each gather is a cache miss and it stalls on
them one at a time. Walking nodes in *cell order* (thread `g` handles
`node_order[g]`, not node `g`) puts a workgroup's threads in neighbouring cells, so
they hit the same grid runs and pyramid entries and the cache absorbs the gathers.
It is exact — the same forces for the same nodes, only a different thread computes
each — and it is worth ~2.9x at 1M nodes. The sort also replaced a fixed-capacity
per-cell list (`dim²·cap`, 537 MB and mostly empty) with an exact run per cell in
an N-entry array.

### Converging and stopping

A global **alpha** scales every force and decays each step, so the layout cools,
settles, and auto-pauses (d3-force style). The decay rate *is* a step budget:
`steps = ln(alpha_min) / ln(1 - decay)`.

That budget scales with **√N**. Forces travel about a cell per step and a layout is
~√N cells across (`world = 1.6·k·√N`, cell ≈ k), so structure needs ~√N steps to
propagate. A fixed decay — d3's, tuned for a few hundred nodes — would freeze a 5M
graph after the same ~366 steps a 50k graph gets, leaving it visibly half-finished
and needing manual reheats. The constant is pinned so 50k still gets exactly its
~366 steps:

| nodes | steps to settle |
|-------|-----------------|
| 50,000 | 366 |
| 1,000,000 | 1,637 |
| 5,000,000 | 3,660 |

`space` reheats a settled layout (or stops a running one); `R`/`G`/`O` restart from
a fresh seed. The HUD shows how many steps remain.

## Scaling to a billion nodes

The design targets a billion nodes: `u32` ids, `u64` edge counts, structure-of-
arrays throughout, GPU-resident positions, O(N) grid repulsion, and dispatch
tiling that already handles billion-scale workgroup counts.

The honest limit today is **memory**. See [docs/MEMORY.md](docs/MEMORY.md) for the
exact per-buffer byte accounting: at average degree 4 a billion nodes costs
~64 GB on the GPU (and the current build also keeps a ~40 GB CPU copy, so ~104 GB
total — not yet optimal). The document works out the ~28-36 GB optimal floor and
the mechanical wins to get there (render edges from CSR, GPU-side algorithms to
drop the CPU copy, derive sizes/colors, fp16 velocities), plus where compression
and out-of-core tiling become necessary past degree 4. nebula demonstrates at the
tens-of-millions the machine holds and does not fake those numbers.

**Rendering is a separate ceiling from storage.** The density-aggregation LOD
draws a graph in O(screen tiles), independent of N — so "can it be shown" is
solved even where "can it be held in memory" is not.

## Performance (Apple M2 Max)

**Layout only**, measured headless with no window and nothing else on the GPU
(`cargo run --release -p nebula-render --example layout_bench -- <nodes> <edges>`),
so these isolate simulation cost from raster cost. Evenly-spread state; add ~1.4x
once the layout contracts into clusters.

| Graph | ms / step | steps / s | before the counting sort |
|-------|-----------|-----------|--------------------------|
| 500k nodes / 5M edges | 3.7 | ~270 | 6.8 ms |
| 1M nodes / 10M edges | 5.0 | ~200 | 14.4 ms |
| 2M nodes / 20M edges | 8.6 | ~116 | 26.9 ms |
| 5M nodes / 50M edges | 24.8 | ~40 | 78.5 ms |

These numbers are sensitive to GPU contention — a browser with an active GPU
process roughly halves them. Measure with the GPU idle.

**In-app** (simulation *and* rendering every frame):

| Graph | What's running | Throughput |
|-------|----------------|------------|
| 30k nodes / 220k edges | full sim + render | ~120 fps |
| 1M nodes / 3M edges | full force sim + render every frame | ~8 fps |
| 10M nodes | render only (nodes) | 30–40 fps |
| 20M nodes / 40M edges | render only (nodes) | ~60 fps |
| 50M nodes / 50M edges | render only (nodes) | loads & renders (single-digit fps) |

(50M nodes / 50M edges live in GPU buffers simultaneously — this is the tens-of-
millions ceiling of a 64 GB machine. Node rendering scales near-linearly; the
frame-time floor is memory bandwidth.)

The worst case for rendering is a large *unsettled* (random) layout: every edge
is a long, screen-crossing line and additive blending makes overdraw dominate.
Settled layouts render far faster.

## Tests

```bash
cargo test
```

Covers CSR construction, generators, all six algorithms, every format parser, the
camera math, and the CPU force layout.
