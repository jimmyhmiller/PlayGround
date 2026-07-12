# nebula

A GPU-accelerated, native graph viewer written in Rust + [wgpu](https://wgpu.rs/).
Built to lay out and render *very* large graphs interactively — the layout physics
and the rendering both run on the GPU, and node positions never leave GPU memory
between simulating and drawing.

On an Apple M2 Max it lays out **1,000,000 nodes / 3,000,000 edges with full
force-directed physics every frame at ~8 fps**, and renders **10,000,000 nodes at
30–40 fps**. It aims at the billion-node regime by architecture (see
[Scaling](#scaling-to-a-billion-nodes)).

```
cargo run --release -p nebula-app -- --blocks 40000 6 300000 --color communities
```

## What it does

- **GPU force-directed layout.** A Fruchterman–Reingold simulation runs entirely
  in compute shaders. Repulsion is approximated with a uniform spatial grid for
  the near field plus a coarse center-of-mass grid (a single-level Barnes–Hut)
  for the far field — this is what lets a grid unfold into a flat sheet and lets
  communities separate, instead of everything collapsing into a ball.
- **GPU rendering.** Nodes are instanced SDF circles (crisp at any zoom); edges
  are additively-blended lines so dense regions read as luminous bundles.
  Positions are shared by *binding* between the compute and render passes — zero
  CPU round-trips.
- **Click a node to inspect it.** GPU id-picking selects the node under the
  cursor; an in-engine panel shows its label, degree, neighbors, algorithm value,
  and position, and its connections are drawn as highlighted links to ringed
  neighbors.
- **Color by graph algorithm.** Connected components, degree, PageRank, greedy
  proper coloring, and label-propagation communities — all computed on the CPU in
  parallel (rayon) and mapped to perceptual palettes (turbo for scalars, golden-
  angle hues for categories).
- **Loads many formats.** Edge lists, CSV, Matrix Market (`.mtx`), DIMACS
  (`.gr`), GML, DOT/Graphviz, JSON (d3-style `{nodes, links}` or `{edges}`), and
  adjacency lists — auto-detected by extension and content.
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
click a node    inspect it          C               clear selection
1..6            color: uniform / components / degree / pagerank / coloring / communities
R / G / O       re-seed random / grid / circle
E / N           toggle edges / nodes
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

Each simulation step is four (well, five) compute passes, each in its own pass so
wgpu inserts the right memory barriers:

1. `clear_grid` — zero the per-cell counters.
2. `build_grid` — bucket each node into a uniform spatial cell (atomic append,
   capacity-clamped to a fixed per-cell sample).
3. `build_coarse` — reduce the fine grid into a coarse center-of-mass grid (no
   float atomics needed: coarse mass is derived from the fine cell counts, using
   fine-cell centers as mass locations).
4. `forces` — read positions (read-only), accumulate near-field repulsion (fine
   3×3 neighborhood, scaled by *true* cell population so dense cells don't
   saturate), far-field repulsion (coarse COM grid), edge attraction (CSR
   neighbors), and gravity; write velocities.
5. `integrate` — advance positions from velocities.

Splitting force accumulation (positions read-only) from integration (positions
write) makes the whole step race-free without double-buffering. Dispatches are
tiled across a 2D workgroup grid so the ~4M workgroups a billion nodes need fit
under the 65,535-per-dimension limit; the shader reconstructs a linear index.

## Scaling to a billion nodes

The design targets a billion nodes: `u32` ids, `u64` edge counts, structure-of-
arrays throughout, GPU-resident positions, O(N) grid repulsion, and dispatch
tiling that already handles billion-scale workgroup counts.

The honest limit today is **memory**. A billion nodes with a modest average
degree needs on the order of 64 GB just for positions, velocities, and edges —
i.e. true 1B is out-of-core territory on a 64 GB machine. nebula demonstrates at
the tens-of-millions the machine holds and is architected for the streaming /
memory-mapped / tiled approach that closes the rest of the gap. It does not fake
those numbers.

## Performance (Apple M2 Max)

| Graph | What's running | Throughput |
|-------|----------------|------------|
| 1M nodes / 3M edges | full force sim + render every frame | ~8 fps |
| 10M nodes | render only (nodes) | 30–40 fps |
| 30k nodes / 220k edges | full sim + render | ~120 fps |

The worst case for rendering is a large *unsettled* (random) layout: every edge
is a long, screen-crossing line and additive blending makes overdraw dominate.
Settled layouts render far faster.

## Tests

```bash
cargo test
```

Covers CSR construction, generators, all six algorithms, every format parser, the
camera math, and the CPU force layout.
