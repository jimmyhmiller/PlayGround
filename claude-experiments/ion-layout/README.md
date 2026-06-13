# Ion Graphviz Layout Plugin

A Graphviz layout engine (`dot -Kion`) that gives **any** existing Graphviz
user the layered, compiler-IR-friendly layout from
[iongraph](https://github.com/mozilla-spidermonkey/iongraph) for free: nice
columns for long edges, loop bodies indented under their headers, backedges
routed through clean right-hand channels.

> **Attribution / License.** The layout algorithm is a faithful Rust port of
> the real iongraph layout (`generic-layout/layout.ts`, the algorithm behind
> the viewer) from [iongraph](https://github.com/mozilla-spidermonkey/iongraph)
> by Ben Visness, and this project is licensed under the **Mozilla Public
> License 2.0** to match. See [`NOTICE.md`](./NOTICE.md) and
> [`LICENSE`](./LICENSE).

Graphviz still owns DOT parsing, attributes, labels, and output formats. The
plugin extracts the parsed graph, calls the Rust layout core, and writes node
coordinates and explicit edge splines back into Graphviz. All output formats
work, including `-Tsvg`, `-Tpng`, `-Tjson`, `-Txdot`, `-Tdot`, `-Tplain`.

## Usage

```sh
./scripts/build.sh
GVBINDIR=target/graphviz dot -Kion -Tsvg corpus/loops.dot > out.svg
```

No loop annotations are needed — back edges, loop headers, and loop depths
are inferred from edge direction (every feedback edge to a header becomes a
loop, like `continue` statements). Graphs produced from SpiderMonkey ion JSON
can pass explicit metadata instead via node attributes (and `rankdir`
TB/LR/BT/RL is honored — horizontal flow runs the layout in transposed
space, so ports land on node sides):

- `ion_loop_depth=<int>`
- `ion_loop_header=true`
- `ion_backedge=true`

`scripts/ion-json-to-dot.mjs` converts an ion JSON dump into such a DOT file.

## Architecture

Deep dives: [docs/ALGORITHM.md](./docs/ALGORITHM.md) (how the layout works,
phase by phase, and the Graphviz integration gotchas) and
[docs/VERIFICATION.md](./docs/VERIFICATION.md) (the test harnesses, the
parity oracle, debugging tools, and a case study on convergence-dependent
placements).

```
src/core.rs            safe layout core, a faithful port of layout.ts:
                       find loops → assign layers → build layout graph
                       (dummies + per-loop backedge channels) → straighten →
                       joints/tracks → verticalize → route. A frontend
                       classifies edges and infers loop metadata so arbitrary
                       digraphs are valid input. (pure Rust, no unsafe)
src/lib.rs             C ABI surface (ion_layout_compute / ion_layout_free_points)
plugin/gvplugin_ion.c  Graphviz layout-engine glue (registers -Kion)
include/ion_layout.h   shared C ABI header
src/bin/ion-dump.rs    stdin/stdout layout dump for the parity oracle
src/bin/stress.rs      perf/robustness stress (50k-node chains, 5k-node CFGs)
```

Edge routes are variable-length cubic bezier chains built by an orthogonal
path builder with rounded corners. Multi-layer edges follow their actual
waypoint chain (one column per layer, jogging at the track bands), so routes
never cut through nodes even when alignment passes shift individual waypoint
columns.

## Verification

This project treats layout quality as a tested property, not a vibe:

- `./scripts/check.sh` — everything below, one command.
- `cargo test` — property suite ported from upstream `generic-layout/test.ts`
  plus route-level invariants (no node overlaps, layer monotonicity, routes
  start/end on node boundaries, **no route passes through a foreign node**,
  everything inside the bounding box, determinism) across ~650 randomized
  graphs: DAGs, explicit-metadata loop graphs, and metadata-free CFGs with
  inferred loops.
- `node scripts/verify.mjs` — renders every `corpus/*.dot` through
  `dot -Kion -Tjson` and machine-checks the geometry Graphviz will actually
  draw: node overlaps, **label overflow**, edge-through-node, bounding box,
  spline/arrowhead presence.
- `./scripts/parity.sh` — the oracle: runs the original `layout.ts` (the
  real iongraph algorithm) and this port on identical graphs (fixed CFG
  shapes, the bundled demo CFG, 100 randomized in-domain graphs) and
  requires **byte-exact node geometry**. Deliberate improvements are
  detected and tagged instead of failing (see below).
- `./scripts/gallery.sh` — side-by-side `-Kion` vs `-Kdot` HTML gallery of the
  whole corpus for human review (`target/gallery/index.html`).
- `target/compare/index.html` — ours vs the **original iongraph renderer** on
  real SpiderMonkey functions from mega-complex.json (synced scrolling).

Details, including the byte-exact parity results on 281 real compiler graphs
and the debugging environment variables, are in
[docs/VERIFICATION.md](./docs/VERIFICATION.md).

## Deliberate deviations from upstream layout.ts

The port is byte-exact against `layout.ts` except where the original would
misbehave on inputs it was never designed for (it assumes well-formed
SpiderMonkey CFGs); all of these are covered by tests:

1. **Port compression / node widening** — output ports are compressed to fit
   inside their node (and very narrow multi-port nodes are widened). In the
   original, ports overflow the node box and routes can start inside a
   neighboring node.
2. **Left-margin shift** — leftmost-dummy compaction has no floor; the
   drawing is shifted right so nothing renders at negative coordinates.
3. **Robustness on degraded input** — the original throws or hangs on:
   graphs with no root, headers with zero/multiple backedge preds, cycles
   whose backedge isn't marked (it infinitely recurses on one pass of its own
   bundled `tokenize-ion.json`), side entries into loop bodies, stale
   backedge layers, and overlapping inferred loops. The port classifies
   edges up front (the layout never sees a cycle it doesn't know about),
   synthesizes one backedge block per inferred loop header, extends loop
   channels to their deepest feeder, and enforces layer monotonicity with a
   fixpoint — all no-ops on well-formed input.
4. **Obstruction-aware header approach** — the horizontal arrow into a loop
   header dodges over sibling nodes instead of drawing through them (the
   HTML viewer relies on z-order to hide this; SVG output can't).
5. **Self-loops** get a real route around the node's right side (the
   original drops them).
6. **Iterative DFS everywhere** — no stack overflow on 50k-node chains.

## Known limitations / next steps

- Clusters are laid out as flat nodes (no cluster boxes).
- Edge labels are parked beside the route midpoint; long labels on busy
  graphs may still collide with other edges.
- All predecessors of a node converge on a single input port (the iongraph
  model); parallel edges overlap on their final descent.
