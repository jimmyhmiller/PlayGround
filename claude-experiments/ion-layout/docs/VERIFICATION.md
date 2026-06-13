# Verification: how we know the layouts are right

Layout quality here is a tested property, not a vibe. Three independent
harnesses cross-check each other, plus visual tooling for human review.

## 1. Property tests — `cargo test`

`tests/properties.rs`, ported from upstream `generic-layout/test.ts` and
extended. ~650 graphs per run: fixed CFG shapes (chains, diamonds, loops,
nested loops, self-loops, multi-edges, disconnected, irreducible),
50–300-seed randomized DAGs, explicit-metadata loop graphs, and
metadata-free CFGs exercising loop inference. Invariants:

- no two node boxes overlap; same layer ⇒ same top y; layers strictly
  descend; everything inside the bounding box; positive coordinates
- layer monotonicity for forward edges; backedge blocks on their header's
  layer
- every edge gets a valid bezier route (3k+1 control points); routes start
  on the tail's boundary and the arrow tip lands on the head's boundary
- **no route passes through the interior of a foreign node** (each cubic is
  sampled; interiors shrunk 2pt)
- determinism: two runs produce identical output
- orientations: LR is the exact transpose of the size-swapped TB run; BT the
  exact mirror

## 2. Rendered-geometry checks — `node scripts/verify.mjs`

The property tests validate the core; this validates **what Graphviz will
actually draw**. Every `corpus/*.dot` (26 files: stress shapes, unicode,
long labels, clusters, undirected, rankdir variants, a real 150-block
SpiderMonkey graph, generated 120-node CFGs) is rendered with
`dot -Kion -Tjson` and machine-checked:

- C1 node-node overlap, C4 bounding box containment
- C2 label text fits inside its node (Graphviz's own text metrics)
- C3 edge splines sampled against all foreign node interiors
- C5 every edge has a spline and (when directed) an arrowhead
- C6 the **drawn shape polygon** matches the node's declared geometry — this
  exists because `-Tjson` reports `ND_width` while renderers draw the
  polygon, and a stale polygon once shipped boxes detached from their edges

`./scripts/check.sh` runs builds + 1 + 2.

## 3. The parity oracle — `./scripts/parity.sh`

The strongest guarantee: `scripts/parity.ts` runs the ORIGINAL
`generic-layout/layout.ts` (via tsx, from the iongraph repo) and this port
on **identical inputs** and requires **byte-exact node geometry** (position,
size, layer for every block). Cases:

- fixed CFG shapes and the iongraph demo CFG (`graph.json`)
- 100 randomized graphs (DAGs + explicit loop graphs)
- **mega-complex.json: every pass of every function** of a real 9.7MB
  SpiderMonkey dump — 281 real compiler graphs up to 69 blocks

Current standing: **362 exact, 0 failures**; the only non-exact cases are
auto-tagged deliberate deviations (below). All 281 real ion graphs match
byte-exactly.

Deviations the oracle knows about and tags instead of failing:

- **ports** — node too narrow for its output ports: we compress/widen
  (upstream lets routes start inside neighboring nodes)
- **shift** — anything left of the margin: we shift the drawing right
  (upstream renders at negative coordinates)
- **TSERR** — the original threw or hung on degraded input that we survive

## 4. Visual tooling

- `./scripts/gallery.sh` → `target/gallery/index.html`: every corpus graph,
  `-Kion` vs `-Kdot` side by side.
- `target/compare/index.html`: ours vs the original iongraph renderer on
  real mega-complex functions, synced scrolling + zoom. Node sizes legitimately
  differ (true font metrics vs the original's 6.5px/char estimate); the
  comparison is structure.

## Case study: when the same algorithm places a block differently

Worth recording because it *looks* like a porting bug and isn't. In
mega-complex f13, Block 66 sits ~2 columns left in the original render vs
ours, even though the parity oracle proves byte-exact equality on equal
inputs. Traced cause:

Block 66 is Block 59's **second successor** (the loop exit).
`straightenChildren` has a crossing guard: per layer, per round, a child is
only aligned if its layer position is right of the last child shifted this
round (`lastShifted`). B59's port-0 child (Block 60) sits to the *right* of
B66's dummy in layer order, so any round that shifts B60 blocks B66's
alignment. The pipeline runs exactly 2 rounds:

- with the original's 6.5px/char sizes, B59 was still drifting right between
  rounds, so B60 needed shifting in round 2 as well → B66 never got a turn →
  it stayed at the left margin;
- with real font metrics, B59 converged in round 1, round 2's B60 shift was
  a no-op, the guard stayed free → B66 aligned under B59's port.

So placements of loop-exit / later-port children are
**convergence-dependent**: sub-point node-size changes flip a discrete
alignment gate, not a proportional shift. Neither outcome is more correct;
with identical inputs the port reproduces the original's choice exactly
(verified both ways). Expect every "why is this block over there?"
difference vs the original viewer to be either node measurement or this.

## Debugging tools

- `ION_DEBUG_LNODES=1 ion-dump < graph.txt` — dump every layout node
  (layer, block, dummy target, flags, x, dst/src adjacency) and every block
  (layer, loop id, header/backedge, layout node).
- `ION_DEBUG_PASSES=1` — x snapshot after every straightening pass.
- `ION_DEBUG_CHILD=<block>` / `ION_DEBUG_CHILD_LI=<layer>` — log every
  `straightenChildren` decision (target, gate state, shift/skip) touching
  that block / on that layer.
- `ION_DUMP_INPUT=1 dot -Kion ...` — the plugin prints the exact layout
  input (sizes with full float precision, edges, metadata) in `ion-dump`'s
  stdin format, so any rendered graph can be replayed through the core
  byte-for-byte.
- `ion-dump` input format: `node <w> <h> <loop_depth> <header:0|1>
  <backedge:0|1>` per node, then `edge <tail> <head>`; output: `cell <i>
  <left> <top> <w> <h> <layer>` + `size <w> <h>`.
