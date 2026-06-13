# How the layout works

This is a faithful Rust port of iongraph's real layout algorithm
(`generic-layout/layout.ts`, the code behind the SpiderMonkey IR viewer),
wrapped in a frontend that makes **arbitrary directed graphs** valid input.
The port is verified byte-exact against the original — see
[VERIFICATION.md](./VERIFICATION.md).

Everything lives in `src/core.rs`. The pipeline:

```
frontend:  classify edges → infer/repair loop metadata → build blocks
layout:    find roots → find loops → assign layers
           → build layout graph (dummies + backedge channels)
           → straighten (horizontal) → joints/tracks → verticalize
routing:   one bezier route per input edge, walked off the layout graph
```

## 1. Frontend (ours — the original doesn't have one)

The original algorithm assumes a well-formed SpiderMonkey CFG: a DAG plus
explicitly marked backedge blocks, exactly one backedge predecessor per loop
header, coherent `loopDepth`s, and at least one root. Arbitrary DOT files
guarantee none of that, and the original throws or **infinitely recurses**
when assumptions break (it hangs on one pass of its own bundled
`tokenize-ion.json`). The frontend makes every input safe *before* the
algorithm sees it:

- **Edge classification.** Every input edge is classified by incremental
  reachability, in input order: an edge whose head already reaches its tail
  through previously accepted forward edges is a **feedback** edge; `a -> a`
  is a **self-loop**; everything else is **forward**. The forward subgraph is
  acyclic *by construction*, so layer assignment can never spin.
- **Loop inference** (when no `ion_*` metadata is present): each feedback
  head becomes a loop header; all feedback edges into one header share a
  single **synthesized zero-size backedge block** (think: multiple
  `continue`s); the natural loop body (backwards walk from the feedback
  tails) gets `loopDepth += 1`.
- **Metadata repair** (when metadata is present but inconsistent): headers
  with several backedge preds keep the first (extras are demoted), headers
  with none stop being headers, backedge blocks keep exactly one successor.
- Self-loops and feedback edges that can't be expressed in the layout are
  excluded from the block graph and routed separately at the end.

## 2. Layer assignment (`find_loops`, `assign_layers`)

Direct ports of the original, converted from recursion to explicit work
stacks (50k-deep graphs would overflow the native stack):

- `find_loops` walks the DFS tree maintaining a stack of enclosing loop
  headers (`loop_ids_by_depth`), clamping bogus depths, and records each
  block's owning loop and each header's parent loop.
- `assign_layers` pushes blocks down (`layer = max(layer, parent + 1)`),
  pins backedge blocks to their header's layer, accumulates each loop's
  height up the parent chain, and **defers loop-exit edges**: a successor
  with a shallower loop depth is parked on the header's `outgoing_edges`
  and visited only after the whole loop body, at `header layer + loop
  height` — that's why loop exits appear below the entire loop.

  The iterative port preserves the recursion's exact visit order, including
  the subtle bit that the defer test runs when a child's frame *pops* (after
  earlier siblings' subtrees finished), and that a header's deferred list is
  iterated live while it grows.

Robustness additions (all no-ops on well-formed input): rootless graphs get
a fallback root; backedge layers are re-pinned after the walk (the header
can move after the backedge was visited); a bounded fixpoint re-runs the
visit for any forward edge that ended up non-monotonic (possible only with
side entries into inferred loop bodies).

## 3. Layout graph construction (`make_layout_nodes`)

Blocks become **layout nodes** arranged per layer, in block input order.
Two kinds of dummy nodes join them:

- **Pass-through dummies.** An edge spanning multiple layers gets one dummy
  per intermediate layer, *coalesced by destination*: all edges heading to
  the same block share one dummy per layer. This coalescing is what makes
  fan-ins render as a single confluent line. Dummies are prepended to each
  layer (left of the real blocks).
- **Backedge dummy chains — the loop channels.** Every loop gets one dummy
  per layer it spans, placed *after the loop's last block on that layer*,
  linked bottom-to-top, ending in an "imminent" dummy on the header's layer
  that connects to the backedge block. Feeders into the backedge attach to
  the chain on their own layer. Because the chain consists of real layout
  cells, the loop channel **reserves horizontal space** on every layer —
  nothing else can occupy it, which is why backedge channels never overlap
  blocks. (Ours extends a chain down to the loop's deepest feeder if
  inferred metadata put a feeder outside the loop's nominal extent.)

Dst slots are indexed by successor position, so "port k" always means "the
k-th successor", regardless of connection order. Orphaned chains are pruned;
the leftmost/rightmost dummy runs of each layer are flagged for later passes.

## 4. Horizontal straightening (`straighten_edges`)

A fixed pipeline of composable passes, run exactly like the original:

```
2 × ( straightenChildren ; pushIntoLoops ; straightenDummyRuns )
straightenDummyRuns
8 × ( nearlyStraightUp ; nearlyStraightDown )
straightenConservative
straightenDummyRuns
suckInLeftmostDummies
```

- `straightenChildren` — pull each child under its first parent's output
  port, guarded by `lastShifted` so alignments never cross an edge already
  aligned this round (see the case study in VERIFICATION.md for how this
  guard makes some placements *convergence-dependent*).
- `pushIntoLoops` — no block may sit left of its loop header.
- `straightenDummyRuns` — every dummy run (same destination) snaps to the
  run's max x, making long edges vertical.
- `nearlyStraightUp/Down` — snap almost-vertical hops (≤30pt of wiggle).
- `straightenConservative` — nudge blocks right onto a parent/child port if
  nothing to their right collides. (Replicates an upstream quirk: the block
  with id 0 is never nudged — `!node.blockId` is falsy for 0 in JS.)
- `suckInLeftmostDummies` — compact each layer's leading dummies back toward
  the content, then unify each run at its **minimum across all layers** so
  leftmost long-edge columns are straight instead of stair-stepping.

Gaps are enforced per layer (`blockGap` = 44pt, plus `portStart` between a
dummy and the first real block). After everything, the whole drawing is
shifted right if anything ended up left of the margin (the original lets
dummies go negative).

## 5. Tracks and vertical placement (`compute_joints`, `verticalize`)

The space between two layers holds horizontal "tracks". Every edge hop with
a horizontal jog becomes a *joint*; joints heading to the **same
destination share one track** ("merge arrows" — the confluence), rightward
and leftward runs get separate track sets, and each joint takes the
innermost non-overlapping track. Layer y positions then stack up:
`layer height + trackPadding + tracks + trackPadding`.

## 6. Routing (ours)

The original's renderer draws each hop separately; Graphviz wants one spline
per input edge. Routes are rebuilt by walking the layout graph:

- **Forward edges** follow their dummy chain hop by hop, bending at each
  layer's track y (with its joint offset), fused into one orthogonal
  polyline with rounded corners (cubic beziers, 3k+1 control points,
  variable length).
- **Edges into a backedge** descend to the feeder's track band, jog to the
  loop channel, ride the chain *up* (jogging where the chain's column
  shifts), and enter the backedge block's right edge at header-arrow height
  — or continue straight into the loop header itself when the backedge
  block is synthesized (inferred loops).
- **Backedge → header** is the horizontal header arrow. If a sibling block
  sits between them, the route dodges through the clear band above the
  layer (the HTML viewer draws straight through and hides it with z-order;
  SVG can't).
- **Self-loops** hook around the node's right edge.

## 7. Graphviz integration (`src/lib.rs`, `plugin/gvplugin_ion.c`)

The C ABI passes node sizes + edges in, and gets node centers, (possibly
widened) sizes, and variable-length routes back through a shared points
buffer (`ion_layout_free_points` releases it). The plugin registers the
`ion` engine, measures labels via `common_init_node`/`gv_nodesize`, runs the
layout, and writes positions/splines back. Hard-won integration notes:

- `graph_init()` must be called and the libgvc global `State` must be set to
  `GVSPLINES`, or xdot-based outputs (`-Tjson`, `-Txdot`) segfault inside
  `agxset`.
- When the layout **widens** a node (to fit its output ports), setting
  `ND_width` is not enough — renderers draw the shape **polygon** built at
  init time. The plugin re-runs the shape's `freefn`/`initfn` after setting
  the new size (`poly_init` sizes boxes from `ND_width` directly; the
  width/height *attributes* are only consulted for `regular` shapes). Never
  call `gv_cleanup_node` mid-layout — it deletes the node's data record.
- Edge labels park beside the route midpoint and stagger when they collide
  (merged parallel edges share midpoints); the graph bounding box grows to
  cover them.
- `rankdir` LR/BT/RL run the layout in transposed/mirrored space (node sizes
  swapped going in, all geometry mapped back coming out), so ports land on
  node sides for horizontal flow.

## Port spacing (ours)

Output ports sit at `x + portStart + k·portSpacing` (16 + k·60). The
original lets ports overflow narrow nodes — a route can then *start inside a
neighboring node*. Here, port spacing compresses to fit the node (never
below 24pt), and nodes too narrow even for that get widened. Real ion blocks
are wide; this never triggers on them (verified — see the parity oracle).
