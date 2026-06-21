# Heap Explorer — design proposal (DRAFT, for review)

The next JVM-grade tooling rung after Target-1a (GC log/histograms) and Target-1b
(allocation-site profiling). Status: **design only — nothing built yet.** This is
the proposal to review before any code, in the spirit of design-first.

The honest framing: a large fraction of a heap explorer *already exists* in
`gc::dump`. This document is mostly about the **delta** — the few new mechanisms
that turn the current program-end text/JSON dump into an interactive,
mid-execution, time-aware explorer — and the **decisions** that delta forces.

---

## 1. What already exists (reuse, don't rebuild)

`crates/gcrust-rt/src/gc/dump.rs` + `Heap` already give us the hard parts:

- **`Heap::walk_live_objects(visitor)`** — walks every allocated object in the
  live spaces (tenured from-space + nursery), handing each `(obj, &TypeInfo)`.
- **`collect_graph`** — builds the object graph: per node `(obj, type_id, bytes,
  refs[])`, where `bytes = TypeInfo::allocation_size(varlen_len)` and `refs` come
  from **`scan_object`** — i.e. *exactly the edges the GC traces*, including
  interior pointers in flattened `#[value]` fields and varlen elements. No
  separate, drift-prone edge model.
- **`retained_sizes`** — a real dominator tree (Cooper–Harvey–Kennedy
  `intersect`) giving per-object exclusive retained size; `top_retainers`.
- **`dump_heap_json`** — a clean, already-structured snapshot:
  `{summary:{objects, bytes, by_type[], roots[], top_retainers[]},
  objects:[{id, type, bytes, retained_bytes, render, refs[]}]}`.
- **`dump_heap_text`** — human report: per-type histogram + per-object reflected
  render.
- **Reflection** (`render_object` + `TypeMeta`/`ValueMeta`) — decodes objects to
  source type/field names and values (`Point { x: 3, y: 4 }`), not raw words.
- **Target-1b `AllocSite` table** — `site_id → (function, type_id)`, already baked
  and installed on the heap (JIT + AOT).
- **`Heap::pause_world() -> WorldPause`** — STW: parks every other mutator at a
  safepoint, holds `gc_lock`. This is the snapshot-consistency primitive the 1b
  profiler dump already uses; the explorer reuses it verbatim.
- **A proven widget pattern** — `widgets/gcr_xray.ft` (+ `gcr_source.ft` /
  `gcr_out.ft`) is a live "Compiler Explorer": a `.ft` widget driving the `gcr`
  binary over the event-driven proc bridge, coalesced/responsive, with `df` UI
  helpers. The heap explorer's UI is a *new widget in this same mould* (a heap
  view, not a pipeline view), not new infrastructure.

So the explorer is not a green-field build. It is: **lift the existing dump to
(a) run mid-execution at a consistent STW snapshot, (b) use the real GC root set,
(c) optionally attribute each object to its alloc site, (d) diff snapshots over
time, and (e) surface it interactively.**

---

## 2. Gaps the explorer must close (the delta), prioritized

| # | Gap | Today | Needed |
|---|-----|-------|--------|
| G1 | **When** | Dump runs only at *program end* (`GCR_HEAP_DUMP`) | Snapshot **on demand, mid-execution**, at a consistent point |
| G2 | **Consistency** | Relies on "program end = quiescent" | Snapshot under **STW (`pause_world`)** so no half-moved object is read mid-collection |
| G3 | **Roots** | `roots` = in-degree-0 *proxy* (approx; mis-handles cycles & root-held-yet-referenced objects) | The **real GC root set** (frames + permanent extras), available for free at an STW safepoint |
| G4 | **Provenance** | Object shows its *type*, not *where it was allocated* | Optional **per-object alloc-site** attribution (fuse 1b) |
| G5 | **Over time** | One isolated snapshot | **Deltas across collections** → leak/growth detection |
| G6 | **Surface** | Text/JSON to stderr | **Interactive** navigation (jim widget) + scriptable CLI |

---

## 3. The core new mechanism: a consistent mid-execution snapshot (G1+G2+G3)

This is the part that directly reuses what 1b built.

Add one runtime entry point:

```rust
// gc::dump (or gc::snapshot)
pub fn heap_snapshot(heap: &Heap) -> HeapSnapshot   // SAFE: takes the pause itself
```

Implementation = the 1b profiler discipline applied to the heap walk:

1. `let _pause = heap.pause_world();` — every other mutator parks at a safepoint;
   `gc_lock` is held so no collection can run. (Same liveness as any STW
   collection: safepoint polls at every loop back-edge + allocation; STATE_BLOCKED
   counts as safepoint-equivalent; `register_thread` loops on `gc_requested` so no
   late unparked registrant. Verified for the 1b dump; identical here.)
2. **Inside the pause**, run the existing `collect_graph` (walk + `scan_object`
   edges + sizes) and `retained_sizes`. No mutator is allocating or relocating →
   every header/varlen-count/pointer is valid and stable. This is the strict,
   enforced version of the "heap must be quiescent" precondition `dump_heap_*`
   documents but does not enforce.
3. **Real roots (G3):** while parked, every thread's roots are reachable exactly
   as the collector sees them — each parked thread published `parked_jit_fp`, and
   `walk_gc_frames` + `permanent_extras` enumerate the true root slots. Mark the
   snapshot's root set from these instead of the in-degree-0 proxy. Result:
   accurate reachability (cycles handled, root-held objects correctly rooted) and
   dominators rooted at a virtual super-root over the *real* roots — a strict
   correctness upgrade that the program-end dump literally cannot do (no stack
   roots at program end).
4. Drop the pause; serialize/format outside it (owned data) to minimize STW time.

The snapshot critical section is **read-only and allocation-free on the GC heap**
(it only reads object memory + builds host-side `Vec`s), so it cannot re-enter
`gc_lock` — same contract the 1b profiler honors. STW duration is O(live objects)
(one walk + one dominators pass); for large heaps this is a real pause and is the
cost knob to watch (see Open Questions).

**Triggers for G1 (request a snapshot while the program runs):**
- `GCR_HEAP_SNAPSHOT_EVERY=<n>` — capture after every *n*th collection
  (piggyback on the existing `record_gc_event` cold path; the world is already
  stopped during a STW collection, so a snapshot there is nearly free — no extra
  pause).
- A signal handler (e.g. `SIGUSR1`) or a tiny control FFI (`ai_heap_snapshot`)
  that requests "snapshot now" → next safepoint takes it. Lets an external viewer
  pull a snapshot from a running program.
- In-language intrinsic `heap_snapshot()` for programs that want to checkpoint at
  a known point.
Each writes the snapshot to a sink (`GCR_HEAP_SNAPSHOT_DIR=<path>` → one JSON file
per snapshot, monotonically numbered) that the viewer tails. No silent capping —
if snapshots are dropped (sink full / rate-limited), say so in the log.

---

## 4. Per-object allocation-site attribution (G4) — fuse 1b, with honest cost

Goal: in the explorer, click an object and see **where it was allocated**
(`function + type`, the 1b site), not just its type. This is the capability that
makes 1b and the explorer more than the sum of their parts.

The hard constraint: in a **moving** GC, you cannot key a side-table by object
address (addresses change on every relocation). So per-object provenance must
travel *with the object*. Options, with costs:

- **(A) Site id in the object header — opt-in.** Add a `site_id: u32` word to the
  object header, written by `ai_gc_alloc_*` (we already pass `site_id` there for
  1b — it's in hand) and **preserved across evacuation** (copy it when the
  collector forwards the object). Cost: +4–8 bytes/object and one extra store on
  the alloc slow path. Too heavy to be always-on; gate behind an
  **explorer/debug build or a runtime flag** (`GCR_TRACK_ALLOC_SITE=1`) that
  selects a wider header. Exact, per-object, survives GC.
- **(B) Type-level join — free, approximate.** Don't store anything per object.
  The explorer already has per-type live counts (heap walk) and per-site totals
  (1b). Show, for a selected type, the sites that allocate it (`AllocSite` table
  filtered by `type_id`) and their cumulative bytes. Not exact per *instance*, but
  zero cost and often enough to answer "who allocates all these `Vec<I64>`s?".
- **(C) Deferred.** Ship the explorer without per-object provenance; add (A) later
  behind the flag.

**Recommendation:** default to **(B)** (free, useful), offer **(A)** behind
`GCR_TRACK_ALLOC_SITE=1` for when you need exact per-instance provenance. Flag the
header-width change for review — it touches the alloc hot path and the evacuator,
so it gets the same adversarial scrutiny 1b's ABI change did.

---

## 5. Growth / leak detection across snapshots (G5)

With snapshots numbered over time, leak detection is a diff, computed host-side
(in the widget or a `gcr heap-diff` CLI) — no runtime cost:

- Per **type** and per **site**: retained-bytes and live-count series across
  snapshots; flag types/sites whose retained size grows **monotonically** over
  *k* consecutive snapshots (the classic leak signature).
- Per **dominator subtree**: which retainer's exclusive retained size is growing
  — "object #42 (`Cache`) retains 4 MB and climbing."
- Surface as a "growth" tab: sortable by slope.

This is pure analysis over the JSON snapshots; it needs no new runtime mechanism
beyond G1 (being able to take more than one snapshot).

---

## 6. Surface (G6)

Three layers, cheapest first:

1. **CLI / env (scriptable, headless):** the snapshot JSON (extend the existing
   `dump_heap_json` shape with `roots` = real roots, optional `site` per object,
   and a `snapshot_seq`/`gc_seq` header). `gcr heap-diff a.json b.json` for
   offline leak diffing. Works for JIT (`gcr run`) and AOT binaries identically,
   since both already share the dump path.
2. **Interactive jim widget — `gcr_heap.ft`** (the headline): modeled on
   `gcr_xray.ft`'s proc-bridge pattern, but a heap view:
   - left: type histogram / top-retainers / roots list (sortable by retained
     bytes);
   - right: object inspector — reflected fields, outgoing refs as clickable
     edges (navigate the graph), incoming refs (retainers), dominator parent;
   - a "growth" tab driven by the snapshot series (G5);
   - a "snapshot now" button → fires the control trigger (§3) and reloads.
   Reuses `df` helpers + the coalesced proc bridge already proven in the compiler
   explorer.
3. **Live drive (stretch):** point the widget at `GCR_HEAP_SNAPSHOT_DIR` of a
   *running* program and watch the heap evolve (tail new snapshot files).

---

## 7. Proposed phasing (each slice independently reviewable + shippable)

- **P1 — Consistent snapshot API.** `heap_snapshot()` under `pause_world`; real
  roots (G3); versioned JSON (snapshot_seq, gc_seq). Reuses `collect_graph` /
  `retained_sizes` unchanged. *No hot-path change.* (The safe, high-value core.)
- **P2 — Triggers (G1).** `GCR_HEAP_SNAPSHOT_EVERY` (free, piggybacks STW
  collection) + `GCR_HEAP_SNAPSHOT_DIR` sink + a control trigger (signal or FFI).
- **P3 — Widget `gcr_heap.ft` (G6).** Interactive navigation over P1/P2 output.
- **P4 — Growth/leak diff (G5).** `gcr heap-diff` + the widget "growth" tab.
- **P5 — Per-object alloc-site (G4, opt-in).** Header-width change behind
  `GCR_TRACK_ALLOC_SITE=1`, evacuator preserves it. Adversarial review of the
  hot-path/evacuator change (like 1b).

P1 alone is a meaningful upgrade (consistent, real-rooted, mid-execution
snapshots) and carries no hot-path risk; the riskier work (P5) is last and
isolated.

---

## 8. Open questions for review

1. **STW pause duration on large heaps.** The snapshot is O(live objects) under
   STW. Acceptable for a debugging/tooling pause? Or do we need an incremental /
   concurrent snapshot later (much harder; probably not for v1 — tooling pauses
   are expected)?
2. **Snapshot at an existing STW collection vs. its own pause.** Piggybacking
   `GCR_HEAP_SNAPSHOT_EVERY` on a collection is free (world already stopped) but
   only fires when GC runs; an on-demand snapshot needs its own pause. Offer both?
3. **Per-object alloc-site (P5): header growth vs. value.** Is exact per-instance
   provenance worth +4–8 bytes/object behind a flag, or is the free type-level
   join (B) enough for v1?
4. **Real roots vs. proxy roots** for the *program-end* dump too — should P1's
   real-root reachability replace the in-degree-0 proxy everywhere, or only for
   mid-execution snapshots (program end has no stack roots, so the proxy is the
   only option there)? Likely: real roots when a live root set exists, proxy as
   fallback.
5. **Widget vs. CLI priority.** Is the interactive widget the headline (P3 early),
   or is the scriptable JSON + `heap-diff` the more valuable first deliverable?

---

*Prepared by Rust-gc for Leader review. Design-first; no implementation started.
Reuses Target-1b's `pause_world` snapshot discipline and `AllocSite` table.*
