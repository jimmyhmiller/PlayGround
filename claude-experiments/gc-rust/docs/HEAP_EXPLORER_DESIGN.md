# Heap Explorer — design (APPROVED; design-first, build gated on 1b sign-off)

The next JVM-grade tooling rung after Target-1a (GC log/histograms) and Target-1b
(allocation-site profiling). Status: **design APPROVED by Leader review
(2026-06-20); no code yet — P1 build starts once the Target-1b verdict clears**
(the explorer builds on 1b's `pause_world` STW + `AllocSite` foundation).

> **Resolved decisions (Leader, 2026-06-20)** — folded into the sections below:
> 1. **STW pause cost** is fine for v1 (this is how JVM heap dumps work — opt-in
>    profiling, not steady-state latency). v1 **must MEASURE + report** the pause;
>    incremental/concurrent snapshot is a future optimization, not a v1 gate.
> 2. **Own-pause (on-demand) is primary** — a tool needs a snapshot *when asked*.
>    Piggyback-on-collection is a later cheap periodic-sampling mode.
> 3. **Per-object provenance is OPT-IN** (`GCR_TRACK_ALLOC_SITE=1`); default to the
>    free type-level join. The +4–8 B/object + evacuator touch must NOT be a
>    default cost. Last phase, isolated, same adversarial review 1b got — the
>    reviewer specifically checks the `site_id` word is **preserved across
>    evacuation** (the moving-GC hazard).
> 4. **Real roots everywhere** — the in-degree-0 proxy is simply *wrong* (mishandles
>    cycles + root-held objects). Real roots (parked frames + permanent_extras) is
>    a strict correctness upgrade, not optional; backport to the program-end dump
>    too (at end, roots = globals/permanent_extras — the proxy is wrong even there).
> 5. **CLI + heap-diff first.** The snapshot API + snapshot-to-snapshot diff is the
>    data layer AND the leak-hunting killer app; the interactive widget renders
>    that data afterward.

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
3. **Real roots (G3) — not optional, everywhere.** While parked, every thread's
   roots are reachable exactly as the collector sees them — each parked thread
   published `parked_jit_fp`, and `walk_gc_frames` + `permanent_extras` enumerate
   the true root slots. The snapshot's root set comes from these, never the
   in-degree-0 proxy (which is *wrong*: it mishandles cycles and root-held objects
   that also have an in-edge). Reachability and the dominator super-root use the
   real roots. **Backport to the program-end dump too:** at end there are no stack
   roots, but globals/`permanent_extras` are still the real roots and the proxy is
   wrong even there — replace it. (Proxy remains only as a last-ditch fallback if
   no root source is available at all.)
4. **Measure + report the pause (v1 requirement).** Record the STW snapshot
   duration (reuse the 1b/`record_gc_event` timing style) and emit it with the
   snapshot (`pause_ns` in the header) + the log. STW is fine for v1 (JVM heap
   dumps work this way); we just never hide its cost. Incremental/concurrent
   snapshot is a future optimization, not a v1 gate.
5. Drop the pause; serialize/format outside it (owned data) to minimize STW time.

The snapshot critical section is **read-only and allocation-free on the GC heap**
(it only reads object memory + builds host-side `Vec`s), so it cannot re-enter
`gc_lock` — same contract the 1b profiler honors. STW duration is O(live objects)
(one walk + one dominators pass).

**Triggers for G1 — on-demand own-pause is PRIMARY:**
- **Primary (on-demand):** a control trigger — a signal handler (e.g. `SIGUSR1`)
  / a tiny control FFI (`ai_heap_snapshot`) / an in-language intrinsic
  `heap_snapshot()` — requests "snapshot now"; the next safepoint takes its own
  `pause_world` snapshot. A tool needs a snapshot *when asked*, not whenever GC
  happens to fire.
- **Later (periodic sampling):** `GCR_HEAP_SNAPSHOT_EVERY=<n>` piggybacks the
  existing STW collection (world already stopped during a collection → nearly free,
  no extra pause). A cheap sampling mode, secondary to on-demand.
Each writes the snapshot to a sink (`GCR_HEAP_SNAPSHOT_DIR=<path>` → one JSON file
per snapshot, monotonically numbered) that the viewer/diff tooling reads. No
silent capping — if snapshots are dropped (sink full / rate-limited), say so in
the log.

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

Three layers — **data core (CLI + heap-diff) first**, interactive after:

1. **CLI / env (scriptable, headless) — the data core, built first.** The snapshot
   JSON (extend the existing `dump_heap_json` shape with `roots` = real roots,
   `pause_ns` + `snapshot_seq`/`gc_seq` header, optional `site` per object) **plus
   `gcr heap-diff a.json b.json`** — the snapshot-to-snapshot diff that is both the
   reviewable data layer and the highest-value use case (leak hunting). Works for
   JIT (`gcr run`) and AOT binaries identically (shared dump path). This is P1.
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

## 7. Phasing (re-ordered per Leader review — data core first)

- **P1 — Snapshot data core + heap-diff (FIRST; no hot-path change).**
  `heap_snapshot()` under `pause_world` (own-pause); **real roots (G3) everywhere,
  incl. backporting the program-end dump off the proxy**; versioned JSON header
  (`snapshot_seq`, `gc_seq`, `pause_ns`); reuses `collect_graph` / `retained_sizes`
  unchanged. Plus **`gcr heap-diff a.json b.json` (G5)** — per-type / per-site /
  per-retainer growth across snapshots: the leak-hunting killer app. This is the
  reviewable data layer; the safe, high-value core.
- **P2 — On-demand trigger (G1).** Primary control trigger (signal / FFI
  `ai_heap_snapshot` / in-language `heap_snapshot()`) → own-pause snapshot at the
  next safepoint + `GCR_HEAP_SNAPSHOT_DIR` sink. (Periodic
  `GCR_HEAP_SNAPSHOT_EVERY` piggyback-on-collection is a secondary sampling mode.)
- **P3 — Widget `gcr_heap.ft` (G6).** Interactive navigation + a "growth" tab,
  *rendering* the P1/P2 data, modeled on the `gcr_xray.ft` proc-bridge pattern.
- **P4 — Per-object alloc-site provenance (G4; opt-in; last + isolated).**
  `site_id` word in the header behind `GCR_TRACK_ALLOC_SITE=1`, written at alloc,
  **preserved across evacuation**; default stays the free type-level join.
  Adversarial review of the hot-path/evacuator change like 1b — the reviewer
  specifically verifies the `site_id` survives a relocation.

P1 alone is a meaningful upgrade (consistent, real-rooted, mid-execution snapshots
+ leak diffing) and carries no hot-path risk; the riskier work (P4) is last,
isolated, and off by default.

---

## 8. Decisions (resolved at Leader review, 2026-06-20)

All five originally-open questions are now decided — see the **Resolved decisions**
box at the top, folded into §§3–7:

1. STW pause cost → fine for v1; **measure + report** it (`pause_ns`). Incremental
   snapshot is future, not a gate.
2. Trigger → **own-pause on-demand primary**; piggyback-on-collection is a later
   periodic-sampling mode.
3. Per-object provenance → **opt-in `GCR_TRACK_ALLOC_SITE=1`**, default to the free
   type-level join; last phase (P4), isolated; reviewer verifies `site_id`
   survives evacuation.
4. Roots → **real roots everywhere** (proxy is wrong even at program end);
   backport the program-end dump off the proxy.
5. Surface → **CLI + heap-diff first** (data core + leak killer app); widget after.

**Build gate:** P1 starts once the Target-1b verdict clears (the explorer builds
on 1b's STW + `AllocSite` foundation).

---

*Prepared by Rust-gc; design APPROVED at Leader review 2026-06-20. No
implementation started (gated on 1b sign-off). Reuses Target-1b's `pause_world`
snapshot discipline and `AllocSite` table.*
