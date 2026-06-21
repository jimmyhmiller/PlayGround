# Future Work: gc-rust GC + tooling roadmap

An honest engineering roadmap, grounded in the current implementation (real
files/symbols), flagging what is *verified* versus what still needs an *audit*.

The project's reason to exist is **Rust-with-a-GC**: a precise, moving,
generational collector (no box zoo, no string-type zoo), high performance (a bit
slower than Rust is fine), the safety a real GC buys, and — the headline —
**tooling at or beyond JVM grade**. The roadmap is ordered to that: tooling
first, then the remaining collector soundness/robustness work, then performance.

The compiler is Rust + inkwell (LLVM). It monomorphizes; generics specialize per
concrete type, so every heap-object slot has a statically-known representation.
(A self-hosting *bootstrap* compiler — a proof-of-concept that emitted i64-
uniform code onto a separate non-moving C runtime — was dropped. It is not a goal
and nothing here depends on it.)

## Current state (honest snapshot)

A precise, monomorphizing, generational, moving GC lives in the LLVM-free
`gcrust-rt` crate. New objects allocate in a young nursery collected by cheap
minor GCs; survivors promote to a tenured generation (copying semi-space). Roots
are precise: every `Ref` local is a frame root slot, every value-with-ref local
is rooted via indirect frame roots, and heap objects are traced by a per-
`type_id` `TypeInfo` shape table (with `interior_ptrs` for references embedded in
flattened `#[value]` fields). A generational write barrier marks card tables on
old→young pointer stores **and on flattened value-with-reference stores** (see
P0 history below).

**Precise-layout soundness (closed).** Tracing is precise from layout:
`gc::scan_object` visits exactly the value-field slots, the `interior_ptrs`, and
the varlen `Values` elements — never arbitrary words. The monomorphizing front
end places every scalar in the untraced raw region and zeroes unused enum
pointer-slots, so a traced slot only ever holds a pointer-or-null. The collector
therefore trusts the layout and never heuristically re-identifies pointers (which
would be unsound in a moving GC). A **precise-layout detector** (`gc::heap::
gc_verify_armed`) is armed under debug, `--gc-stress`, and release+`GCR_GC_VERIFY=1`:
it panics loudly if a traced slot ever points at a non-object header — surfacing
a layout/rooting bug instead of masking it. Proven by `gc::tests`:
`semi_space_adversarial_int_payload_in_raw_region_not_relocated` (a real from-
space address stored as an enum int-payload is never followed),
`..._negative_control` (a real address in a *traced* slot always is), and
`semi_space_detector_panics_on_bad_header_in_traced_slot` (the detector
demonstrably trips).

What is **not** yet production-grade: the JVM-grade tooling below mostly does not
exist; allocation is still a runtime call per object (no inline fast path); the
concurrent collector path is unverified; heap sizing is fixed; weak refs /
finalization are absent. Details below.

---

## P0 — Tooling ≥ JVM (the headline goal)

This is the project's whole reason to exist and is where the largest gap is. We
already have strong substrates — make them an actual toolchain.

- **Have today:** heap snapshot dumps (`GCR_HEAP_DUMP=1` text / `=json`
  structured, with per-type histogram, root set, and dominator-based retained
  sizes — `gc::dump`); a statemap tracer (`StatemapTracer`); full runtime
  reflection metadata (nominal type/field names, enum variants); the `gcr emit
  <tokens|ast|core|layout|mono|reflect>` structured-IR dump; and **GC stats +
  pause-time histograms + a GC log** — `GCR_GC_STATS=1` prints collection counts
  and per-kind pause p50/p99/max plus reclaimed/promoted bytes, and
  `GCR_GC_LOG=<path>` writes one JSON object per collection (`{seq, kind,
  pause_ns, before_bytes, after_bytes, reclaimed_bytes, promoted_bytes}`) for
  offline analysis. Both work on the JIT (`gcr run`) AND AOT (`gcr build`) paths
  via a shared `Heap::gc_stats_summary`/`gc_log_jsonl` (the events are recorded
  cold-path, once per collection, in `Heap::record_gc_event` — never on the
  allocation hot path). (Previously `GCR_GC_STATS` was JIT-only and printed only
  raw counts — corrected here.)

- **A real debugger.** Source-level breakpoints, stepping, locals/heap
  inspection that understands the GC object model (decode objects via reflection
  metadata, follow references, show value-type fields inline). DWARF emission
  from the LLVM backend is the natural substrate; pair it with the reflection
  table so a debugger renders *language* values, not raw words.

- **Allocation-site profiling — DONE + SIGNED OFF (Target-1b).** `GCR_ALLOC_PROFILE=1`
  prints per-site count+bytes (JIT + AOT). NON-ATOMIC per-thread `SiteCounter`
  vector on `ThreadState` (owner-only writes, zero atomics/locks on the alloc hot
  path), merged at dump across live threads + a `retired_alloc_counters`
  accumulator (folded at deregister so joined workers aren't lost). `ai_gc_alloc_*`
  carries a `site_id`; baked side-table `site_id → (function, type_id)` via the
  reflect blob (AOT) / `set_alloc_sites` (JIT). v1 granularity = **function +
  allocated-type** (one site id per pair; no faked `file:line:col` — Core IR has no
  span; line-precise sites await span-threading, the shared debugger prereq below).
  The dump-time read of the non-atomic counters is done under a `pause_world` STW
  pause (a live unjoined thread would otherwise make it a data race — confirmed a
  real heap-UAF under ASan in review, fixed STW-correctly, ASan-proven +
  independently approved). The pre-existing `dump_heap_text`/`dump_heap_json` were
  folded into the same STW discipline.
  *Code-state pointer (sweeps scramble per-commit attribution):* signed-off state
  is tagged `gcr-target-1b`; key files: `runtime.rs` (ai_gc_alloc_* ABI +
  gcr_runtime_main install), `gc/thread.rs` (SiteCounter/record_alloc), `gc/heap.rs`
  (alloc_sites table + `alloc_site_profile` STW merge + retired fold), `gc/reflect.rs`
  (AllocSite + blob §4), `gc/dump.rs` (STW-gated dumps), `src/codegen.rs` (site-id
  assignment), tests in `gc/tests.rs` (incl. the ASan resize-stress concurrent
  test) + `tests/alloc_profile.rs`; gate `scripts/asan_dump_race.sh`.

- **Live heap explorer.** The JSON snapshot is already a clean substrate; drive
  it from a safepoint *during* execution (not just at program end) into an
  interactive viewer (e.g. a jim widget) — navigate the object graph, retained
  sizes, dominators, leak/growth deltas across collections.

- **Automatic leak/growth detection.** Diff live-set composition across
  collections and surface types whose retained size grows monotonically.

- **Beyond the JVM (stretch).** Deterministic record-replay / time-travel of an
  execution (the moving GC + precise roots make a consistent heap snapshot cheap),
  and allocation-aware differential debugging.

---

## P1 — Collector soundness & robustness

The known P0 *soundness* holes from the previous cycle are now closed (see
History). Remaining soundness/robustness items:

- **FFI pinning audit.** `as_c_bytes` copies a `String`/array's bytes to the
  (non-moving) native stack for the duration of an extern call, which is correct.
  Verify there is no path where a *bare* GC reference is handed to C and then a
  collection moves it (e.g. a struct pointer passed to a callback that re-enters
  managed code and allocates). If such paths exist, they need explicit pinning or
  a copy.

- **Safepoint coverage audit.** Polls are emitted at loop headers
  (`emit_safepoint_poll`). Verify the complete set: every loop back-edge, calls
  for STW liveness, and any unbounded allocation-free computation. A long-running
  allocation-free loop that never polls can block a stop-the-world collection on
  another thread — a liveness defect (hang, not corruption), but still a bug.

- **Concurrent/incremental major GC: verify or remove.** The heap carries
  machinery for concurrent collection (`GcPhase::Copying`, a global `SATBQueue`,
  snapshot-at-the-beginning barriers). Determine whether it is sound, complete,
  and exercised, or experimental scaffolding (the default major GC may be STW).
  Either finish and verify it (with `--gc-stress` + the precise-layout detector
  applied to the concurrent barriers) or remove it to avoid a false sense of
  capability.

- **Dynamic heap sizing and OOM policy.** Sizes are fixed (1 MB nursery, 256 MB ×
  2 tenured). Production needs configurable initial/max heap, grow-on-pressure /
  shrink-when-idle, and a graceful OOM path (today `alloc_with_published_frame`
  aborts). Surface a recoverable error or a controlled crash with diagnostics.

- **Tenured-generation overhead.** The tenured generation is a copying semi-space
  (2× live old-gen). For large heaps consider mark-compact or mark-sweep to halve
  the footprint, at the cost of collector complexity.

- **Large-object handling and fragmentation.** Objects too big for the nursery go
  straight to tenured. Verify behavior for very large and varlen objects near
  space limits; consider a dedicated large-object space.

- **Weak references and finalization.** Not present. If the language needs weak
  maps, caches, or resource finalizers (FDs held by GC'd objects), these need
  first-class support (weak roots, finalizer queues, resurrection semantics).

- **Continuous stress in CI.** `--gc-stress` is real and the suite passes under
  it. Wire the whole corpus + examples to run under `--gc-stress` (with the
  detector armed) on every CI run.

---

## P2 — Performance

- **Inline bump allocation in generated code.** Every allocation is a call to
  `ai_gc_alloc_fixed`/`ai_gc_alloc_varlen` (the `alloc_window` fast path is kept
  *closed* for the generational/stress heaps). Emitting an inline bump-pointer
  fast path (compare-and-bump against the nursery limit, call the runtime only on
  overflow) is the single biggest allocation-throughput win.

- **Inline write-barrier fast path.** The card-marking barrier goes through
  `ai_gc_write_barrier`. Inline the common-case check (is the store old→young?)
  so most stores cost a few instructions, not a call.

- **Thread-local allocation buffers (TLABs).** Multithreaded allocation contends
  on a shared atomic bump pointer (`AtomicBumpAllocator`). Per-thread TLABs remove
  that contention — a prerequisite for allocation-heavy multithreaded scaling.

- **Parallel collection.** The collector is single-threaded. Parallelizing the
  copy/scan across workers cuts pause times on multi-core machines.

- **Root-set minimization.** The ANF pass conservatively let-binds *every* non-
  atomic GC subexpression, so frames carry more root slots than strictly
  necessary. A liveness-aware pass that only roots values live across a safepoint
  shrinks frames and speeds root scanning. Correctness-first today.

- **Generational tuning.** Nursery size, promotion policy (promote-on-first-
  survival vs aging), and minor/major ratio should be measured and tuned against
  real allocation profiles rather than fixed.

---

## P3 — Testing & verification

- **GC fuzzing.** Generate random programs (varied object graphs, sharing,
  cycles, threads, value-with-ref nesting) and run under `--gc-stress` with the
  detector armed, diffing results against a no-GC reference. This is how residual
  rooting/barrier bugs will be found.

- **Multithreaded stress matrix.** `threads_stress` collects for real; expand it
  to more spawn/join/atom/channel patterns and higher thread counts under both
  generational and semi-space heaps.

  *Coverage note (no silent cap):* `comprehensive_concurrency_stress` defaults to
  `GCR_STRESS_ITERS=50`, but `--gc-stress` collects on **every** allocation, so a
  multithreaded program is ~quadratic and 50 randomized programs are intractable
  per-commit. The **per-commit gate runs 6 iterations in release with the detector
  armed** (`GCR_GC_VERIFY=1 GCR_STRESS_ITERS=6 cargo test --release --test
  concurrency_stress`). The full 50+ -iteration soak (or a wall-clock-bounded
  longer run) should run as an **occasional/nightly** validation, not on every
  commit — it is reduced for speed, not dropped.

- **Property/differential testing** of the collector primitives (relocation +
  rooting invariants) beyond the unit tests in `gc::tests`.

---

## History — closed P0 soundness items (this cycle)

Kept for context; all three are fixed and regression-tested.

- **Generic-enum int/pointer ambiguity — RESOLVED BY CONSTRUCTION.** The only
  model in which an int could be mis-identified as a pointer was the deleted
  bootstrap's i64-uniform codegen on a non-moving runtime. On the Rust moving
  path the monomorphizing layout is precise (scalars in the untraced raw region,
  pointers-first enums with two disjoint cursors, unused ptr-slots zeroed), so no
  traced slot ever holds a scalar. The previous `copy_or_forward` /
  `promote_or_forward` **silent-skip `type_id`-range guards** (conservative-GC
  pointer identification — unsound in a moving collector, and they masked bugs)
  were removed in favour of trusting precise layout + the armed
  `gc_verify_armed()` detector. The 256 MB "avoid collecting" nursery is gone
  (now 1 MB; the workload actually collects). Proven adversarially in `gc::tests`.

- **Stale-register audit of string/array primitives — AUDITED CLEAN.** Verified
  site-by-site: ANF (`anf.rs`) hoists every non-atomic GC operand of `StrConcat`/
  `StrSubstring`/`ArraySet`/`ArrayGet`/`SetField`/… to a rooted frame local
  (`operand`/`boxed`), so no GC value is stranded in a register across a sibling
  operand's allocation; the allocating runtime fns (`ai_str_concat`, etc.) copy
  operand bytes into Rust-owned memory *before* allocating; and `gen_alloc` /
  `gen_make_closure` reload GC operands after their allocation safepoint
  (`repr_relocates`). Two independent layers; no stale-register hazard found.

- **Interior-reference write barrier for tenured value fields — FIXED.**
  `SetField` previously emitted the generational write barrier only for
  `FieldLoc::Ptr` stores; mutating a flattened value-with-references field
  (`FieldLoc::ValueAt`) in a tenured object left the card unmarked, so a minor GC
  could reclaim a still-referenced young object. Now each interior reference of
  the stored value (the object layout's `interior_ptrs` within the field's byte
  range) is barriered, marking the card iff a real old→young edge is created.
