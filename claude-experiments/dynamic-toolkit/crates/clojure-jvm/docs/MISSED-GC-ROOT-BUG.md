# Missed GC root at a JIT call-return safepoint (clojure-jvm, form 430)

**Status:** open, unfixed. Tree is healthy and the bug is honestly *unmasked*
(it reproduces under the precise-rooting stress oracle, see Repro).

> ## ⚠️ 2026-06-01 UPDATE — the "Root cause (working theory)" section below is WRONG. Read this first.
>
> A fresh investigation **refuted** the theory that the bug is a coverage gap in
> the `register-alloc` crate's single-`[min,max]`-interval liveness. Two
> independent proofs:
>
> 1. **The interval test cannot drop a live root.** `register-alloc`'s
>    `liveness.rs` step 3 grows each vreg's `[start,end]` to the bounding box of
>    *every* block where it is `live_in`/`live_out` (plus its def/use positions).
>    So for any vreg genuinely live across a safepoint at position `P`,
>    `start ≤ P ≤ end` *always* holds (live-before ⟹ `start ≤ P`; live-after ⟹
>    `P ≤ end`). The single interval is an **over**-approximation — it can only add
>    dead roots, never drop a live one. A debug assertion added to
>    `build_stackmaps` (assert every dataflow-live-across `GcPtr`/`I64` vreg is in
>    the stackmap) **never fired**, yet the crash reproduced.
>
> 2. **clojure-jvm does not even use that allocator.** Its JIT path is
>    `compile_form_to_jit` → `JitModule::new_empty` + `extend` (`dynlower/src/lib.rs`),
>    which lowers via the in-tree **`Lowerer` + `GreedyRegState`** allocator
>    (`dynlower/src/regalloc.rs`), **not** `register-alloc`'s `LinearScanAllocator`
>    / `batch_lower.rs`. The doc's "single-interval projection (suspected root)"
>    pointer is to dead-for-this-path code. The "spill-all-types had no effect ⟹
>    coverage gap" argument is also unsound: `DynIRFunction::safepoint_action`
>    *already* returns `SpillAndRecord` for every pointer-eligible type, so forcing
>    it on the rest (F64/I8/I32) is a no-op by construction.
>
> ### What is actually true (proven 2026-06-01)
>
> - The crash is a real dangling pointer to a relocated semi-space object reaching
>   `cljvm_rt_first` as a live JIT argument (header bit 63 set, `to`-space target,
>   16 MiB flip — all confirmed by instrumenting `cljvm_rt_first`).
> - **The real root-recording lives in `dynlower/src/lib.rs`**:
>   `Lowerer::collect_live_root_slots` / `record_call_return_safepoint` /
>   `emit_safepoint`. They record a stack slot for a value only when it is
>   (a) root-eligible (`is_gc()`/`I64`), (b) deemed live by the **use-count**
>   model (`GreedyRegState::remaining_uses(v) > 0`), and (c) has a regalloc spill
>   slot (`value_spill_slot(v).is_some()`), plus all `is_gc_root` stack slots.
> - A post-collection conservative stack scan (diagnostic only) confirms the
>   relocating GC leaves the live value in a JIT-frame stack slot at an offset that
>   is **not in any recorded root set** (every stale word found is at an
>   UNRECORDED offset — recorded roots are visited and updated, so are never stale).
> - The missed slot is **not** caught by any tracked category. Verified by
>   experiment (each tried independently and in union, crash persists):
>   removing the `remaining_uses == 0` filter, recording *all* values with a spill
>   slot, recording *all* block-param canonical slots, and `is_gc_root` slots.
>   So the live pointer is preserved in a frame location the `GreedyRegState`
>   root model does not enumerate at all, on the loopy `map`/`reduce1`-over-closures
>   path (`closure.rs` arg-list reader + `recur` loop headers).
> - **Refuted along the way:** nested-`run_jit` fence skipping
>   (`fence_depth == 1`, `outer_jit_frames_skipped == 0` at every walk during
>   form 430); stale global safepoint index (`extend` computes the flat-array base
>   correctly).
>
> ### Update 2 (2026-06-01, later) — JIT recording is NOT the gap; trigger still open
>
> Pursued the "use-count liveness" theory above and **it is not the cause**:
>
> - Added a sound backward-dataflow live-out pass in the `dynlower` `Lowerer`
>   and a compile-time check asserting every heap value live-across each
>   safepoint (call-return AND explicit `Inst::Safepoint`) is in the recorded
>   `root_slots`. The check **never fired** — every live JIT root IS recorded.
> - Replacing the use-count filter with the live-out set, recording all
>   block-param canonical slots, and recording all spilled values: **none**
>   changed the form-430 crash.
> - Form 430 is `max-key` (variadic + `loop`/`recur`). A Rust backtrace shows
>   the dangling value reaches `cljvm_rt_first` **directly from JIT** as its
>   argument — i.e. the JIT handed `rt_first` an already-stale pointer.
> - Making `walk_jit_ancestor_roots` traverse past the innermost `run_jit`
>   fence (scan outer JIT frames too, like `walk_parked_thread_jit_roots`)
>   did **not** fix it either.
>
> So the value is stale *when the JIT loads it*, despite the JIT having recorded
> it as a root — pointing to either (a) the JIT re-reading the value from a
> stale register/location after the recorded slot was updated, or (b) the value
> being produced stale by a runtime extern that holds a heap pointer in an
> unrooted Rust local/cache across an allocating sub-call. The exact producer is
> not yet isolated. (The single-forwarding-follow trick is unreliable here:
> `CLJVM_GC=every` flips the semi-space many times, so a long-stale pointer's
> forwarding word now lands in reused space.)
>
> ### Fixes actually landed (real, but NOT sufficient for form 430)
>
> Several genuine unrooted-heap-pointer caches were found by inspection and
> fixed (they violated the codebase's own "root heap pointers across allocs"
> contract; all pass forms 1-429 under `CLJVM_GC=every`):
>
> - **LazySeq/Delay cache** (`runtime.rs`): `LazyState.thunk_bits`/`value_bits`
>   are NaN-boxed heap pointers in a Rust `Arc<RefCell>`; the Arc was kept alive
>   but the inner pointers were never GC roots. Added a `LazyRootSource`
>   (`register_lazy_state` + thread-local registry), registered as a permanent
>   extra root source in `Session::new`, mirroring `var_roots::VarRoots`.
> - **ChunkBuffer/IChunk** (`runtime.rs`): `Arc<RefCell<Vec<u64>>>` of element
>   pointers, same hazard — now scanned by the same root source
>   (`register_chunk_buffer`).
> - **`cljvm_inst_reduce_3`** (`reduce1`'s chunked fold): rooted the folded
>   items + running accumulator across each `cljvm_rt_invoke_2` via `with_scope`.
>
> These are correct hardening but do not resolve form 430, so the trigger above
> remains open. Next step: instrument to catch the *producer* of the stale value
> (e.g. a post-GC conservative stack scan that records each stale slot's value,
> looked up when `rt_first` receives the dangling arg) and identify whether it is
> a JIT register read-after-update or a specific unrooted runtime extern local.
>
> Everything below this box is the ORIGINAL (disproven) report, kept for the record.

**Audience:** whoever owns the JIT GC-rooting path
(`dynlower` `Lowerer`/`GreedyRegState` — see update box above; the
`register-alloc` pointers below are a wrong turn).

This is a clean standalone report. The full (messy) investigation trail, including
two wrong turns (a disproven "write-barrier" theory and a *conservative stack scan*
that was implemented and then removed because our GC is precise), lives in
`everypoint-rooting-bug.md`.

---

## One-sentence summary

A heap pointer held in a JIT register/spill slot **across a GC safepoint** is
dropped from that safepoint's stackmap, so the precise copying collector relocates
the object **without updating that slot**, leaving a dangling pointer that later
crashes when dereferenced. The drop is (working theory) a coverage gap in the
register allocator's single-`[min,max]`-interval liveness approximation for loopy
control flow.

---

## Symptom (proven)

Loading upstream `clojure/core.clj` panics at **form 430** — a `defn` whose macro
expansion runs higher-order core fns (`map` / `reduce1`) over closures:

```
clojure-jvm: RT.first on unsupported heap type_id 136   (crates/clojure-jvm/src/runtime.rs, cljvm_rt_first)
```

`type_id 136` (`0x88`) is **not a real type**. Inspecting the object header at the
offending pointer shows **bit 63 set** — the collector's `FORWARDING_BIT`
(`crates/dynalloc/src/semi_space.rs`, `FORWARDING_BIT = 1 << 63`) — and `0x88` is
just the low byte of the forwarding (to-space) address.

So `RT.first` received a pointer to an object the GC **already relocated**, but the
slot holding the pointer was **never updated** to the new location. A textbook
dangling pointer after a moving/copying collection. The from→to delta is exactly
the heap size (16 MiB), i.e. a semi-space flip.

## Repro

```
cd crates/clojure-jvm
CLJVM_GC=every cargo test -p clojure-jvm --test load_upstream_core -- --ignored --nocapture
# stops at form 430: "RT.first on unsupported heap type_id 136"
```

Note: the macro-expansion path is currently hard-coded to `GcPolicy::EveryPoint`
(see `compiler.rs`, `macroexpand_once`), so this also reproduces under the default
policy — the bug is not masked. Under `OnPressure` in normal use it is *latent*:
GC rarely fires in the exact window, so the dangling slot usually isn't created
before the value is consumed.

---

## The GC model (proven — important for the reviewer)

- **It is a plain semi-space copying collector.** `GcConfig::generational(16 MiB)`
  passes `nursery_size: None` (`crates/dynlang/src/lib.rs`, `fn generational`), which
  routes to `Heap::new` (not `Heap::new_generational`) in
  `crates/dynlang/src/gc.rs` (`DynModule::new`). So `has_nursery() == false`; the
  nursery / card table / write barrier are **inert** despite the "generational" name.
  (An earlier "missing write barrier" theory was therefore wrong.)
- **Collections are whole-heap Cheney copies** (`crates/dynalloc/src/heap.rs`,
  `collect_inner`): scan roots → evacuate reachable objects to to-space → set each
  evacuated object's old header to `to_addr | FORWARDING_BIT` → **update every
  enumerated root slot to the to-space address** → swap spaces.
- **The collector is PRECISE.** It updates exactly the slots it is told are roots.
  It does **not** scan raw stack memory guessing which words are pointers. Because
  values are NaN-boxed (a slot may hold a pointer or a raw int), each *known* root
  slot is tag-decoded via `PtrPolicy::try_decode_ptr`; non-pointers are skipped.
  > Do **not** "fix" or "guard" this with a conservative stack scan — that violates
  > the precise-GC contract and cannot even distinguish a dead spill from a live
  > missed root. Any fix/verification must stay within the precise model.
- **Root sources** for a collection (`collect_inner` + the safepoint handler):
  global roots, per-thread shadow-stack roots, and **JIT-frame roots enumerated from
  stackmaps**. The collection at form 430 is triggered by a safepoint poll:
  `crates/dynruntime/src/jit.rs`, `active_jit_safepoint_handler` →
  `mutator_triggered_gc_with_extras` (the whole-heap path).

## How JIT-frame roots are enumerated (proven)

- Each JIT function records a `SafepointRecord { return_offset, root_slots }` for
  every call site and explicit safepoint poll
  (`crates/dynlower/src/batch_lower.rs`: `record_call_return_safepoint`,
  `Inst::Safepoint`, and `collect_live_root_slots`).
- `root_slots` are the **spill-slot frame offsets** of the vregs the allocator says
  are live at that safepoint (`alloc.stackmaps[inst_id]`, filtered to `SpillSlot`
  locations).
- At GC time, the FP-chain walker
  (`crates/dynlower/src/lib.rs`, `walk_jit_ancestor_roots`) finds each JIT frame by
  its return address, looks up the matching `SafepointRecord`, and visits exactly
  those `root_slots`. The collector then updates them.

**Conclusion:** if a live heap pointer's vreg is **not** in `alloc.stackmaps` for a
safepoint, its slot is never visited, never updated → dangling after relocation.

---

## Root cause (working theory)

The crash value reaches `RT.first` as a **JIT-level argument** (confirmed: a runtime
backtrace shows `cljvm_rt_first` is called directly from JIT, no Rust frame above
it). So the dangling slot is a **JIT vreg live across a call-return safepoint** —
defined before a call, live across it, used after — that is **absent from that
safepoint's stackmap**.

The stackmap is built in the external `register-alloc` crate
(`register-alloc/src/linear_scan.rs`, `build_stackmaps`). For each safepoint it
records the vregs whose **live interval** covers the safepoint position:

```rust
// build_stackmaps, paraphrased
if pos < interval.start || pos > interval.end { continue; }   // "not live here"
```

Liveness is computed in `register-alloc/src/liveness.rs`:
1. Correct **backward dataflow** producing per-block `live_in` / `live_out` sets.
2. **Projection to a single contiguous `[min, max]` interval per vreg** over
   RPO-linearized instruction positions (the comment calls it "the conservative
   single-interval approach").

That projection assumes a vreg's live region is **contiguous in linear (RPO) block
order**. When control flow does not match the linear layout — loops/back-edges, or
a block whose linear position falls outside `[min(def), max(use)]` — a safepoint
*inside* such a block can land **outside** the vreg's `[min,max]` interval even
though the vreg is genuinely live there in real control flow. The vreg is then
judged "not live" at that safepoint and omitted from the stackmap.

Form 430's `defn` macro expansion runs `map`/`reduce1` over closures, i.e. exactly
the loopy CFG that exercises this.

### Why we believe it is a *liveness-coverage* gap (not a record-policy gap)

- Forcing the allocator to `SpillAndRecord` **every** vreg type at safepoints
  (`crates/dynlower/src/regalloc_bridge.rs`, `safepoint_action`) did **not** fix the
  crash. `build_stackmaps` only even *considers* a vreg if its interval covers the
  safepoint; so a no-effect from "record everything" means the vreg's interval does
  not cover the safepoint — a coverage gap upstream of the record decision.
- This matches a previously-noted cross-block-liveness issue in this allocator (it
  had earlier broken the `ns` macro).

---

## Proven vs. not proven (please don't take the theory as fact)

**Proven**
- The crash is a dangling pointer to a relocated object (forwarding bit set).
- The collector is a precise semi-space copier; it only updates enumerated roots.
- The dangling slot is reached as a JIT vreg argument at a call-return safepoint.
- It is a liveness-*coverage* problem, not a record-policy problem (spill-all-types
  experiment had no effect).

**Not yet proven**
- The exact vreg and exact CFG shape that triggers the interval miss. On paper the
  dataflow is correct and the interval-construction step (`liveness.rs`, step 3)
  extends coverage to every block where a vreg is `live_in`/`live_out`, so a simple
  live-through vreg *should* be covered. The precise discrepancy has not been
  isolated; it needs the assertion below to surface it.

---

## How to confirm precisely (recommended next step — no stack scanning)

Add a **debug assertion inside `register-alloc/src/linear_scan.rs::build_stackmaps`**:
at each safepoint instruction, compute the set of vregs that are **live across** that
safepoint from the trusted `live_in`/`live_out` dataflow (a value is live-across if
it is in `live_out` of the safepoint's position, or live-in to the block and used at
or after the safepoint). Assert that every **GcPtr/I64** vreg in that set is present
in the stackmap just built. The first violation is the bug — reported deterministically
at allocation time, naming the exact vreg and safepoint, entirely within the precise
model (no GC needed, no memory scanning).

Once the exact vreg is known, the fix is in `liveness.rs`: make the interval (or the
stackmap-liveness test) cover every safepoint the vreg is live across — e.g. drive
the safepoint-liveness decision directly off `live_in`/`live_out` instead of the lossy
single `[min,max]` interval, or split/extend intervals so they are not assumed
contiguous in linear order.

The same assertion, kept as a debug-build check, is the permanent guard: it catches
any future missed-root regression at allocation time rather than as silent
moved-object corruption far away.

---

## Key file pointers

| What | Where |
|---|---|
| Crash site / forwarding-bit detail | `crates/clojure-jvm/src/runtime.rs` — `cljvm_rt_first` |
| Heap = semi-space (no nursery) | `crates/dynlang/src/lib.rs` `fn generational`; `crates/dynlang/src/gc.rs` `DynModule::new` |
| `FORWARDING_BIT` | `crates/dynalloc/src/semi_space.rs` |
| Whole-heap collection | `crates/dynalloc/src/heap.rs` — `collect_inner`, `mutator_triggered_gc_with_extras` |
| Safepoint → collection | `crates/dynruntime/src/jit.rs` — `active_jit_safepoint_handler` |
| Stackmap build / root-slot recording | `crates/dynlower/src/batch_lower.rs` — `record_call_return_safepoint`, `collect_live_root_slots`, `Inst::Safepoint` |
| JIT-frame root walk at GC | `crates/dynlower/src/lib.rs` — `walk_jit_ancestor_roots` |
| `safepoint_action` (which vregs to record) | `crates/dynlower/src/regalloc_bridge.rs` |
| **Liveness + single-interval projection (suspected root)** | `register-alloc/src/liveness.rs` — `compute_liveness` |
| **Stackmap liveness test** | `register-alloc/src/linear_scan.rs` — `build_stackmaps` |

(`register-alloc` is at `claude-experiments/register-alloc`, a path dependency of
`dynlower`.)
