# Stage I — generational GC (nursery + card-marked old gen)

Decision (Jimmy, 2026-07-15): "We need a generational gc. We have tons of
examples lying around." This is gap #1 after Stage H proved the wall is
allocation throughput: a 4 GiB non-generational semi-space bumping through
virgin pages, and every collection copying the whole live set, versus a JVM
TLAB in a nursery. `MICROLANG_HEAP_MB=256` alone bought vecbuild 57→48 on
page locality — the structural fix is a nursery.

## The examples we are copying from (READ THESE FIRST)

1. `claude-experiments/gc-rust/crates/gcrust-rt/src/gc/card_table.rs` — the
   shape we already share a lineage with (Stage D came from this runtime):
   512-byte cards (`CARD_SHIFT = 9`), ONE byte per card, plain idempotent
   store to dirty, bounds-checked index, `iter_dirty`, `clear_all` under
   STW. Its `barrier.rs` SATB machinery is for CONCURRENT collection — we
   are STW, so we do NOT need SATB or read barriers. Ignore that half.
2. **beagle's `src/gc/generational.rs` — and above all the post-mortem in
   the memory note `project_beagle_concurrent_gc_write_barrier`.** That is a
   *crash report* for the exact bug this stage can ship. Its lessons are
   binding here:
   - The write barrier MUST be lock-free. The card byte array is the SINGLE
     SOURCE OF TRUTH, marked with one idempotent `Relaxed` `AtomicU8` store.
     Concurrent same-value marks are benign; visibility comes from the STW
     safepoint, not from a per-barrier fence. **Jimmy explicitly rejected a
     write-barrier mutex.**
   - NO auxiliary `Vec`s in the barrier. beagle's original kept
     `dirty_card_indices` + a `remembered_set` `Vec` and pushed to both per
     write: a real data race under ≥2 threads → lost old→young edges → a
     young object referenced only from old gen is never promoted → later
     reads hit stale/forwarded pointers. Symptom was a mystery crash in
     unrelated code. Find dirty cards by SCANNING the card array at STW
     instead (race-free, and the table is 1/512 of the heap).
   - Scan the card array WORD-AT-A-TIME: read 8 card bytes as one `u64` and
     skip the whole run when `== 0`. beagle's byte-at-a-time + per-GC
     `HashSet` was pathological under gc-stress.
   - Mark the OBJECT BASE, not the field address. Then a dirty card always
     has a live object starting in it, which is what makes the card→object
     lookup tractable (and a dirty card with no object start is a hard
     panic, never a silent skip).
   - Young gen is FULLY evacuated every minor GC, so no old→young edge
     survives a minor → clearing ALL cards afterwards is correct.
     HotSpot-style selective card *cleaning* is moot. (This is why we
     promote on first survival — see below.)
   - beagle needed a Block Offset Table because its old gen is a
     free-list mark-and-sweep. OURS IS COPYING/COMPACT — objects in the old
     space are contiguous from `base` to `used`, so "find the object starts
     in this card" is a forward walk from a known start. We still need a
     per-card start index (same idea, far simpler to maintain: record the
     offset of the first object that BEGINS in each card as we bump-allocate
     / as we promote, and rebuild it exactly during evacuation).
3. `scry` (oo-lang) shipped a precise STW generational GC — consult if a
   question arises the above two don't answer.

## Design

### Spaces
- **Nursery**: one bump space (default 32 MiB, `MICROLANG_NURSERY_MB`).
  ALL allocation goes here — `Heap::alloc` and the JIT's inline AllocWindow
  fast path (the window points at the nursery; the D5 inline sequence is
  unchanged, it just bumps a different space).
- **Old gen**: the existing semi-space PAIR (`MICROLANG_HEAP_MB` per space),
  with the existing Cheney evacuation as the MAJOR collection.
- Objects are allocated young and PROMOTED ON FIRST SURVIVAL (beagle's
  model). No survivor spaces, no age field — the header's `spare` u16 stays
  free. Rationale: it makes "nursery is empty after a minor GC" an
  invariant, which in turn makes card clearing trivially correct and gives
  us the verify check below. Aging is a tuning refinement for later; get
  the invariant first.

### Minor GC (the new hot path)
Roots = the existing enumeration (globals, shadow stacks, dyn stacks, parked
mutators' published roots/envs, consts, method impls, arglists, `()`
singleton, kont/future registries, live envs, live kont, native stack-map
roots) **plus the dirty-card scan of the old gen**. For each root slot: if
it points into the NURSERY, copy the object to the old gen (bump), install
the FORWARDING_BIT header, rewrite the slot. Cheney-scan the promoted
objects in old space (their fields may point at nursery objects that must be
promoted transitively). Afterwards: nursery cursor = 0 (empty), ALL cards
clean, old-gen card-start index updated for the promoted range.

Because promotion appends to the old space, a minor GC can exhaust it →
trigger a MAJOR GC (the existing full Cheney over the old gen, which is
already written and tested). Order: minor first, then major if the old space
is over its own threshold.

### Major GC
Unchanged from Stage D/E: Cheney over the old-gen semi-space pair, flip and
reuse. The nursery is empty at that point (a minor runs first), so the major
only sees old objects. After a flip the card table is rebuilt against the
new active space's base and cleared (everything was just scanned).

### The write barrier
Needed on EVERY store of a pointer into an object that might be OLD:
`Gc::set_field`, `values_mut`/`arr_slice_mut` element stores, the atom
store/CAS (`%atom-set`/`%atom-cas` — this is beagle's exact crash shape: a
long-lived atom `swap!`-ed to a fresh young value every frame), `arr_extend`
(re-points the handle's blob field), the F3 transient in-place edits (a
transient can be promoted mid-edit, so its subsequent `assoc!` writes are
old→young), and the JIT's inline `aset` arm.

NOT needed for the initializing stores of a freshly allocated object: it is
in the nursery by construction, so those are young→anything. This is what
keeps the D5 inline allocation sequences (`%cons`, closure creation) barrier
free — the hot paths pay nothing.

Barrier implementation (lock-free, no Vec, no branchy generation test):
mark the card for the OBJECT BASE with one relaxed `AtomicU8` store, guarded
by a single UNSIGNED bounds compare against the old space
(`(addr - old_base) as usize < old_size` catches both "below base" and
"above limit" in one compare — the same trick the D5 `aget` bounds check
uses; a nursery address wraps to a huge offset and is skipped). Cards live
in a `Vec<AtomicU8>` (same layout as `Vec<u8>`, so the JIT's inline store
of a raw byte is valid — beagle pins this exact property).

JIT: `emit_card_mark(obj_addr)` = load old_base + old_size + card_base from
the RunCtx (mirror them next to the AllocWindow, re-pointed at flips under
STW), one subtract, one unsigned compare, shift, byte store. Wire it into
the inline `aset` arm. Everything else routes through shims that call the
Rust barrier.

### Verify mode: the missed-barrier detector (the reason this is shippable)
A missed write barrier is silent and catastrophic — it is exactly the bug
beagle shipped and then had to hunt through an unrelated crash. Make it
LOUD: when verify is armed, after every minor GC walk the ENTIRE old gen and
assert that NO old object holds a pointer into the nursery range. The
invariant is exact (a minor fully evacuates the nursery), so any surviving
old→young pointer means an edge was missed — panic naming the object, its
type, the slot offset, and the target. Combined with nursery poisoning, a
missed edge dies at the collection that caused it instead of a mystery
corruption later. This check is O(old gen) so it is verify-only, but the
gc-stress battery must run WITH it armed.

## STATUS

**I2 LANDED (2026-07-16): minors are LIVE.** The pressure safepoint runs a
minor (`GcKind::Auto`), escalating to a major on `MinorOutcome::needs_major` or
when `minor_will_fit()` says no; the explicit `(gc)` prim stays a MAJOR (tests
and the bench battery are written against "collect everything now"). `relocated`
— the IC epoch — is bumped by minors too, because promotion is a copy; one bump
covers a minor+major pair (nothing reads it between them, the world is stopped).
gc-stress = a minor at EVERY safepoint + a major every 8th (the majors are what
keep the flip and the card-table rebuild under the hammer).

The barrier is routed at THREE CHOKE POINTS rather than at each store, so it
cannot be forgotten: `Runtime::arr_slice_mut` (marks the DATA BLOB — that is
where array elements live, not the handle), `Runtime::values_mut` (marks the
record/values object; this is what covers the F3 transients, promoted mid-edit),
and `Runtime::arr_extend` (marks BOTH: the blob it writes elements into, and the
handle whose field it re-points on the grow path). Plus `%atom-set`/`%atom-cas`,
which take the atom's base explicitly. Fresh-object initializing stores stay
barrier-free — the D5 inline alloc paths pay nothing. Each choke point was
mutation-tested: deleting its barrier makes the detector fire, naming the
object and the slot.

**I3 LANDED (2026-07-16): the JIT marks its own cards.** `%aset` is RE-INLINED
and the de-inline's cost is paid back — `reduce-map` 35→27 ns/op (better than
I1's 28; before/after measured on one machine against the same bench).

`emit_card_mark(obj_base)` is `CardTable::mark` emitted straight-line: load
base/size/cards, sub, ONE unsigned compare, shift, relaxed byte store. No
generation test — a nursery address wraps to a huge offset and fails the
compare, exactly as in Rust.

The three inputs are NOT a copy. `CardTable`'s first field is now a `#[repr(C)]`
`CardWindow` (base/size/cards at 0/8/16) that `mark` itself reads, and the JIT
holds its address in the RunCtx beside the `AllocWindow` pointer. A mirror the
Rust barrier did not itself read could drift from it silently, and "the JIT
marked a card in the wrong table" is the exact class of bug this stage exists to
make impossible. `base` follows the major flip (`CardTable::point_at`, STW) and
emitted code re-reads it per mark, so there is one re-point, not two.

The inline arm marks the DATA BLOB — the object that actually holds the words —
not the ARRAY handle it dereferenced to get there. That agrees with
`arr_slice_mut` by construction, and the mutation test proves it: deleting the
`emit_card_mark` call makes the detector name `old-gen array-data ... slot +16`,
the blob at the index the program wrote.

Audited for other emitted stores that could hit an old object: there are none.
The only other emitted stores are frame slots (`cur_slots` — Rust-side, traced
as roots, not heap objects) and the writes into `emit_alloc`'s fresh address
(inline cons head/tail, closure meta/code/captures). Those are nursery by
construction, so they stay barrier-free — that is the whole point of the design,
and adding a mark there to "be safe" would be a regression, not caution.
`INLINE_OBJECTS` still gates all of it: NanBox/HighBit route `%aset` through the
barriered `arr_slice_mut` shim, unchanged.

Proof: `tests/jit.rs::jit_inline_aset_marks_its_card` — a promoted vector, a
`%aset` of a fresh young cons from JIT'd code, then a minor forced through the
pressure path. It asserts a minor ran DURING the build (so `v` really was
promoted and the store really was old→young) and another AFTER the last store
(so something really had to find the young value through the card table); either
missing and the test would pass having proved nothing. The explicit `(gc)` prim
is useless here and is not used: it is a MAJOR, which traces the old gen from
the roots and finds the edge whether or not its card was marked. Only a minor
consults the table.

### Two PRE-EXISTING gaps found while gating I2 (NOT Stage I bugs) — NOW CLOSED
Both were rooting holes reached from the **clojure-stub frontend** — bare `u64`
values held across a macro expansion / a lazy-seq force, which evaluate code,
which reach a safepoint, which collects and relocates them. Both reproduced at
I1 (779529155), before minors existed, so they bounded what the detector could
be pointed at rather than being caused by it:
1. `MICROLANG_GC_STRESS=1` could not boot `clojure.core` at all — `Session::new`
   died in `compile_fn` with "fn: params must be symbols" (a relocated param
   list).
2. Even on the pressure path, a low trigger use-after-moved: for EVERY program
   on the JIT tier, and for `(reduce (fn [acc i] (assoc acc i ..)) {} ..)` on
   TreeWalk.

The audit that closed them found the causes were only PARTLY in the frontend;
the rest were in the toolkit, exposed the moment the frontend stopped dying
first. In the frontend (`clojure-stub/src/roots.rs` — a `RootVec` shadow-stack
handle; `expand_each` takes `&mut RootVec`, so a caller with a bare `Vec<u64>`
cannot call it): the expander's element walks, `rebuild_binder`'s param/binding
lists, and the `require` path (a load RUNS a file). In the toolkit: `seq_step`/
`seq_flatten` accumulated bare values across the seq fn's invoke; the
`Prim::Apply`, `Prim::CallEc` and `Ir::Dispatch` arms invoked WITHOUT publishing
the caller's env (only `Ir::Call` did), so a collection inside the callee never
traced the caller's still-live frame; `Ir::Try` held both the body's result and
the suspended signal's payload bare across an allocating `finally`; and the
pending signal was not a root at all.

Result: `gc_generational.rs`'s JIT half is un-`#[ignore]`d and the `reduce`/
`assoc` map shape is back in its battery, and the hammer now swings at the real
standard library — `clojure-stub/tests/gc_stress_library.rs`.

## Phases (each suite-gated)
- I1: `heap.rs` — nursery space, promotion/minor evacuation, card table
  (word-at-a-time dirty scan, per-card object-start index rebuilt during
  promotion), the barrier entry point, the verify walk. Unit tests in the
  heap's own style (the D1 tests are the model): promotion, transitive
  promotion, dirty-card discovery, missed-barrier detector fires, minor
  under a full nursery triggers major, cycles, varlen objects.
- I2: runtime — route all heap field stores through the barrier; minor at
  the pressure safepoint; `MICROLANG_GC_STRESS` = minor at every safepoint
  (+ a major every N); counters (minor/major/promoted bytes).
- I3: JIT — AllocWindow points at the nursery; `emit_card_mark` + the inline
  `aset` arm; the RunCtx's `CardWindow` pointer (base/size/cards). DONE.
- I4: gates + measurement. Targets: vecbuild ≤45, group-by ≤160, no core
  band regression, and the suite-order effect (Stage F scoreboard's
  interleave 3000 vs 1865 isolated) should largely vanish because
  collections stop copying the accumulated live set.

## Non-goals
Concurrent/incremental collection (STW stays — the park rendezvous is the
contract everything else is built on). SATB/read barriers. Survivor spaces
and aging. A remembered set separate from the card table (beagle proved it
redundant AND racy).
