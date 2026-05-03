# 02 — GC-Rooted Temporaries Across `gc_alloc`

## Implementation status: ✅ Implemented

- **Code**: `DynFunc::fresh_slot_name`, `DynFunc::with_rooted`, and the `RootedSlot` type in [`crates/dynlang/src/lib.rs`](../../crates/dynlang/src/lib.rs).
- **Tests**: `tests::fresh_slot_name_is_unique_and_with_rooted_reloads` (verifies uniqueness + reload semantics).
- **Migration**: beagle dropped `arr_lit_counter`, `push_counter`, and `hoist_counter` from `Lowerer`; the four `__beagle_*__` slot-naming conventions are gone. `lower_array_literal`, `lower_push`, and `StructCreation` use `with_rooted`; the LICM hoist uses `fresh_slot_name`.
- **Differs from plan**: the `with_rooted_safepoint` convenience and `rooted_slot` (single-slot variant) weren't needed in practice — the closure-style API with explicit `safepoint(&[])` covers all four sites.

## Problem

Any value live across a `gc_alloc` (or any safepoint) must reside in a slot
the GC root scanner can find. With NanBox + a moving collector, the toolkit's
root manager scans named stack slots — so the embedder pre-evaluates every
operand into a fresh `def_var(name, v)`, calls the alloc, then re-reads
through `get_var(name)` to pick up any forwarding the GC performed.

This works, but the slot-name minting is hand-rolled at every call site, with
a per-`Lowerer` counter to avoid name collisions in nested expressions.
Beagle has four such counters and four near-identical bodies:

- **Array literals** ([`lower.rs:1007-1041`](../../crates/beagle/src/lower.rs#L1007))
  — `__beagle_arr_{id}_elem_{i}__` + `arr_lit_counter`.
- **`push`** ([`lower.rs:1048-1109`](../../crates/beagle/src/lower.rs#L1048))
  — `__beagle_push_src_{id}__` / `__beagle_push_val_{id}__` +
  `push_counter`.
- **Struct creation** ([`lower.rs:825-869`](../../crates/beagle/src/lower.rs#L825))
  — `__beagle_tmp_{fname}__`. Note: this one *doesn't* have a counter, which
  is a latent bug — two creations with the same field name in nested form
  would alias.
- **`while`-LICM `length` hoist** ([`lower.rs:780-789`](../../crates/beagle/src/lower.rs#L780))
  — `__hlen_{x}_{n}__` + `hoist_counter`.

The boilerplate isn't just verbose — getting it wrong is silent corruption.
Forgetting to reload after the alloc means using a from-space pointer; the
struct-creation site only avoids it because the only post-alloc operation
(`store(val, raw, off)`) reads `val` from a slot that was already pinned.

## Proposed API

A scoped helper that hides the slot allocation, the safepoint, and the
post-alloc reload:

```rust
// Pin `[v1, v2]` across whatever happens inside the closure. The closure
// receives reloaded Values that reflect any forwarding done by allocations
// inside it.
let new_obj = f.with_rooted(&[v1, v2], |f, rooted| {
    let raw = f.gc_alloc(ty, len);
    f.fb.store(rooted[0], raw, off1);
    f.fb.store(rooted[1], raw, off2);
    raw
});
```

Implementation: each input `Value` is bound into a uniquely-named slot
(`__rooted_<auto-id>__`); the closure runs; on entry to each `gc_alloc` /
safepoint, the slot is the source of truth; the helper re-reads each slot
once at the start of the closure and returns those reloaded handles.
Variants:

- `f.with_rooted_safepoint(&[...], |f, rooted| { ... })` — emits the
  `safepoint(&[])` for you (the current pattern at every site).
- `f.rooted_slot(v) -> RootedSlot` — for cases where the rooting scope
  spans multiple lexical regions (`lower_push` reloads after the alloc *and*
  inside the copy loop). Drop semantics frees the slot name.

## Implementation plan

1. **`DynFunc::scoped_slot_name() -> String`.** Internal helper that mints
   a guaranteed-unique slot name. Today every embedder reinvents this with
   a per-Lowerer counter; it belongs on `DynFunc` so collisions across
   inlined frames are also impossible.

2. **`with_rooted` / `rooted_slot` on `DynFunc`.** Implementation is small
   (~30 lines). The reload-after-alloc behavior should be automatic: if the
   closure body calls `gc_alloc` or `safepoint`, on the next read of any
   `RootedSlot` the helper re-emits a `get_var`. (Easiest: re-read on every
   `RootedSlot::get(&self, f)` call. Optimal: cache the SSA value and
   invalidate on safepoint emit.)

3. **Beagle migration.** Delete `arr_lit_counter`, `push_counter`,
   `hoist_counter`, the `__beagle_*__` slot names, and the four
   `def_var`/`get_var` boilerplate blocks. Estimated delta: −80 LOC, plus
   closes the latent struct-creation aliasing bug.

## Open questions / risks

- **Closure ergonomics.** Returning `[Value; N]` of reloaded values is
  cleanest when N is fixed. For variadic (e.g. `lower_array_literal` with
  N elements known only at runtime), a `Vec<Value>` is fine but loses the
  array-destructure pattern. Acceptable.
- **Nested calls.** A `with_rooted` inside another `with_rooted` must keep
  outer slots live. As long as slot names are globally unique per
  `DynFunc`, this works without special handling.
- **Non-NanBox embedders.** If a future frontend uses tagged pointers
  rather than NanBoxes, the rooting policy differs (ptr-only vs. all
  values). The helper should be parameterized by the
  `ExecutionConfig::PtrPolicy` already in the module, not hardcode NanBox
  semantics.
