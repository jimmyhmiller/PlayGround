# The view layer (docs/02) — implementation status

## Proper linear types: the `linear` declaration (DONE)

Linearity is now a **declarable property of a type**, not a hardcoded special case.
`linear postulate T : …` and `linear enum T { … }` mark `T` as a resource: its
values may not be dropped (leak) or duplicated (double-use). Recorded in
`Signature.linear_types`; `contains_linear` consults it, so an un-annotated binder
of a linear type defaults to multiplicity 1.

This does two things: (1) it **closes a soundness hole** — before, only `Own` was
recognized, so a bare (un-annotated) binder of a view type `PtsTo l a` silently
defaulted to ω and could be LEAKED; now the prelude marks `Own` and `PtsTo`
`linear`, and a bare view binder correctly rejects. (2) it makes the language a
**proper linear type system**: users declare their own linear resources — file
handles, sockets, capabilities, session channels — and get acquire/use/release
enforcement (no leak, no double-use) for free, with the multiplicities erased at
runtime. A datatype that *contains* a linear field is itself linear (transitive).
See `examples/linear_resource.tal` and the `linear_resource_*` / `bare_view_binder_*`
tests. Remaining expressiveness gap: no multiplicity POLYMORPHISM yet (one
combinator generic over 1/ω) — structural linear combinators already subsume the
common case (a body that uses each field once type-checks at any multiplicity).



The view layer is the part of the design flagged as the "genuine research risk"
(FUTURE_WORK §13): C-level memory operations (raw pointers, strong update, manual
free) made memory-safe by the L3 address/permission split. This doc tracks what is
actually built.

## Slice 1 — the L3 split, TYPE-LEVEL (DONE)

The docs/02 §1–3 discipline is expressible and **sound at the type level in the
existing QTT core**, with no kernel changes — validated by `examples/views.tal`
and the `view_*` tests. The primitives are trusted postulates (the same method
that introduced `Own`/`alloc`/`free` in v1.1):

```
postulate Loc   : Type                    -- static locations, erased in use
postulate Ptr   : Loc -> Type             -- a COPYABLE (ω) address naming l
postulate PtsTo : Loc -> Type -> Type     -- `a @ l`, the LINEAR (1) permission

enum Cell (a : Type) {                     -- ∃ l. (Ptr l ⊗ a @ l)
    MkCell : {0 l : Loc} -> (p : Ptr l) -> (1 v : PtsTo l a) -> Cell a
}
postulate valloc : {0 a} -> a -> Cell a
postulate vwrite : {0 a}{0 b}{0 l} -> Ptr l -> (1 PtsTo l a) -> b -> PtsTo l b  -- STRONG UPDATE
postulate vfree  : {0 a}{0 l}     -> Ptr l -> (1 PtsTo l a) -> Unit
```

What this establishes:

- **The alias/permission split** — `Ptr l` is `ω` (copy it freely; holding one
  grants no access), the view `PtsTo l a` is `1`. A pointer used many times with
  the view threaded once type-checks; the safety comes from the view, not the
  address.
- **Strong update** — `vwrite` changes the stored type `a → b` in place. Sound
  because the view is linear: no stale `a @ l` can survive. Rust cannot express
  this directly.
- **Safety by linearity, no bespoke analysis** — leak (view dropped, `0 ⋢ 1`),
  double-free and use-after-free (view reused, `ω ⋢ 1`) are all *type* errors,
  discharged by the same QTT checker as everything else. The existential location
  is opened by pattern-matching `MkCell`.

## Slice 2 — codegen: the view layer RUNS END TO END (DONE)

The primitives are now first-class **built-in prelude** members (`src/rust_surface.rs`
`PRELUDE`) with real LLVM lowering (`src/dep_codegen.rs` `compile_postulate`), so
`examples/views.tal` type-checks, JIT-runs, and AOT-links to a native executable.

Runtime representation: a location is erased; `Ptr l` and the linear view
`PtsTo l a` are BOTH the raw cell address (linearity guarantees at most one live
view, so a shared address is sound). Lowering:

- `valloc(x)` → `malloc` one slot, store `x`, pack the address as both the pointer
  and the view (`box_single_ctor("Cell", "MkCell", [addr, addr])`).
- `vwrite(p, v, new)` → store `new` at the cell; return the (same) address as the
  new view — a real in-place STRONG UPDATE (the stored type may change).
- `vread(p, v)` → load the payload, `free` the cell, return the payload
  (destructive read).
- `vfree(p, v)` → `free` the cell.
- `vtake(p, v)` → MOVE the payload out, retyping the slot `Hole`: load the value
  and return `(value, PtsTo l Hole)` (`Taken a l`), WITHOUT freeing. Sound for any
  payload — a move, not a copy, so no double-own. The `Hole` view must be refilled
  (`vwrite`, which is universally quantified over the old type so `Hole` refills)
  or reclaimed (`vfree`) — forgetting it is a leak. This is docs/02's `take`, and
  it gives read-modify-write without destroying the cell (the sound alternative to
  a borrowing read, which would need `a` copyable).

Verified: `view_alloc_write_read_runs` (→ 2), `view_strong_update_changes_type_runs`
(Bool cell → Nat 5, → 5), `view_take_modify_write_read_runs` (take 2, write 3,
read → 3), `view_take_then_free_hole_runs` (→ 1), plus the type-level safety tests
(leak / double-free / use-after-free / take-then-leak-the-hole all rejected).
`tally run examples/views.tal → 2`; `tally build` produces a working native binary.

Honest limitations that remain:

- No BORROWING `read` (value copied out, view kept) — that needs a copyability
  constraint on `a`, which the type system cannot yet express. `vtake` (move-out)
  is the sound substitute for read-modify-write.
- No separating conjunction `⊗` / disjointness yet.
- Naming: `valloc`/`vwrite`/`vread`/`vtake`/`vfree` (the prelude's `alloc`/`free`
  are the older `Own`-based primitives with their own specialized lowering).

## Next slices

- **Borrowing `read`** — needs a copyability predicate on types (usable at ω),
  which does not exist yet; `vtake` covers the sound move case in the meantime.
- **Separating conjunction `⊗`** + disjointness — the first real consumer of the
  stratum-(A) **location (in)equality** decision (docs/03), which was deferred
  precisely because it had no consumer until now.
- **Recursive/user-defined views** (`dataview`) for linked structures (docs/02
  §5.5) — generalizing the canned DLL to arbitrary heap layouts.
