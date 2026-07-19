# Phase B — inline value layout (byte-level zero-leak = actually-as-low-as-C)

**STATUS: PLAN — design-first, for the Leader's scope/soundness sanity-check before
building (the established pattern for a substantial change).**

## Goal

Close the gap between "type-level-safe" (proven by the differentiator demo) and
"genuinely manual memory, as low as C" (the bytes ARE reclaimed). Today the all-boxed
codegen DOUBLE-boxes `Own`-of-struct: `alloc(Node{h,t})` builds the `Node` struct as its
own cell (`compile_constr`), then `alloc` boxes the POINTER in a second `Own` cell.
`unbox` frees the `Own` cell; the inner `Node` struct cell is freed by NOTHING (the match
reads it but never frees it) → a byte-level leak (IR: 5 mallocs, 2 frees, for the 2-node
demo). The TYPE level is fully safe; the codegen isn't yet leak-free.

After Phase B: ONE cell per node (the struct's fields stored INLINE in the `Own` cell),
reclaimed in full by the consuming traversal → the demo runs with mallocs == frees.

## The three changes (minimal — scoped to the demo's shape; general layouts later)

1. **Inline construction — `alloc(Constr(…))` builds the struct IN ONE cell.** Today
   `alloc` receives an already-boxed struct pointer. Fuse construction+alloc: when the
   `alloc` argument is a constructor application, `malloc` N slots (N = field count) and
   store the fields INLINE, returning that pointer as the `Own` cell (no second box). So
   `Own (Node-struct)` = a pointer to a single inline `[head, tail]` cell.

2. **`unbox` of an inline struct is the IDENTITY (does NOT free).** With the struct inline,
   the `Own` cell IS the struct cell, so `unbox(o : Own struct)` returns `o` as the struct
   pointer and must NOT free it (the struct is still live, being matched). Freeing moves to
   the match (change 3). `unbox` of a SCALAR payload (`Own Nat`) keeps today's behavior
   (load the scalar, free the cell — there is no following match to reclaim it). So
   `unbox`'s lowering branches on payload kind (inline-aggregate vs scalar).

3. **Linear eliminator FREES the box.** When a `match`'s scrutinee TYPE is linear
   (`type_is_linear` — the codegen already has the type), the value is consumed exactly
   once (linearity guarantees it), so after reading the fields, FREE the scrutinee's cell.
   This reclaims the single inline cell (and is the general "a consumed boxed value's cell
   is reclaimed" rule). A non-linear (`ω`) scrutinee may be matched again elsewhere → do
   NOT free (unchanged).

Net for the demo: 1 malloc + 1 free per node; `unbox` is identity; the match reads `[h,t]`
then frees the one cell. As low as C.

## Soundness

- **No use-after-free:** the eliminator READS all fields BEFORE freeing the cell (same
  load-before-free ordering the demo reviewer verified for `unbox`). The moved-out linear
  fields (`tail`) are DISTINCT cells (a separate node's inline cell), never pointers into
  the freed cell.
- **Free-iff-linear is sound:** a linear-typed scrutinee is consumed exactly once (the
  type-checker proved the program; linearity ⇒ this match is its sole consumer), so its
  cell is dead after the match → safe to free. A non-linear scrutinee is left alone
  (it may be shared) — no double-free.
- **The kernel/type-checker is unchanged** (Phase B is codegen-only): the type-level
  safety the foundation proved still holds; Phase B only makes the BYTES match the types.

## Red-team (bring with the implementation)

- the demo runs `= 3` AND mallocs == frees (byte-level zero-leak — assert via an
  allocation counter or IR inspection: N nodes ⇒ N mallocs, N frees);
- a non-linear (`ω`) matched value is NOT freed (no double-free when shared/re-matched);
- nested/recursive inline structs reclaim fully (every node's single cell freed once);
- the type-level red-team (leak/double-free rejected) is UNCHANGED (codegen-only change).

## Scope / open question for the sanity-check

- v1 scope = the demo's shape (single-constructor inline structs; `Opt`-of-`Own`
  recursion). General layouts (multi-variant enums inline, real size/align/offsets,
  niche/null-pointer opts) are the FULL Phase B, layered after the demo is byte-clean.
- KEY DECISION to confirm: free-iff-linear-TYPE in the eliminator codegen (using
  `type_is_linear` on the scrutinee type) vs threading the checker's USAGE to codegen.
  Lean: the TYPE test (codegen already has the type; a linear type ⇒ consumed-once ⇒ safe
  to free; no new plumbing). Confirm this is sound + sufficient, then build.
