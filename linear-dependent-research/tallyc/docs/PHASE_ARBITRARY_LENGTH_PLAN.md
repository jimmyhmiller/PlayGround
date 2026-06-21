# Phase #1 — arbitrary-length recursion/iteration over LINEAR structures

**STATUS: PLAN — verify-before-building turned up that this is BIGGER than "thread the
accumulator's multiplicity." Three sub-issues; the headline one is a soundness-sensitive
totality extension. Bringing the design for the Leader's steer before building.**

## Goal

Let a real program recurse/iterate over an owned (linear) structure of ANY size — most
importantly, TRAVERSE + FREE an arbitrary-length owned linked list/tree. The
differentiator demo unrolls a FIXED length precisely because this is missing; every real
program (the interpreter) needs unbounded recursion over linear data.

## What I found (probing the obvious fuel approach)

The natural attempt is a fuel-driven accumulator fold (1a′) whose accumulator is the
linear structure:

```
sumFree : Nat -> Opt (Own Node) -> Nat
fn sumFree(fuel, l) { match fuel {
  Zero => Zero,
  Succ(f) => match l { none => Zero, some(o) => match unbox(o) { Node(h, t) => add(h, sumFree(f, t)) } } } }
```

Three distinct blockers surfaced, in order of depth:

1. **Linear-accumulator multiplicity (small, known fix).** The 1a′ motive Pi chain
   (`elab_nat_match_acc`, rust_surface.rs ~1940) hardcodes `Mult::Omega` for every
   accumulator. A LINEAR accumulator passed at `ω` scales its usage to `ω` ⇒ `ω⋢1`. Fix
   = thread the accumulator's REAL quantity (`Mult::One` when `type_is_linear`). Verified
   this flips the error on a simple-Var varying accumulator (and does NOT regress the 118
   suite). One line — but NOT sufficient alone (below). (Reverted for now; lands with the
   rest.)

2. **The fuel base case can't free a linear list (semantic, fundamental to fuel).** With
   the mult fixed, `sumFree` now fails `0⋢1`: when fuel runs out, the `Zero` arm DROPS the
   remaining linear list `l` (a leak). The base case would have to CONSUME an
   arbitrary-length remaining list — which itself needs recursion (more fuel). Circular.
   So **fuel is the wrong shape for freeing a linear structure.**

3. **Acc-fold routing only detects simple-Var varying accumulators.** `sumFree` (varies
   `t`, a Var) routes to the accumulator lowering; a builder that varies
   `some(alloc(Node(k, acc)))` (a complex expression) does NOT, so it never gets the
   accumulator treatment at all.

## The right design — structural-unbox descent (the headline, soundness-sensitive)

The clean traversal is STRUCTURAL recursion on the LIST itself — no fuel:

```
sumFree : Opt (Own Node) -> Nat
fn sumFree(l) { match l { none => Zero, some(o) => match unbox(o) { Node(h, t) => add(h, sumFree(t)) } } }
```

Here the base case is `none` (trivially consumed) and the recursion is on `t`, the
unbox'd tail. For the totality checker to certify this, it must treat **`unbox(o)`'s
recursive field as a STRUCTURAL sub-term** of the matched `o` (one `Own`-indirection
deeper) — i.e. `unbox` "peels" an owned cell exactly like matching a constructor peels a
boxed value, so the peeled value's recursive fields descend.

**Soundness (the bar this must clear):** the descent terminates because an owned
structure built from `alloc` is FINITE and ACYCLIC — and acyclicity is GUARANTEED BY
LINEARITY (no aliasing ⇒ no back-pointers ⇒ no cycles). The extension is: "a recursive
call on a recursive field obtained by `unbox`-ing a linearly-owned sub-binder of the
scrutinee is a structural decrease." This must be red-teamed like 1a′/the positivity
relaxation — especially: can a NON-decreasing or aliased call sneak through (it must not);
does it stay rejecting for a non-linear / ω value (where acyclicity isn't guaranteed)?

## Proposed build order (bring each for the maximal bar)

1. The linear-accumulator multiplicity fix (#1 above) — small, lands with #2.
2. **Structural-unbox descent** in the totality checker — the headline. Certify
   `sumFree`-style traversal total; red-team the soundness (acyclicity argument, the
   non-linear / non-decreasing cases stay rejected).
3. Acc-fold routing for complex varying accumulators (#3) — only if a real program needs
   the builder shape (let the interpreter's friction decide).

Open question for the Leader: structural-unbox descent (recommended — it's the natural
shape + the acyclicity argument is exactly the linearity guarantee) vs. finishing 1b
well-founded recursion (`natWf`, deferred) vs. a dependent fuel/length index. I lean
structural-unbox descent: smallest, most ergonomic, and its termination proof IS the
linearity invariant we already enforce.
