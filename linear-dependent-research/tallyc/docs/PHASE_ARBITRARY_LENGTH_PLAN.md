# Phase #1 — arbitrary-length recursion/iteration over LINEAR structures

**STATUS: (A) DONE + RUNS — at the maximal-bar review.** Arbitrary-length recursion over a
LINEAR OWNED structure now builds/traverses/FREES + runs natively (%partial), via the new
kernel `Term::Case` (non-recursive general case-split) + `compile_case` + `elab_fix`/`in_fix`
+ context-var-call inference. Tests: `owned_list_traversal_runs_natively` (=3),
`heap_general_recursion_runs_natively` (=3), `positivity_sees_through_case_hiding_a_negative_occurrence`
(E3), `partial_heap_recursion_still_enforces_linearity` (leak/double-free REJECTED). The
three cardinals hold: linearity enforced (NOT relaxed), kernel-opaque `Fix`, E3 positivity
traverses `Case`. Discipline: the recursive result must be `let`-sequenced (CBV-`let`) to
sequence a linear consumption into an `ω` position. **(B)** structural-unbox-descent (→ the
`%total` subset) is the friction-driven follow-up, when a real `%total` need appears.

---

**Original plan (kept for record): verify-before-building turned up that this is BIGGER than
"thread the accumulator's multiplicity." Three sub-issues; the headline one became the kernel
`Term::Case`. (A) Fix-on-heap was built first per the Leader's steer.**

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

## UPDATE — the real blocker is a CODEGEN gap (Fix is Nat-only), under BOTH totality verdicts

Probing the structural (no-fuel) `sumFree` directly surfaced the actual wall:

```
[plain] `fn sumFree` is partial (… `t` … not a sub-structure of `l`) BUT CANNOT BE LOWERED:
        general/mutual recursion is only supported on a `%builtin Nat` scrutinee so far.
```

So a function that recurses on a HEAP structure (`Opt (Own Node)`, `Own Expr`) cannot be
lowered AT ALL today — neither as `%total` (no eliminator path for heap-tail recursion) nor
as `%partial` (`Fix`/general recursion is `%builtin Nat`-scrutinee-only). The native backend
has no "recurse on a boxed/heap value" path beyond the structural boxed ELIMINATOR (a fold
with verbatim args), which this is not.

This means the headline #1 unblock is a **CODEGEN extension — native recursion on a heap
structure** — and it's needed under BOTH verdicts:
- **(A) `Fix` on a heap scrutinee** → a heap-recursive fn RUNS as `%partial` general recursion
  (the interpreter's `eval` need not be total). The DIRECT unblock for "write real programs."
- **(B) structural-unbox descent** → certifies `%total` (the opt-in total subset), then lowers
  via the same heap-recursion codegen. The refinement on top of (A).

`match unbox(o)` desugars to `let v = unbox(o); match v { … }`, and the recursive fields are
`Own T` / `Opt (Own T)` (Own-wrapped — `rec_field_arity` doesn't see them), so BOTH the
codegen (recurse through the unbox'd cell) and the totality analysis (sub-term provenance
through `let v = unbox(o)` + the match) must look through the unbox.

SOUNDNESS for (A): a `%partial Fix` makes NO termination claim and the kernel treats it
opaquely — so it carries no soundness risk beyond the linearity the kernel already checks
(each `Own` consumed once). (B) is where the maximal-bar termination argument lives
(acyclicity-from-linearity), layered after (A) RUNS.

RECOMMENDATION: build (A) heap-recursion codegen FIRST (the interpreter runs as `%partial`),
then layer (B) unbox-descent totality (→ `%total`). (A) is the actual "arbitrary-length"
unblock; (B) is the total-subset refinement.

## (A) IMPLEMENTATION SCOPE — the crux is a new kernel `Term::Case` (boxed case-split)

Scoping (A) to the bottom: a `%partial` heap-recursive fn lowers to `Fix(ty, λparams.
<body>)` where recursion is via the `Fix` self-binder (self-calls) and the body's matches
are CASE-SPLITS. The body's boxed match CANNOT reuse `Term::Elim`: `Elim` always computes
the induction hypotheses (it recurses the structure), so with the self-calls ALSO recursing
the result is **EXPONENTIAL** (every node's subtree is elim'd once per enclosing self-call).
Verified by reasoning through `eval(Add(a,b)) = elim(unbox(e))` — the discarded IHs re-walk
each subtree. So a `Fix` body needs a NON-recursive boxed case-split, which the kernel does
not have (`NatCase` is exactly this, but only for `%builtin Nat`).

**The crux: add `Term::Case(D, motive, methods, scrut)`** — the general-datatype analog of
`NatCase`: switch on the constructor, bind its fields into the matched method, NO IH, NO
recursion. Touch points (the full kernel-term surface — same as the CBV-let change):
- `dep.rs`: the `Term::Case` variant; `eval`/`vcase` (reduce on a constructor, substitute
  fields — like `velim` but drop the IH args); `quote`; `map_vars`/shift (Case binds fields
  in each method — cross the right binder counts); `Neutral::NCase` (stuck on a neutral);
  `infer` (type-check like `Elim` but methods have NO IH binders).
- **POSITIVITY (E3-CRITICAL — the CBV-let lesson):** `occurs` and `strictly_positive` MUST
  traverse `Term::Case`'s subterms (motive, methods, scrut). A subterm-bearing variant the
  positivity checker doesn't descend = a reopened Curry hole. The `occurs` match is
  EXHAUSTIVE (no `_ =>`), so adding the variant forces handling it — verify + regression-test
  a negative occurrence hidden in a `Case`, exactly like the `Let` test.
- `dep_codegen.rs`: `compile_case` (switch on the tag, bind fields, run the method — the
  per-constructor branch of the eliminator WITHOUT the recursive helper call); the
  `Fix`-on-boxed lowering uses it.
- `rust_surface.rs`: a Partial fn whose body recurses on a BOXED/heap scrutinee (directly,
  or via `let v = unbox(o); match v`) lowers to `Fix` with `Case` bodies (generalize
  `elab_fix_nat` → `elab_fix` using `Case` not `NatCase`; route boxed scrutinees here
  instead of the hard error).

THE TWO CARDINALS for the review (Leader): (1) `%partial` relaxes TERMINATION not LINEARITY —
the `Fix` body is still fully linearity-checked by the kernel (each `Own` once on every path;
a leaking/double-freeing `%partial` is REJECTED); (2) the kernel treats `Fix` OPAQUELY (never
unfolds it in type-level reduction) — `Case` only reduces on a concrete constructor at
runtime, and a `Fix` never reduces in the checker. Plus the E3 positivity-traversal of `Case`.

Open question for the Leader: structural-unbox descent (recommended — it's the natural
shape + the acyclicity argument is exactly the linearity guarantee) vs. finishing 1b
well-founded recursion (`natWf`, deferred) vs. a dependent fuel/length index. I lean
structural-unbox descent: smallest, most ergonomic, and its termination proof IS the
linearity invariant we already enforce.
