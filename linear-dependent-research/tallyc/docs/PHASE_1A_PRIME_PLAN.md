# Phase 1a′ — accumulator-style `%total` (fold-into-function), implementation plan

**Status:** designed + confirmed ELABORATION-ONLY (no kernel change). Feasibility
proven by the kernel test `accumulator_fold_into_function_lowering_is_valid_today`
(commit 7d4944c6c): an accumulator fold lowers to a `NatElim` with a
**function-typed motive** and type-checks + computes on today's kernel.

## Goal
Make accumulator-style structural recursion — a recursive call that descends on
the scrutinee (a strict subterm) but **varies other arguments** — certifiable
`%total`. This unblocks `div`/`gcd`/`lt`/`sub` (all decrement multiple args).
Today E1 declines these as `Partial("accumulator-style")`.

## The lowering (the standard fold-into-function encoding)
For `f(scrut, a₁…a_K)` matching on `scrut`, recursing `f(smaller, e₁…e_K)`:
- **motive** = `λ scrut. λ a₁…λ a_K. Result` — a function of the accumulators
  (so the IH is itself a function `acc_tys → Result`).
- **Zero method** = `λ a₁…a_K. <Zero-arm body>`.
- **Succ method** = `λ k. λ ih. λ a₁…a_K. <Succ-arm body>` where each recursive
  call `f(k, e₁…e_K)` becomes `ih e₁ … e_K` (apply the IH to the NEW accumulators).
- **final** = `λ scrut a₁…a_K. (NatElim motive z s scrut) a₁ … a_K`.

Scope v1: NON-dependent `Result` (independent of the varying args — true for
`div`/`gcd`/`lt`/`sub`) and a `%builtin Nat` scrutinee. Dependent-accumulator and
boxed-datatype accumulator folds are follow-ups.

## The three edits
1. **`totality.rs` `structural_verdict`** — drop the "other args verbatim"
   rejection: a recursive call that strictly descends on the scrutinee is
   `Total` **regardless** of other args, GATED on lowerability (Nat scrutinee, or
   verbatim-other-args for boxed). The SCRUTINEE-descent requirement STAYS — a
   call that does not descend on the scrutinee is still `Partial` (the dual-failure
   guard: `loop(Succ k)` must remain rejected). Needs the scrutinee's "is it a
   builtin Nat" bit threaded into `FnClauses`.
2. **pass D routing** (`rust_surface.rs`) — an accumulator Nat-fold routes to the
   new `elab_nat_match_acc` (function-typed-motive `NatElim`), not `elab_fix_nat`
   (Fix/partial). Verbatim-arg folds stay on the existing simple path (no regression).
3. **`elab_nat_match_acc`** — the new lowering above. Intricate de-Bruijn: the
   motive abstracts `scrut + K` binders (shift `Result` by `1+K`); each method
   binds the accumulators after the field/IH; `ih_for` is generalized so a
   recursive call maps to `App(ih, e₁′, …, e_K′)` with the new accumulator args
   elaborated in the arm context.

## Red-team (E1-class dual-failure bar)
- genuinely-terminating accumulator folds (`sumacc`, `fuel-div`, `lt`, `sub`)
  → `%total` ✓ and run natively;
- a fold that does NOT structurally descend on the scrutinee (or varies it
  non-well-foundedly) → still REJECTED, not mis-certified via the accumulator path;
- verbatim-arg folds unchanged (no regression);
- the kernel re-checks the function-typed-motive `NatElim` (a lowering bug ⇒
  rejected program / re-check failure, never unsound).

## Proof target
`%total fuel-div` (+ `lt`/`sub` becoming total) — written in 1a surface syntax
(`let`, nested/expression `match`), running natively.
