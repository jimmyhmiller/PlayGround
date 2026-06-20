# Phase 1a′ — accumulator-style `%total` (fold-into-function), implementation plan

**Status: ✅ IMPLEMENTED.** All three edits landed and proven (81 lib tests green).
The verdict change (`scrut_is_nat` gates the verbatim-args rejection), the routing
(`is_acc_fold` → `elab_nat_match_acc`), and the function-typed-motive lowering are
all in. Proof target met: `%total fuel-div` (composing accumulator folds `lt`,
`sub`, and the fuel-driven `div`) is certified total and runs natively
(`div(10,7,2)=3`), written in 1a surface syntax (nested/expression `match`). The
dual-failure red-team passes: a non-descending fold is still rejected (scrutinee
descent stays unconditional), a boxed-datatype accumulator is still declined, and
verbatim folds are unchanged. Tests: `phase_1a_prime_*` in
`src/rust_surface/tests.rs`. The plan below is the original spec, kept for record.

---

**Original status (pre-implementation):** designed + confirmed ELABORATION-ONLY (no
kernel change). Feasibility proven by the kernel test
`accumulator_fold_into_function_lowering_is_valid_today`
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

## Refinements worked out while building the verdict half (read these first)
- **The verdict change is the EASY half** (drafted + validated): add
  `scrut_is_nat: bool` to `FnClauses` (set in `fn_clauses` from
  `nat_types.contains(scrut_datatype)`); in `structural_verdict`, wrap the
  "other args verbatim" rejection in `if !f.scrut_is_nat { … }`. The scrutinee
  strict-subterm descent check STAYS unconditional (dual-failure guard).
- **CRITICAL CORRECTION on the motive shape** (I almost got this wrong): a
  `NatElim` motive has type `Nat → Type`, so the motive is
  `λ scrut'. PiChain(acc_tys → R)` — a function from `Nat` to a **Pi TYPE**
  `T₁ → … → T_K → R`, i.e. `Term::Lam(Pi(ω,T₁', Pi(ω,T₂', …, R')))`. It is NOT
  `K+1` nested lambdas. (Nested lambdas are the METHODS, which return VALUES.)
  For `fuel-div`: `motive = Lam(Pi(ω,Nat, Pi(ω,Nat, Nat)))` = `λ_. (Nat→Nat→Nat)`.
- **Methods** (these ARE nested value-lambdas): `z = λ a₁…a_K. <Zero body>`;
  `s = λ k. λ ih. λ a₁…a_K. <Succ body>`, with `ih : T₁→…→T_K→R`.
- **final fn body** = `(NatElim motive z s scrut_var) acc₁_var … acc_K_var` (apply
  the built function to the actual accumulators), then pass-D λ-wraps the params.
- **`ih_for` needs an ACCUMULATOR MODE**: a recursive call `f(k, e₁…e_K)` becomes
  `App(…App(ih, e₁′)…, e_K′)` where the `eᵢ′` are the NON-scrutinee args
  elaborated in the Succ-arm context (skip `scrut_pos`, in param order). Add an
  `acc: Option<&[String]>` (accumulator param names) to `Rec`, or a parallel
  path. The accumulator ORDER = non-scrutinee params in param order.
- **v1 RESTRICTIONS** (error clearly otherwise): NO implicit params (all
  explicit — `fuel-div`/`lt`/`sub` qualify); NON-dependent return type `R`
  (independent of scrut/accs). Note `full_params` includes implicits but
  `explicit_pos` is explicit-space — reconcile carefully (or require all-explicit).
- **MUST LAND TOGETHER**: the verdict change alone (without the lowering) routes
  accumulator Nat-folds through `elab_nat_match`, whose `ih_for` maps the call to
  the plain IH and IGNORES the varying args → a well-typed-but-WRONG term (for a
  non-dependent `R`, types match so the kernel does not catch it; the native-run
  test does). It also flips the existing `total_certificate_rejects_accumulator…`
  test (`addacc` would become accepted). So implement verdict + lowering + the
  `addacc` test update + the routing in one diff.
- **Routing**: simplest is for `elab_nat_match` to self-detect accumulator (some
  recursive call varies a non-scrutinee arg) and dispatch to `elab_nat_match_acc`;
  verbatim folds stay on the current simple path (no regression).

## Suggested build+test order
1. `elab_nat_match_acc` for ONE accumulator; prove with `sumacc` (native run = 3).
2. Generalize to K accumulators; prove with `fuel-div` (+ `lt`,`sub`) native run.
3. Verdict change + routing + `addacc`-test update; full suite green.
4. Dual-failure red-team (non-descending fold still rejected; verbatim unchanged).
