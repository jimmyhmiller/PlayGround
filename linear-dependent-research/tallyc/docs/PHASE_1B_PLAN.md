# Phase 1b — well-founded recursion (`Acc`) surface exposure, plan + status

**STATUS: the wf-recursion MACHINERY is DONE + tested; the `natWf` LIBRARY LEMMA is a
documented finish-later (Leader decision: pivot to Phase A — the memory model is the
goal — and revisit natWf after).** Done + tested: higher-order/`Acc` recursive-field
elimination, the value-correctness guard (1a′-bar, closes the dropped-arg wrong-value
hole), the totality relax (varying args only under higher-order descent), gap-(i)
implicit inference, partial application confirmed, the `Acc` family declared + an
`Acc`-eliminating fn certified total. So the CAPABILITY exists. FINISH-LATER (a
substantial type-theory investment, deprioritized vs the memory-model goal): the
`natWf : (n:Nat) → Acc Nat Lt n` accessibility lemma so user code gets `Acc` proofs for
free — blocked on (1) MULTI-INDEX absurd discharge + the for-all-T sentinel [a
soundness-sensitive coverage-checker extension — see Findings], and (2) a
bound/transitivity lemma + assembly [proof engineering]. NO current soundness hole is
left open (single-index Fin discharge stays sound; the for-all-T sentinel is a FORWARD
prerequisite, not a current bug).

**Goal:** let the surface write functions whose termination is justified by an
`Acc`-style accessibility proof on a well-founded relation (gcd, quicksort, log/div
by a measure) and certify them `%total`. The kernel side landed in E3 (the generic
eliminator already computes well-founded recursion — see below); 1b is the
ELABORATION-side surface exposure, kernel-backstopped.

## What the kernel already gives us (E3, no kernel work needed)

`Acc (A:Type) (R:A→A→Type) : A → Type` is an ordinary INDEXED inductive family with
ONE constructor whose recursive field is HIGHER-ORDER:

```
acc : (x:A) → ((y:A) → R y x → Acc A R y) → Acc A R x
```

The generic eliminator (`velim` + `build_ih` in `dep.rs`) computes its recursor: the
IH for the higher-order field `h : (y:A)→R y x→Acc A R y` is the FUNCTION
`λ y r. elim (h y r)` (`build_ih` at telescope arity m=2) — i.e. the genuine
sub-derivation on any `y` with `R y x`. Proven by the kernel tests
`acc_accessibility_family_is_well_formed`, `higher_order_*`, and
`indexed_higher_order_ih_uses_the_recursive_occurrences_own_index`. So well-founded
recursion is a kernel-checked eliminator, exactly like a structural fold.

## ✅ DONE — the foundation: higher-order recursive fields at the surface

A recursive constructor field may now be DIRECT (`data idxs`) or HIGHER-ORDER
(`(z…) → data idxs`). Landed (commits 643e4b268 / 17909173b):

- `dep::rec_field_arity` — detects a recursive field (direct or higher-order) and its
  telescope arity, mirroring the kernel's `rec_spine`.
- `smaller_binders` returns `(direct, higher_order)` recursive-field binders;
  `ArmInfo.ho_smaller` carries the higher-order ones.
- `structural_verdict` (totality) treats a recursive call whose matched-position
  argument is `f(args…)` for a higher-order field `f` as a structural DECREASE (the
  scrutinee-descent requirement stays unconditional — the dual-failure guard holds).
- `elab_match_body` detects higher-order recursive fields (via `rec_field_arity`) and
  generates the functional IH (the binder types already come from
  `elim_method_telescope`/`method_ty_tm`).
- `ih_for` higher-order branch: a recursive call `g(f(args…))` lowers to
  `App(…App(ih, args₀)…, argsₙ)`.

Proof: `phase_1b_wtype_higher_order_fold` — a W-type `Tree { node2 : (Bool→Tree)→Tree }`
with a `%total` fold (`size`), certified total + computes (kernel evaluator).
Construction passes a NAMED helper (the surface has no lambdas; a top-level fn is a
value). NOTE: this is the W-type case; `Acc` is the INDEXED case below.

## ⬜ REMAINING — `Acc`-based well-founded recursion (the real 1b deliverable)

The shape of a wf-recursive function is two-argument:

```
f : (x:A) → Acc A R x → B
f x (acc x h) = … f y (h y r) …        -- recurse on y, justified by r : R y x
```

The matched scrutinee is the `Acc` proof (param 2); the recursive call's
matched-position argument is `h y r` (the accessibility fn applied), and the IH from
the eliminator is `ih : (y:A) → (r:R y x) → B`, so `f y (h y r)` ≡ `ih y r`. The
`ih_for` higher-order branch already maps `…(h y r)` → `App(App(ih, y), r)`. The
remaining problems:

1. **The explicit descent argument (value-correctness — the 1a′-class subtlety).**
   The recursive call is `f(y, h(y,r))`: position 0 passes `y` (the new `x`), which
   `ih_for` currently IGNORES (it uses only the `h`-application's args). For a
   non-dependent `B` a mismatched explicit arg (`f(z, h(y,r))`, `z≠y`) would lower to
   `ih y r` SILENTLY DROPPING `z` and the kernel would not catch it (types match).
   FIX: when the descent is higher-order, REQUIRE each non-scrutinee argument to be
   syntactically the corresponding `h`-application argument (here `y`), else reject.
   This is the exact load-bearing guard the 1a′ reviewer demanded (correct-pin
   accepted, wrong-pin rejected) — must be red-teamed the same way.

2. **Totality: relax verbatim-other-args ONLY under higher-order descent.** Today the
   boxed-datatype path requires non-scrutinee args verbatim. For wf-recursion the new
   `y` legitimately VARIES. Allow it IFF (1) holds (the varying arg equals the
   descent's index arg). Keep it rejecting otherwise (dual-failure guard).

3. **Native codegen for higher-order eliminators.** `build_elim_helper` handles
   FIRST-ORDER recursive fields (IH = a `self` call on the field). A higher-order
   field's IH is a FUNCTION `λz…. self(field z…)`; the backend must compile
   `ih(args…)` to `self(field-applied-…)`. Until then, wf-recursion runs in the
   KERNEL EVALUATOR only (honest: like the 1a′ folds before the backend extension).
   Mirror the 1a′ `acc_ih_selves` approach (identify by env LEVEL so it threads
   through helpers).

4. **Supporting machinery for a runnable end-to-end demo.** Building an `Acc` proof
   needs a relation `R`, accessibility helper fns, and discharge of the empty
   `R y x` cases (absurd — E2). A `natWf : (n:Nat) → Acc Nat lt n` accessibility
   lemma (kernel/prelude-level) is the ergonomic entry point so user code never
   builds proofs by hand. Decide: prelude-provide `Acc` + `natWf`, or require users
   to declare them.

## Findings (verified this session)

- **The 1b foundation generalizes to the indexed higher-order `Acc` family** (not just
  non-indexed W-types): `AccN : Nat -> Type` with the accessibility field declares
  (strict positivity holds) and an `Acc`-eliminating function type-checks + is
  certified total (`phase_1b_acc_indexed_higher_order_family_eliminates`).
- **Inductive `Lt` machinery works with EXPLICIT indices**: `ltS(0,1,ltZ(0)) : Lt 1 2`
  (`phase_1b_inductive_lt_constructs_with_explicit_indices`).
- **GAP (named, prioritizable): implicit-from-result-index inference is missing for
  inductive-relation constructors.** `ltZ : {0 n} -> Lt Zero (Succ n)` against expected
  `Lt Zero (Succ Zero)` FAILS ("cannot infer implicit argument") even though `n=Zero`
  is forced by the expected indices. natWf builds many `Lt` proofs; without this it is
  workable but verbose (explicit indices everywhere). Fixing it = extend `solve_ctor`
  to unify a constructor's implicit args against the EXPECTED type's index spine.
- **gap (i) FIXED** (commit lands implicit-from-result-index inference): `solve`
  reconciles the packed `VNatLit` literal with the `Succ`-spine, so relation-ctor
  implicits solve from the expected indices (`ltZ : Lt 0 1`, `ltS(ltZ) : Lt 1 2` need
  no explicit indices). Exact, decidable, kernel-backstopped.
- **partial application WORKS** (verified): `add(Succ Zero) : Nat -> Nat` as a value and
  as a higher-order argument — so the accessibility function `(y) -> Lt y x -> Acc y`
  can be provided as `accStep(x)` (partial app of a named helper). The no-lambda
  concern is resolved for natWf.
- **GAP (next blocker, BIGGER than a quick win): absurd discharge is SINGLE-INDEX
  only.** `match (pf : Lt y Zero) { }` fails because `try_absurd_match` returns `None`
  for any family with `indices.len() != 1` (rust_surface.rs:1533) — `Lt` is two-index.
  natWf's base case (`Lt y Zero` empty) needs multi-index absurd discharge. Extending
  it is SOUNDNESS-SENSITIVE: the code's own BACKSTOP CAVEAT warns the current Nat
  sentinel makes the kernel-rejection backstop T-dependent (silently accepts a
  mis-classified-reachable case when the result type is `Nat`), so a for-all-T sentinel
  (a fresh uninhabited `Void` + `elim Void`) must land BEFORE extending to multi-index /
  per-constructor discharge. So this is a real coverage-checker investment, not a
  one-liner.
- **GAP: natWf-by-strong-induction needs a bound/transitivity lemma** (`Lt z y -> Lt y
  (Succ k) -> Lt z k`) plus assembly — multi-lemma dependent-type proof engineering on
  top of the above.
- **GAP: the no-lambda surface blocks the inline accessibility function.** natWf's
  `acc(n, λ y prf. …)` needs a function `(y:Nat) -> Lt y n -> Acc y` that recurses;
  with no surface lambdas it must be a NAMED top-level helper that threads the
  recursion explicitly (Jimmy's blessed direction: named fns, no implicit closures) —
  workable but it shapes how natWf is written (a mutually-recursive helper, or a
  worker taking the fuel/measure as a param). This is the linchpin chunk.

## Red-team (E1/1a′-class bar)

- a wf-recursive fn whose recursive call does NOT descend through the accessibility
  proof → still REJECTED (scrutinee-descent unconditional);
- the value-correctness pin: a correct `f(y, h(y,r))` ACCEPTED + right value; a wrong
  `f(z, h(y,r))` REJECTED (problem 1);
- non-strictly-positive `Acc`-like family REJECTED (kernel positivity — already
  enforced, see `acc_accessibility_family_is_well_formed`'s negative case);
- once native codegen lands: the same fn RUNS natively (not only in the evaluator).

## Suggested order

1. value-correctness guard + totality relax (problems 1–2), proven on an `Acc`
   wf-recursive fn in surface syntax (kernel evaluator) + red-team;
2. native codegen for higher-order eliminators (problem 3) + native-run test;
3. the `natWf` ergonomic lemma (problem 4) so wf-recursion needs no hand-built proofs.
