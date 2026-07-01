# The CONVOY — index refinement in dependent match (DONE)

**STATUS: BUILT, GREEN, RUNNING NATIVELY.** This document was the handoff plan; it now
records what was actually built (which is MORE than the plan — the plan's mechanism
alone could not type `lookup`'s inner match).

## What it is

`elab_nested_match` (src/rust_surface.rs) now builds a DEPENDENT motive whenever the
scrutinee's type is indexed by (`Succ` of) a context variable and that matters:

1. **Detection** (`detect_convoy`): single-index families over builtin `Nat`, index
   value `= x` (`r = 0`) or `= Succ x` (`r = 1`) for a context var `x`. The DEPS are
   the context vars whose types mention `x` (transitively). Fires when deps exist, or
   (`r = 1`) when an arm is refuted or the expected type mentions `x`.
2. **Motive**: `λ idx. λ scrut. Π deps[x := idx]. expected[x := idx]` (r = 0), or
   `λ idx. λ scrut. NatCase[λ_.Type] idx Nat (λ p. Π deps[x := p]. expected[x := p])`
   (r = 1) — the **`NatCase` large elimination computes the predecessor back from the
   index**, the J-free move (same technique as the absurd discharge) that the original
   plan was missing: it is what re-types `rest : Vec Nat k` through a match on
   `i : Fin (Succ k)`, and what types `vhead`/`vtail` index projections.
3. **Methods**: each arm re-binds the deps (same names — they shadow) at the REFINED
   types read off by evaluating the kernel's own method type (`elim_method_telescope`
   + NbE reduces `motive ctor-idx (ctor …)` to the refined `Π`-chain). Arms refuted by
   the index (`Zero`-ctor vs `Succ`-headed index — decidably disjoint, so no
   misclassification is possible) MUST be omitted; their methods are the `Nat`
   sentinel of the motive's `Zero` branch (typed dead code).
4. **Result**: `App(Case/Elim(D, motive, methods, scrut), dep₁ … dep_k)`.
5. **Codegen** (`commute_apply_into_methods`, src/dep_codegen.rs): the case-commuting
   conversion pushes the applied deps into the arms post-check, so the backend emits
   its ordinary tag-switch. Erasure intact: motive/indices/`Fin` bounds leave no trace.

Also wired at the FN level: a total, non-self-recursive `fn` whose body is a match
takes this path when the convoy fires (recursive convoy bodies go through `%partial`
`Fix`, where the self-call needs no IH).

**Soundness**: the elaborator is untrusted; the kernel re-checks the motive, every
method, and the final application. Every red-team case is a test:
`convoy_refinement_is_real_not_a_loophole` (a blanket coercion is rejected),
`convoy_impossible_arm_must_be_omitted_and_reachable_arm_must_be_present`.

## What it delivers (tests + examples)

- `lookup : {0 n} -> Fin n -> Vec Nat n -> Nat` — total-coverage, bounds-check-free
  vector lookup, RUNS natively (`examples/lookup.tal`, `tally run` → 2;
  `convoy_dependent_lookup_runs_natively`). The `Nil` arm is discharged as absurd
  (`fzv` + `exfalso`); the `Cons` arm's inner match re-binds `rest` through the
  Succ-inversion so the recursive call's implicit is inferred consistently.
- `vhead`/`vtail` over `Vec a (Succ k)` with the impossible `Nil` arm omitted —
  real dependent COVERAGE (`convoy_vec_head_tail_run_natively`).

## v1 limits (honest, conservative — misses fall back to the constant motive)

- Single-index families, index domain builtin `Nat`, `Succ`-depth ≤ 1 (deeper needs
  nested `NatCase`s whose inner case sticks on a neutral).
- No deps typed by the scrutinee itself.
- Elim-path (total) convoy fires only for non-self-recursive bodies; a recursive
  convoy body is `%partial` (dep-applied IHs not built).
- Location/boxed-indexed families: needs the stratum-(A) location equality (planned).
