# CONVOY handoff — the last piece for the RUNNING dependent eval

**STATE AT HANDOFF: clean + green (127 tests +1 ignored). The convoy is NOT started (no
uncommitted work). Everything around it is built + committed.** The full thesis is already
PROVEN + committed (the demonstration cardinal + MVP-0). The convoy makes the dependent `eval`
EXECUTE end-to-end (currently it only type-checks the AST; lookup can't run yet).

## What's already done (committed, green)

- **MVP-0** — linear owned-AST interpreter, freed as it walks, runs to 7
  (`interpreter_mvp_arithmetic_runs_natively`, `examples/interpreter.tal`).
- **MVP-1 demonstration cardinal** — depth-indexed AST: out-of-scope vars rejected at
  type-check (`dependent_ast_makes_out_of_scope_variables_impossible_by_typing`).
- **for-all-T sentinel** — `Void` + `exfalso` (`void_and_exfalso_are_the_for_all_t_sentinel`);
  and `fzv : Fin Zero -> Void` via the existing all-absurd discharge type-checks (T=Void).
- **piece (2)** — self-call-with-implicits in a `Fix` body resolves (the (A) extension in
  `solve_fn_call`: a callee not in `defs` is resolved as the in-scope Fix self-binder);
  `dependent_partial_recursion_with_implicit_runs_natively` (`vlast` over `Vec Nat 3` = 3).
- **No J needed** — confirmed the kernel has `Eq`/`Refl` but no `Eq`-eliminator; the convoy is
  J-free (refinement via motive abstraction, not equality transport).

## The target (J-free, needs ONLY the convoy)

```
fn lookup(i, env) {
  match env {
    Nil           => exfalso(fzv(i)),                 -- env=Nil ⇒ n=Zero ⇒ i : Fin Zero ⇒ Void
    Cons(v, rest) => match i { FZ => v, FS(j) => lookup(j, rest) },
  }
}
```
needs: in the `Nil` arm, `i : Fin n` must be seen as `i : Fin Zero` (so `fzv(i)` checks); in
the `Cons` arm, `i : Fin (Succ k)`. The recursion (`lookup(j,rest)`, `%partial` Fix) already
works via piece (2). So lookup runs once the convoy refines `i`'s type per arm.

## The CONVOY — the mechanism (integration point: `elab_nested_match`, rust_surface.rs ~1323)

Today `elab_nested_match` builds a CONSTANT motive (`λindices. λscrut. expected`, ~line 1330)
and methods `λfields. arm_body`. A surface-only substitution (`i : Fin Zero` in the Nil arm
body) FAILS the kernel re-check (the kernel still sees `i : Fin n`). The sound fix is to
ABSTRACT the index-dependent context vars INTO the motive (the convoy / "view from the left"):

1. **Detect deps.** The scrutinee is `e_term : Data(dname, dargs)`; the INDEX values are
   `dargs[np..np+ni]`. For each index value that is a context VAR (`VNeu(NVar(level))`), that
   level is a "refinable index". DEPS = context vars (in `cx`) whose TYPE mentions a refinable
   index level. For lookup's `match env`: index = `n` (a var); dep = `i : Fin n`.
   (If an index is not a bare var — a complex expr — do NOT refine on it: conservative, sound.)
2. **Motive abstracts the deps.** Motive body becomes
   `Pi(dep_ty_1', … Pi(dep_ty_k', expected') …)` where each `dep_ty_j'` is `dep_ty_j` with the
   scrutinee's index VAR replaced by the motive's corresponding index BINDER (and shifted by
   the `ni+1` motive binders). For lookup (ni=1): motive = `λn'. λenv'. (Fin n') -> Nat`
   (the dep `Fin n` → `Fin (Var 1)`, the index binder `n'`).
3. **Methods bind + refine the deps.** Each method = `λfields. λdeps. arm_body`, with the deps
   bound in `arm_cx` at their REFINED types: substitute (scrutinee-index-var := the ctor's
   result index, in `arm_cx`) into `dep_ty`. For `Nil` (index `Zero`): `i : Fin Zero` →
   `exfalso(fzv i)` checks. For `Cons` (index `Succ k`): `i : Fin (Succ k)` → `match i` checks.
   The arm body is elaborated with the deps re-bound (shadowing the outer deps).
4. **Result applies to the actual deps.** `App*(Elim/Case(dname, motive, methods, e_term),
   actual_dep_1, …)` — apply the eliminator's result (a function of the deps) to the original
   dep values (`Var(dep_db)`).

The de-Bruijn is the intricate part (the index-var → motive-binder substitution in step 2; the
per-arm `n := ctor_index` substitution in step 3; the application in step 4). Use `map_vars` /
`dep::subst` / `dep::shift_term`. Works for both the `Elim` (total) and `Case` (`in_fix`, the
heap-recursion %partial) paths — lookup is %partial so it's `Case` + `Fix`.

## SOUNDNESS CARDINAL (maximal bar)

- The convoy abstracts ONLY genuinely-index-dependent vars, and each ctor genuinely determines
  its index (Nil → Zero, Cons → Succ k). **The kernel re-check is the un-fakeable backstop:**
  a wrong convoy (over-/under-abstraction, or a wrong refinement) produces an ill-typed term
  the kernel rejects — you can only build a convoy the kernel accepts.
- RED-TEAM: (a) a non-index-dependent context var is NOT abstracted (no spurious convoy);
  (b) a should-reach case stays demanded (the discharge still rests only on the conservative
  all-absurd `fzv : Fin Zero` + `exfalso` — genuinely empty); (c) refine only on a genuine
  index determination (a bare index var), never a false unification.

## TEST TARGETS (bring for the bar like the kernel terms)

- `lookup : {0 n} -> Fin n -> Vec Nat n -> Nat` type-checks + runs (a real index → the right
  element). Then the dependent `eval : Vec Nat d -> Own (Expr d) -> Nat` over the depth-indexed
  AST runs end-to-end (the complete RUNNING dependent-half demo).
- Regression: every existing test stays green (the convoy only fires when index-dependent deps
  exist; constant-motive matches are unchanged).
