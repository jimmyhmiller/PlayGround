# Multiplicity polymorphism — design

## Status of the problem

The language is *already fully expressive* for linear types with explicit
multiplicities — `examples/linear_generic.tal` maps `free` over a whole list of
linear `Own`s with a generic `lmap`. Multiplicity polymorphism is a **write-once
convenience**: it collapses the two combinator versions you otherwise write (a
`(1 x)`-callback map for linear-consuming callbacks, and a `(ω x)`-callback map for
value-storing callbacks) into one `lmap : (m : Mult) -> (ω f : (m x : a) -> b) -> …`.
It is NOT a missing capability.

## Design: monomorphization, kernel unchanged

The multiplicity variable lives **only in the surface**. It never reaches the QTT
kernel — at a call site it is substituted to a concrete `Mult` before elaboration,
so the kernel's rig stays `{0,1,ω}` and its soundness argument is untouched.

- **`(m : Mult)`** is an *explicit* multiplicity parameter (explicit-first, matching
  the project's stance — the caller writes `lmap(1, …)` / `lmap(w, …)`). `Mult` is a
  surface-only sort; the parameter is erased (it parametrizes multiplicities, not
  runtime values).
- A binder may use `m` as its multiplicity: `(m x : a) -> b`.
- **Per-call-site monomorphization**: at `lmap(1, …)`, `m := 1` is substituted
  through the signature and body, producing an ordinary monomorphic function that
  is elaborated and usage-checked concretely. Because each instance is concrete,
  soundness is exactly the existing checker's (e.g. `m := 0` dropping a linear
  element is caught by `0 ⋢ 1`, `m := ω` on linear data is caught by call-site
  scaling `ω ⋢ 1`). No symbolic-rig reasoning, no new trusted code.
- Instances are memoized per concrete `Mult`.

## Why not check once generically

Usage checking depends on multiplicities, so a mult-poly body cannot be checked
once at an abstract `m` and stay sound (an abstract element type hides whether
`m := 0` leaks). Monomorphization sidesteps this: every checked instance is
concrete. The finite rig makes this cheap (≤ 3 instances per mult parameter).

## Slices

1. **DONE — foundation.** Surface `SMult { Lit(Mult), Var(String) }` threaded
   through `Binder.mult` and `Ty::Arrow`; `parse_mult` + `binder_open` recognize a
   multiplicity-variable binder `(m x : a)`; `Elab.mult_env` + `SMult::resolve`
   resolve a variable to a concrete `Mult` (residual variable = clean hard error,
   never a panic). All existing code uses `Lit`, so this is behavior-neutral: 169
   tests pass, zero regressions. A `(m x : a)` signature parses and, until slice 2,
   reports `unresolved multiplicity variable` (test:
   `mult_variable_binder_parses_and_errors_cleanly`).

2. **DONE — monomorphization (surface pre-pass).** `monomorphize_mult_poly`
   (src/rust_surface.rs) runs on the parsed item list, BEFORE elaboration: a `fn`
   with an explicit `(m : Mult)` parameter is pulled out; every call site's mult
   argument (`0`/`1`/`w`, or an enclosing instance's own mult parameter) selects a
   concrete instance (`lmap$1`, `lmap$w`, …), synthesized once (signature with the
   `Mult` arrows stripped + `SMult::Var`s substituted; body with nested mult-poly
   calls rewritten under the substitution, so poly-calls-poly chains resolve) and
   spliced in at the original definition's position. Mult variables never reach
   elaboration, the kernel's rig stays `{0,1,ω}`, and each instance is checked
   concretely — including the LINEAR-CAPABILITY inference, which is what rejects
   `lmap(0, …)` over linear elements (the leak) at the instantiating call.
   `examples/mult_poly.tal`; tests `mult_poly_*`.

3. (Optional later) implicit `{m : Mult}` solved from the argument's
   argument-multiplicity, so the caller need not write it.

## Note

The language is *already fully expressive* without slice 2 (see
`examples/linear_generic.tal`); slice 2 is the write-once convenience that unifies
a linear-callback and an unrestricted-callback combinator into one definition.
