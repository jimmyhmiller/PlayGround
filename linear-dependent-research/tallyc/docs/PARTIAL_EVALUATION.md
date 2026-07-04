# Partial evaluation — specialising interpreters

tallyc's answer to Idris's [specialising interpreters](https://docs.idris-lang.org/en/latest/reference/partial-evaluation.html):
a recursive function applied to a **static constructor tree** (an interpreter
applied to a fixed program, a matcher applied to a fixed pattern, …) is
automatically specialised into straight-line residual code — the recursion, the
case dispatch, and the tree's `alloc`/`unbox` all vanish.

```
tally run examples/pe_interp.tal          → 33
tally dump examples/pe_interp.tal run     → \x. 1 + mul(2, x)    (the residual)
TALLY_PE_LOG=1 tally check <file>         → logs each `[pe] specialised <name>`
```

For `eval(prog(), x)` — a first-order interpreter over an owned `Expr` tree, at a
dynamic input `x` — the specialised `run` drops from **9 mallocs + tag dispatch**
to **0 mallocs, straight-line arithmetic**, byte-for-byte the hand-written C.

## The idea: PE is NbE with the dynamic input left symbolic

The kernel already normalises by evaluation (`eval` + `quote`) — it reduces
`Vec (2+2)` to `Vec 4` at the type level. Partial evaluation is the *same*
machinery pointed at runtime code: evaluate the function with the dynamic
arguments as **neutral variables**; everything static reduces; `quote`
residualises the result. The dynamic `x` stays symbolic and comes out as
`Var(0)` in the residual.

The kernel's `eval` deliberately withholds exactly two reductions (to keep
type-checking terminating and the memory layer opaque). The PE evaluator
(`peval` in `src/dep.rs`) adds them back:

1. **`Fix` unfolds.** A recursive function unfolds against its static argument
   (via a `VLamNative` self-closure). Termination is driven by the static tree
   shrinking, with a fuel backstop (2M steps) for any mis-fire.
2. **`unbox (alloc v) ⟶ v`.** The memory-layer β-rule: a statically-built owned
   tree is read through at compile time, so its `alloc`/`free` never reach the
   residual.

Both are meaning-preserving conversions (the fixpoint law and the intended
memory semantics), so the residual is genuinely equal to the original.

## Trigger and safety

The pass (`pe_reduce_body`) walks each elaborated def body and fires wherever a
`Fix` is applied to a **constructor/`alloc`-headed** static argument (a real
inductive structure — an AST, list, tree) plus at least one dynamic argument. A
bare `Nat` loop-counter is deliberately excluded: a counting recursion is not
bounded by it and would diverge (the fuel would catch it, but the type guard
avoids the wasted attempt).

The pass is **untrusted and self-verifying**, in the same discipline as every
other elaboration step:

- each rewritten body is **re-checked by the kernel** against the def's original
  type, and
- **reverted to the original on any failure**.

So PE can only make a well-typed program faster; it can never change what a
program means or reject a valid one. A bug in `peval` yields a residual that
fails the re-check and is dropped — never an unsound accepted program. This is
why the pass can run automatically on every program with no annotation.

## What it does not do yet

- **Higher order.** The interpreter must be first-order (tallyc has no surface
  lambdas). A HOAS interpreter (`Lam`/`App` returning closures) needs
  defunctionalization first.
- **Memoisation / knot-tying.** The current pass fully unfolds a *finite* static
  tree into one straight-line residual (no residual recursive calls). A static
  structure that drives recursion over a *dynamic* value (so the residual is
  itself recursive) would need memoised residual definitions — the classic
  online-PE knot-tying — which is future work.
- **Binding-time via `0`.** Today the trigger is "the argument is a
  statically-known constructor tree." The cleaner design — mark the argument `0`
  (erased ⇒ no runtime dispatch possible ⇒ *must* specialise) so PE is
  demanded by the type rather than opportunistic — is future work (it needs the
  PE pass to run before the erasure check for such functions).

## Key code

- `src/dep.rs`: `peval` / `pvapp` / `pvcase` / `fix_value` (the PE evaluator);
  `pe_specialize` (specialise a body to static args, leaving N dynamic);
  `pe_reduce_body` (the pass over a term); `is_static_data_root` / `is_static_tree`
  (the trigger).
- `src/rust_surface.rs` `check_program`: runs the pass with re-check-and-fallback.
- `examples/pe_interp.tal`; gates `pe_specialises_interpreter_to_straight_line_code`
  and `pe_end_to_end_interpreter_overhead_removed` in `src/dep_codegen.rs`.
