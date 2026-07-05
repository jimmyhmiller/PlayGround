# microlang

A trait sketch that re-cuts the dynamic-language toolkit at the two axes the
original conflated, with three micro-languages that exercise it. Backed by a
tree-walking executor so the *traits* are the point, not the codegen.

## The two axes as traits

| Axis | Trait(s) | File | Carved at |
|---|---|---|---|
| Value model | `Repr`, `ValueModel` | `model.rs` | **immediacy** (`is_immediate(Cat)`), not float-vs-tagged |
| Execution | `CodeSpace` (+ `TreeWalk`, `ClosureComp`, `Traced`) | `code.rs`, `compiled.rs` | meaning (`Ir`) vs strategy; re-entrant `invoke` |

Two real execution tiers, both behind the same `CodeSpace` contract and `Ir`:
`TreeWalk` (interpreter, re-dispatches the `Ir` each run) and `ClosureComp`
(`compiled.rs`, compiles each `Ir` subtree to a Rust closure once, caches
function bodies). `tiers_agree` pins that they produce identical results on every
program. `ClosureComp` also proves the contract's two hard promises:
compile-once (`compiled_bodies() == 1` for a 6-deep recursion) and late binding
(a compiled fn calls one defined later — mutual recursion with a forward
reference).

`value.rs` holds the neutral vocabulary (`Cat`, `Val`, `Obj`) with `Int` a peer
of `Float`. `runtime.rs` ties it together: reader (code is data), `encode`/
`decode` (the boxing seam, `allocs`-counted), `macroexpand` (re-enters compiled
code), `analyze` (`Val` -> `Ir`), and the value-model-aware primitives.

## Run it

```
cargo run --example calc        # value axis: same engine, two models, measured allocs
cargo run --example microlisp   # eval axis: defmacro, recursion, macro-time re-entrancy
cargo run --example backends    # execution axis: interpreter vs compiler, same Ir
cargo test                      # locks all three
```

`calc` prints integers free on `LowBit` / boxed on `NanBox`, and the exact
inverse for floats: the whole "value layout is a free choice" claim, made real.
`microlisp`'s `add2` macro calls the compiled `inc` during its own expansion.

## Two real models, ~30 lines apart

`LowBitModel` and `NanBoxModel` differ in one method (`is_immediate`) plus the
bit twiddling. Both drive the same generic `arith`. That is the discipline that
keeps the axis honest: a second real, tested consumer, not a fork.

## Design-tension #1: resolved (backends compose)

Originally `Runtime<M, C: CodeSpace<M>>` baked the backend into the runtime's
type. That swapped cleanly but did not *compose*: a wrapping backend
(`Traced<Inner>`) failed to typecheck, because `Inner::eval_ir` wanted
`Runtime<M, Inner>` while handed `Runtime<M, Traced<Inner>>`. Two changes fixed
it, and both were load-bearing:

1. **Backend is a value, not a type parameter.** `Runtime<M>` carries no
   backend; `eval_ir`/`invoke` take `&self` and you pass the code space in.
   `Traced` now holds a `Box<dyn CodeSpace<M>>` and delegates.

2. **Open recursion.** Passing the backend as a value is not enough: if a
   backend recurses through `self`, a wrapper only observes the calls the
   *runtime* initiates — the inner backend's own nested calls bypass it. So
   every method also takes `top`, the outermost backend, and recurses through
   `top`. Now `Traced` observes EVERY call (the test pins `invoke_count() == 5`
   for `(fact 5)`; a `self`-recursing wrapper would see `1`).

The lesson generalizes: "swappable" is cheap; "composable" needs the fixpoint.
Stopping at #1 would have left the exact "almost composes" seam that rots into a
hardcode — which is the failure mode this whole exercise is about.

## Slot resolution: the Ir/env cut, validated

Lexical variables resolve to `(up, idx)` slots at analyze time (`Ir::Local`), so
the runtime never searches a frame by name — `frame_get` is a pointer walk plus
an index. Globals stay `Ir::Global(Sym)`, resolved through the Var table at call
time (late binding preserved). Resolution lives in `analyze`, so both tiers get
it for free and share one frame-layout definition (`Runtime::build_call_frame`).

Doing this was the real test of whether the `Ir`/env split was cut in the right
place. Two findings:

- **The cut held.** `analyze` grew a compile-time scope stack; the runtime frame
  became `Vec<u64>`; both backends changed only their `Local`/`Let`/`Lambda`
  arms. No change to the value model, the `CodeSpace` contract, or the macro
  pipeline. The seam absorbed it.
- **It caught a real bug** (the point of the stress tests). `analyze` counts a
  `let` as one pushed frame, but the runtime initially ran the *first* binding's
  init under the parent env directly — off by one level, so an init referencing
  an outer variable walked past the root frame and panicked. The
  `slot_resolution_*` tests (deep capture, shadowing, on both tiers) surfaced it;
  the fix is that the first init also runs under an (empty) let frame. This is
  exactly the class of bug name-based lookup silently tolerates and slot
  addressing makes loud — which is a point in favor of resolving early.

## What a real toolkit adds on top (not sketched here)

- A JIT `CodeSpace` whose `define`/`invoke` are incremental and late-bound
  (call functions not yet defined). This is the genuinely hard engineering.
- GC + a `Handle` rooting discipline; the one place the axes fuse is rooting the
  compiler across a macro `invoke`.
- The runtime library (persistent collections, seqs) generic over `ValueModel`.
