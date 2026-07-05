# microlang

A trait sketch that re-cuts the dynamic-language toolkit at the two axes the
original conflated, with three micro-languages that exercise it. Backed by a
tree-walking executor so the *traits* are the point, not the codegen.

## The two axes as traits

| Axis | Trait(s) | File | Carved at |
|---|---|---|---|
| Value model | `Repr`, `ValueModel` | `model.rs` | **immediacy** (`is_immediate(Cat)`), not float-vs-tagged |
| Execution | `CodeSpace` (+ `TreeWalk`) | `code.rs` | meaning (`Ir`) vs strategy; re-entrant `invoke` |

`value.rs` holds the neutral vocabulary (`Cat`, `Val`, `Obj`) with `Int` a peer
of `Float`. `runtime.rs` ties it together: reader (code is data), `encode`/
`decode` (the boxing seam, `allocs`-counted), `macroexpand` (re-enters compiled
code), `analyze` (`Val` -> `Ir`), and the value-model-aware primitives.

## Run it

```
cargo run --example calc        # value axis: same engine, two models, measured allocs
cargo run --example microlisp   # eval axis: defmacro, recursion, macro-time re-entrancy
cargo test                      # locks both
```

`calc` prints integers free on `LowBit` / boxed on `NanBox`, and the exact
inverse for floats: the whole "value layout is a free choice" claim, made real.
`microlisp`'s `add2` macro calls the compiled `inc` during its own expansion.

## Two real models, ~30 lines apart

`LowBitModel` and `NanBoxModel` differ in one method (`is_immediate`) plus the
bit twiddling. Both drive the same generic `arith`. That is the discipline that
keeps the axis honest: a second real, tested consumer, not a fork.

## Honest design tension (found while building this)

`Runtime<M, C: CodeSpace<M>>` bakes the backend `C` into the runtime's type.
That makes the two-tier story typecheck cleanly (`Runtime<M, TreeWalk>` today, a
`Runtime<M, Jit>` tomorrow, nothing above changes) but it makes a *wrapping*
backend (e.g. a `Traced<Inner>` that delegates) awkward, because `Inner::eval_ir`
wants `Runtime<M, Inner>` while it is handed `Runtime<M, Traced<Inner>>`. A
production cut would thread the code space as a value/`dyn` object rather than a
type parameter, or make `eval_ir` generic over the runtime. Noting it rather
than hiding it, because "the abstraction almost composes" is exactly the kind of
thing that rots into a hardcode later if left unsaid.

## What a real toolkit adds on top (not sketched here)

- A JIT `CodeSpace` whose `define`/`invoke` are incremental and late-bound
  (call functions not yet defined). This is the genuinely hard engineering.
- GC + a `Handle` rooting discipline; the one place the axes fuse is rooting the
  compiler across a macro `invoke`.
- The runtime library (persistent collections, seqs) generic over `ValueModel`.
