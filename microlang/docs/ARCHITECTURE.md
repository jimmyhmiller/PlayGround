# Architecture

microlang is not "an interpreter." It is one small, meaning-free IR with several
pluggable execution engines behind a single trait, one pluggable value
representation, and a moving GC underneath. "Interpret vs compile" is itself a
point on the execution axis, not a fork in the design.

## The pipeline

```
surface language        reader          analyze            CodeSpace<M>
(Scheme / microlisp) ─▶ (code-is-data)─▶ (Val form → Ir) ─▶ runs the Ir
                        Val / cons cells  resolve binding    over a ValueModel M,
                                                             on a moving GC heap
```

1. **Reader** turns source text into *data* — `Val` / cons cells (`value.rs`).
   Code is data; the reader has no notion of "expression."
2. **`analyze`** (`runtime.rs`) lowers a `Val` form into `Ir` (`ir.rs`). This is
   the one compile step everything shares. It resolves lexical variables to
   `Local { up, idx }` (frame-hops + slot, no run-time name search), leaves
   globals as `Global(Sym)` (late-bound, so a reference may precede its
   definition), and interns literals into a constant pool referenced by
   `Const(ConstId)`.
3. A **`CodeSpace<M>`** executes that `Ir`. Which engine is a *choice*.

## The two axes are traits; the IR sits between them

| Axis | Trait | Decides |
|---|---|---|
| Value model | `ValueModel` / `Repr` (`model.rs`) | how a value lives in a `u64` — tagging vs NaN-boxing vs high-bit; immediate vs boxed |
| Execution | `CodeSpace<M>` (`code.rs`) | *how* the `Ir` runs; the trait is just `eval_ir` + `invoke` |

The `Ir` is deliberately neutral to both. It says `Prim(Add)`, `Call`, `Global`,
`Dispatch{site,method}` — semantic operations, never `AddRawLowBit` or a resolved
pointer. A frontend lowers to it once; every execution strategy and value model
consumes the same `Ir`. `matrix.rs` runs the full cross product — {3 value
models} × {3 general tiers} × {dispatch} × {speculation}, moving GC underneath —
and checks all 45 combinations compute the same answer.

## The `CodeSpace` contract

```rust
pub trait CodeSpace<M: ValueModel> {
    fn eval_ir(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, ir: &Ir, locals: &Locals) -> u64;
    fn invoke(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64;
}
```

Two subtleties make it *compose*, not merely swap:

- **Backend is a value, not a type parameter.** `Runtime<M>` carries no backend;
  you pass the code space in. So a wrapper like `Traced` can hold a
  `Box<dyn CodeSpace<M>>` and delegate.
- **Open recursion via `top`.** Every method also receives `top`, the outermost
  backend, and recurses through it — so a wrapper observes *every* call, not just
  the ones the runtime initiates. (The `Traced` test pins `invoke_count() == 5`
  for `(fact 5)`; a `self`-recursing wrapper would see `1`.)

## The five engines

| Engine | File | Kind | Notes |
|---|---|---|---|
| `TreeWalk` | `code.rs` | recursive interpreter | re-dispatches the `Ir` each run; escape continuations via unwind |
| `ClosureComp` | `compiled.rs` | closure compiler | compiles each `Ir` subtree to a Rust closure once; caches bodies |
| `BytecodeVm` | `bytecode.rs` | emit tier | value model emits model-specific bytecode via `ModelEmit` |
| `CekMachine` | `cek.rs` | stackless abstract machine | full/delimited multi-shot continuations; GC-survivable |
| `JitCranelift` | `jit_cranelift.rs` | native emit tier | compiles the `Ir` to host machine code via Cranelift; opt-in `--features jit` |
| `Tiered` | `jit_cranelift.rs` | tiering wrapper | JIT with an automatic `CekMachine` fallback per body |
| `Traced` | `code.rs` | wrapper | counts invocations (feeds speculation) |

`JitCranelift` is the machine-code analogue of `BytecodeVm`: per-model arithmetic
(the same value-axis split `ModelEmit` encodes — LowBit/HighBit shift to untag,
NanBox boxes so it has no fast path) is emitted inline, while everything
heap/globals/call-shaped funnels back through the runtime via `extern "C"` shims.
Beyond the bytecode tier it adds `let`/`set!`, **proper tail calls** (a trampoline,
so unbounded tail recursion is O(1) stack), and a **guarded fixnum fast path** that
range-checks and falls back to the runtime's promoting arithmetic — giving it the
full numeric tower (bignum, floats) the tree-walker has, which the wrapping
bytecode tier lacks. That is enough to run **all of Scheme except first-class
continuations**; `Tiered` composes the JIT with the `CekMachine`, running each
body native when it can and on the stackless machine when it uses `call/cc` /
`apply`. The full R7RS conformance suite passes on `Tiered` (54/61 live cases
fully native, oracle-checked). It still does not model the GC safepoint (native
temporaries are not yet roots — the frame/roots emit axis is the next step). See
[CODEGEN_AXES.md](CODEGEN_AXES.md).

Some `Ir` nodes are only meaningful under some strategies: `Prim::CallCc` is a
well-formed node on every tier, but only `CekMachine` can give it meaning; the
others raise a loud, specific error. See [CONTINUATIONS.md](CONTINUATIONS.md).

## Underneath: value model + moving GC

`Runtime<M>` (`runtime.rs`) owns the heap (`Vec<Obj>`), globals, the constant
pool, the interned-symbol table, the shadow stack (GC roots), and primitive
dispatch (`prim`). `encode`/`decode` is the boxing seam the `ValueModel`
controls. `gc.rs` is a moving (Cheney copying) collector.

One constraint threads through everything: **`Ir` holds no embedded heap
pointers** — literals go through the constant-pool `Const(ConstId)` indirection.
That is exactly what lets a moving GC relocate the whole heap without rewriting
compiled code, and it is why continuations (`Kont` chains) could be made
GC-survivable. See the GC section of the top-level [README](../README.md).

## Where the languages live

The core (`microlang` crate) knows nothing about any surface language. Each
frontend is a separate crate/example touching only the public API:

- `scheme/` — the R7RS-flavored Scheme ([SCHEME.md](SCHEME.md)).
- `clojure-stub/` — a second frontend that keeps the library/language split
  honest ([LIBRARY_LANGUAGE_SPLIT.md](LIBRARY_LANGUAGE_SPLIT.md)).
- `examples/microlisp`, `examples/calc` — small languages that exercise the axes.

So a language is a *library on top of the toolkit*, not baked into it — the whole
point the project sets out to demonstrate.
