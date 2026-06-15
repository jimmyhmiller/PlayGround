# coil is AOT — architecture correction

This supersedes the execution-model framing in `ELABORATION.md`. coil is an
**ahead-of-time compiler**, not an interpreter. A coil program is compiled to a
native object / executable through MLIR → LLVM. There is **no tree-walking
evaluator that runs the object language**. The earlier "build = eval, run macros
in the same interpreter" wording is withdrawn for the program's execution model.

## The pipeline (all AOT)

```
source ──read──▶ forms : Val
       ──expand──▶ core forms          (macro expansion; structural)
       ──emit────▶ MLIR module          (symbol-table-directed codegen — the "mapping")
       ──verify──▶ (real MLIR diagnostics, mapped to source spans)
       ──passes──▶ lowered module        (canonicalize, convert-to-llvm, …)
       ──llvm────▶ LLVM IR
       ──codegen─▶ object / executable
```

The program never runs inside coil. It runs as native code, later, on its own.

## "No interpreter" — what that means precisely

- **Runtime:** the compiled program is native. coil ships a compiler + runtime
  support, not an evaluator.
- **Codegen (`emit`) is not interpretation.** Walking core forms with a symbol
  table to *build* IR is ordinary AOT backend work (exactly what `lispier`'s
  `ir_gen.rs` did). It produces IR; it does not execute the program. This walk
  is the heart of the "mapping" and is what we build next.
- **Macros:** compile-time metaprogramming is **staging**, not object-language
  interpretation. The model is Rust's proc-macros, not Lisp's `eval`:
  - A bootstrap set of expanders is provided by the compiler (host = Rust).
  - User macros that need real compile-time computation are themselves compiled
    AOT (by coil, to native) and *called* by the compiler during expansion.
  - There is never a general tree-walking interpreter of coil in the loop.

This keeps the `defmacro` ergonomics from the design intact (macros are coil
functions over `Val`) while honoring AOT: the macro is *compiled*, then invoked,
just like a proc-macro. It also avoids `lispier`'s mistake — the macro is a
normal coil function, not a hand-marshalled `*const Value -> *mut Value` FFI shim,
because the compiler owns both sides and shares the `Val` ABI.

## Consequence for the build order

Because macro staging needs a working compiler, we build the **non-macro AOT
spine first** and add staged macros last:

1. **read** — done.
2. **emit**: core forms → MLIR via a `Backend` trait. *This increment.* Tested
   without MLIR using a `RecordingBackend` that logs builder calls, so the
   mapping is verified before LLVM is in the picture.
3. **MeliorBackend**: implement `Backend` against real MLIR; run a hand-written
   core-form program to an object file end-to-end.
4. **expand**: a fixed expander for the surface sugar (`defn`, op-call, `(: …)`,
   control flow) — structural, no user computation yet.
5. **staged macros**: compile-and-call user `defmacro`s (proc-macro model).

Steps 1–3 are a complete AOT compiler for the *core* language. Sugar (4) and
user macros (5) are layered on without ever introducing an interpreter.

## What stays valid from the design docs

- DESIGN.md's thesis (MLIR is the object language; IR nodes are first-class) —
  unchanged.
- SPEC.md surface and the total mapping to the generic op form — unchanged; it
  is realized by `emit`.
- KERNEL.md's primitive catalog — unchanged, but it is the **compiler's** API for
  building IR, not an interpreter's runtime.
- ELABORATION.md's *hygiene* and *anti-double-emit* analysis still applies to the
  expander/staging; only its "single tree-walking pass that also runs the
  program" framing is replaced by the staged-AOT model above.
