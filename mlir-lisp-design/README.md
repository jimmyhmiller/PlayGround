# coil — a low-level Lisp with first-class MLIR

A redesign of the `lispier` / `mlir-lisp` experiments. The thesis: **MLIR is the
object language, not a backend** — IR nodes are first-class values, building IR
is evaluation, and macros are ordinary compile-time functions sharing one value
universe. That single move dissolves the two things that sank the earlier
attempts: a two-headed macro system and a lisp→MLIR mapping smeared across
phases.

## Read in this order

1. **[DESIGN.md](DESIGN.md)** — rationale. What went wrong in `lispier`
   (the two-macro-system problem; macros running blind so the mapping leaked
   into `ir_gen`, post-passes, and the parser) and the one idea that fixes both.
2. **[SPEC.md](SPEC.md)** — the surface. Reader grammar, special forms, and the
   total desugaring of every operation to MLIR's generic form via one `op`.
3. **[KERNEL.md](KERNEL.md)** — the frozen Rust core. The `Val` type, the
   evaluator, the complete MLIR primitive catalog, hygiene, phasing, diagnostics.
4. **[prelude.coil](prelude.coil)** — proof. `defn`, `print`, `defstruct`,
   control flow, dialects, and passes written *in the language* over the kernel
   primitives. If the prelude holds, the design holds.

## The claim, in one example

```clojure
(defn add [(: a i32) (: b i32)] -> i32
  (func.return (arith.addi a b)))
```

Everything except the reader and a handful of kernel ops is library code.
`defn`, `(: …)`, and `arith.addi`-with-inferred-result are all in `prelude.coil`;
the result type of `arith.addi` comes from MLIR's `InferTypeOpInterface` at build
time, not from a language-level rule.

## Gaps the prelude surfaced (feed back into KERNEL.md)

Writing `prelude.coil` against `KERNEL.md §4` exposed primitives the kernel must
add or bless. These are the concrete next decisions, not hand-waving:

- **Compile-time mutable state.** `intern-cstring!` needs `atom`/`swap!`/`get`/
  `assoc` (or equivalent) usable during expansion. Decision: provide a small
  persistent-map + `atom` in the kernel data prims, scoped per compilation unit.
- **`build` — expand-time evaluation of an op-form to an `MlirValue`.** Used by
  `widen-i64` to learn an operand's type before deciding what to emit. This is
  the linchpin of "macros can see types"; it must be a first-class kernel prim
  (`mlir/build form -> Value`) with a defined builder scope.
- **`module-body` / `block-thunk` / insertion-scope helpers.** The prelude pokes
  the module's top block to append globals/dialects. Needs a blessed
  `mlir/module-body` and a thunk-based `with-block`, both in the kernel.
- **Parameter introspection** (`param-type`, `param-name`, `bind-args`) and the
  `(: name type)` parameter form — a tiny reader/AST helper, kernel-level.
- **Implicit terminators** (`with-implicit-terminator`, `*implicit-terminator*`).
  Region terminator insertion is prelude policy but needs a dynamic var the
  kernel threads; confirm it's expressible without a kernel special form.
- **Hygiene × `defstruct`'s nested `defmacro`.** `defstruct` defines macros whose
  bodies quasiquote the field index — a macro-defining-macro with two quote
  levels. This is the stress test for the hygiene/`gensym` story in `KERNEL §5`;
  it must round-trip cleanly or the hygiene model needs revisiting.

## Status

Design only. No implementation yet. Recommended build path (DESIGN.md §9):
evolve `claude-experiments/lispier`'s Rust core — keep the reader, `melior`
bindings, and JIT; replace the macro/IR-gen middle with the `Val` evaluator and
the primitive catalog; port `examples/` over one builtin at a time as the
regression suite; then land `defdialect`/`defpattern` codegen.

## Relationship to the old projects

| Old | Fate |
|---|---|
| `lispier` reader / tokenizer | **keep** |
| `lispier` `melior` bindings, JIT, pass runner | **keep**, wrap as kernel prims |
| `lispier` `src/macros/builtins/*` (Rust macros) | **delete**, move to `prelude.coil` |
| `lispier` `jit_macro.rs` / `macro_compiler.rs` / `DynamicMacroContext` | **delete** (interpreted macros replace it) |
| `lispier` `StringCollector`, per-op type-inference hacks | **delete** (library + MLIR inference) |
| `mlir-lisp` `defirdl-dialect` / `defpdl-pattern` (parse-only) | **finish** as `defdialect`/`defpattern` with real codegen |
