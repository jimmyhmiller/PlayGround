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
4. **[ELABORATION.md](ELABORATION.md)** — the hard part, resolved. Single-pass
   elaboration (build = eval), the anti-double-emit rule, the three ways a macro
   sees a type (`infer-results` / `value-type` / `build` + `with-scratch`),
   ordered compile-time effects, and scope-set hygiene worked through
   `defstruct`. This is where the remaining design risk lived.
5. **[prelude.coil](prelude.coil)** — proof. `defn`, `print`, `defstruct`,
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

## Gaps the prelude surfaced — and how they were resolved

Writing `prelude.coil` against `KERNEL.md §4` exposed primitives the kernel must
bless. The load-bearing ones are now resolved in **ELABORATION.md**; the rest are
small helpers.

- **`build` — expand-time evaluation of an op-form to a `Value`.** *Resolved
  (ELABORATION §1–3, now a kernel prim):* elaboration is single-pass (build =
  eval), so `build` commits an op and you splice the returned `Value`
  (anti-double-emit rule). Pure type queries use `mlir/infer-results` /
  `mlir/value-type`; speculation uses `with-scratch`.
- **Compile-time mutable state.** *Resolved (ELABORATION §4):* `atom`/`swap!`/
  `get`/`assoc` scoped to the compilation unit, with a defined top-to-bottom
  effect order; `intern-cstring!` is idempotent by construction.
- **Hygiene × `defstruct`'s nested `defmacro`.** *Resolved (ELABORATION §5–6):*
  scope-set hygiene + a prelude fix — generate accessors by closing over
  `sty`/`idx` as **values** spliced once, not via nested-quasiquote
  double-unquote. Values can't capture, so the generated macros are hygienic.
- **`module-body` / `block-thunk` / insertion helpers** — blessed kernel
  insertion-scope wrappers (`mlir/module-body`, thunk-based `mlir/with-block`).
  Small; still to be pinned down in code.
- **Parameter introspection** (`param-type`, `param-name`, `bind-args`) and the
  `(: name type)` parameter form — a tiny reader/AST helper, kernel-level.
- **Implicit terminators** (`with-implicit-terminator`, `*implicit-terminator*`)
  — prelude policy threaded via a kernel dynamic var; expressible without a new
  special form.

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
