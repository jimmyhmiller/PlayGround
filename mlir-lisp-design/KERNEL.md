# coil — Kernel Specification (the frozen Rust core)

The kernel is the part that *cannot* be written in the language. Everything not
listed here lives in `prelude.coil` and user code. The kernel's job: read s-exprs
into `Val`, evaluate a tiny set of special forms, and expose MLIR as primitives.

Sizing goal: the kernel is **small enough to audit in an afternoon**. If a
feature can be a prelude macro, it is not in the kernel.

---

## 1. The `Val` type

One representation, shared by the runtime program *and* the compile-time
(macro) evaluator. This shared representation is what removes `lispier`'s
`*const Value -> *mut Value` FFI boundary entirely.

```rust
enum Val {
    // --- data / homoiconic syntax ---
    Unit,
    Bool(bool),
    Int(i128, IntKind),          // IntKind = {width, signed}
    Float(f64),
    Str(Rc<str>),
    Sym(Symbol),                 // name + optional namespace + hygiene mark
    Keyword(Rc<str>),
    List(Rc<[Val]>),
    Vec(Rc<[Val]>),
    Map(Rc<Vec<(Val, Val)>>),    // insertion-ordered
    Fn(Rc<Closure>),             // params, body, captured env
    Macro(Rc<Closure>),          // same, but runs at expansion time

    // --- reflected MLIR (thin handles over the C API, à la melior) ---
    Type(MlirType),
    Attr(MlirAttribute),
    Value(MlirValue),            // SSA: block arg or op result
    Op(MlirOperation),
    Block(MlirBlock),
    Region(MlirRegion),
    Dialect(MlirDialectHandle),
    Module(MlirModule),
    Pass(MlirPass),
    Pipeline(Rc<PipelineSpec>),  // composable, before lowering to OpPassManager
    Context(MlirContextRef),     // usually implicit, occasionally explicit
}
```

`Symbol` carries a **hygiene mark** (a scope-set or rename id) so macro-introduced
identifiers don't capture. SSA `Value`s need no hygiene — they're opaque handles.

`Syntax` wrapping: every `Val` read from source is paired with a `Span`
(file, byte range). Spans ride along through expansion so MLIR diagnostics map
back to user code (the §7 diagnostic model). In Rust this is an out-of-band
`SpanTable` keyed by node identity rather than wrapping every `Val`.

---

## 2. Evaluation model

A single tree-walking evaluator `eval(form, env) -> Val`, used in two phases:

- **Compile-time (expansion):** macros run here. Same evaluator, same `Val`,
  with a live MLIR `Context` available so macros can build/inspect IR.
- **Build-time:** evaluating op-forms constructs IR into the current builder
  scope. (For a pure-AOT compile these coincide; for staged/`eval-when` they're
  ordered.)

There is no separate "interpreter for macros" vs "compiler for code" — both are
`eval`. Runtime *execution* of the compiled program is separate (JIT/AOT), and
only the program (not macros) is ever JITed.

### Builder state (dynamic)

The evaluator threads an implicit `Builder` in dynamic scope:

```
Builder { context, current_block, insertion_point, module }
```

- `op` inserts at `insertion_point` and advances it.
- `block`/`region` push a nested builder scope (`with-block`, `with-region`).
- Prelude code rarely touches `Builder` directly; it falls out of op-call sugar.

---

## 3. Special forms (intrinsic)

| Form | Semantics |
|---|---|
| `(quote x)` | returns `x` unevaluated |
| `` (quasiquote x) `` | template; `~` evals, `~@` splices |
| `(if c t e)` | `e` optional; `c` is truthy unless `false`/`nil`/`Unit` |
| `(do e…)` | sequence; value is last |
| `(let [n v …] e…)` | sequential lexical binding; `n` may destructure `[a b]` |
| `(fn [p…] e…)` | closure; `&rest` for variadic |
| `(def name v)` | top-level / module binding |
| `(defmacro name [p…] e…)` | define a macro (a `Macro` closure) |
| `(eval-when phase e…)` | `phase ∈ {:compile :load :runtime}` |
| `(require …)` / `(require-for-syntax …)` | module loading with phase |

Macros expand via `eval` of the `Macro` closure on **unevaluated** argument
forms, hygienically; the result is re-expanded until fixpoint (depth-capped).

Everything else — `when`, `cond`, `and`, `or`, `case`, `loop`, `for`, `->`,
threading, etc. — is a prelude macro.

---

## 4. MLIR primitives

Primitive functions exposed to the language (names shown as the language sees
them). These are the *entire* surface between the language and MLIR.

### Op / block / region construction
```
(mlir/op name attrs operands results regions successors) -> Op
(mlir/block param-types)            -> Block          ; detached
(mlir/block-arg block i)            -> Value
(mlir/region blocks)                -> Region
(mlir/op-results op)                -> [Value]
(mlir/with-block block thunk)       -> a              ; sets insertion point
(mlir/append-block region block)    -> Unit
```

### Types & attributes
```
(mlir/parse-type str)               -> Type           ; "!"-literal backend
(mlir/parse-attr str)               -> Attr           ; "#"-literal backend
(mlir/integer-type width signed?)   -> Type
(mlir/float-type kind)              -> Type            ; :f16/:f32/:f64/:bf16
(mlir/index-type)                   -> Type
(mlir/function-type ins outs)       -> Type
(mlir/shaped-type ctor shape elt)   -> Type            ; vector/memref/tensor
(mlir/value-type value)             -> Type
(mlir/type= a b) (mlir/attr= a b)   -> Bool
(mlir/int-attr i type)              -> Attr
(mlir/string-attr s) (mlir/type-attr t) (mlir/symbol-ref s) -> Attr
(mlir/array-attr [attr…]) (mlir/dict-attr {k v…})           -> Attr
```

### Inference & introspection (the Problem-2 fix)
```
(mlir/infer-results name attrs operands regions) -> [Type]   ; InferTypeOpInterface, PURE
(mlir/op-has-trait name trait)      -> Bool
(mlir/op-name op) (mlir/op-attr op k) (mlir/op-operand op i) -> …
(mlir/value-type value)             -> Type                   ; PURE
(build form)                        -> Value                  ; ELABORATE form NOW, commit IR
(with-scratch thunk)                -> a   ; throwaway builder scope; ops discarded on return
```
A macro learns a result type at expansion time three ways, in order of
preference (see ELABORATION.md §3): `mlir/infer-results` (pure, builds nothing),
`mlir/value-type` on an already-built value (pure), or `build` (commits the op —
then you MUST splice the returned `Value`, never the original form, per the
anti-double-emit rule). `with-scratch` is for genuinely speculative construction
(e.g. try-to-verify-then-choose); ops built inside it never reach the module.
These were the linchpin gaps the prelude surfaced; they are now kernel-level.

### Module, verify, passes, codegen
```
(mlir/module)                       -> Module          ; usually *module*
(mlir/verify op)                    -> Result<Unit,Diag>
(mlir/load-dialect name)            -> Unit
(mlir/parse-pipeline str)           -> Pipeline
(mlir/pipeline passes…)             -> Pipeline         ; compose Vals
(mlir/run-passes module pipeline)   -> Result<Unit,Diag>
(mlir/jit module) -> Engine ; (mlir/invoke engine sym args) -> Val
(mlir/emit-object module path) (mlir/emit-llvm module) -> Unit
```

### Meta-dialect registration (finishes mlir-lisp Phase 3)
```
(mlir/register-irdl module)         -> Unit   ; turn irdl.* ops into a live dialect
(mlir/register-pdl module patterns) -> Unit   ; register rewrite patterns
(mlir/apply-patterns op patterns)   -> Unit
```

### Data / utility prims
```
list vec cons first rest nth count map filter reduce concat apply
str sym keyword gensym name namespace
=  <  >  +  -  *  /  not    ; on plain numbers/strings (NOT the op-emitting ones)
print-val  read-string  slurp  ; host IO for tooling
```

> Naming: data `+` (host integer add, used in macros) is distinct from emitting
> `arith.addi`. The prelude exposes the latter as `arith.addi`/`i+`; the kernel
> `+` only does compile-time arithmetic on `Val::Int`.

---

## 5. Hygiene

- Each macro expansion gets a fresh scope mark. Identifiers *introduced* by the
  macro carry that mark; identifiers *passed in* keep theirs. Resolution compares
  marks, so a macro's `tmp` never collides with the caller's `tmp`.
- `(gensym "x")` for explicit fresh names; `(unhygienic e)` to intentionally
  capture; `(datum->syntax ctx datum)` to graft identifiers into a given scope.
- SSA values, block refs (`^bb`), and symbol refs (`@f`) interact with hygiene:
  `^bb`/`@f` resolve in the *definition* scope by default; opt out via
  `unhygienic`. This is the §11.2 open question made concrete — the rule above is
  the proposed default.

---

## 6. Phase / loading algorithm

```
load(path):
  forms      = read(path)
  for f in forms scanning top-level:
      if (require-for-syntax m): load(m) into compile-env
      if (require m):            queue m for runtime-env
  expand-env = base ∪ compile-env
  expanded   = fixpoint-expand(forms, expand-env)     ; macros run here
  build(expanded)                                      ; op-forms construct IR
```

`require-for-syntax` populating `compile-env` *before* expansion is the whole
trick: it gives macros their dependencies at the right time, replacing the
fragile pre-scan-for-`defmacro` heuristic in `lispier`.

---

## 7. Diagnostics (the main UX risk, designed up front)

1. Reader attaches `Span` to every node (via `SpanTable`).
2. Expansion propagates spans: an introduced node inherits the macro-call span;
   a passed-through node keeps its own (Racket "syntax taint"/`syntax/loc`).
3. `mlir/op` records, on each created `MlirOperation`, the originating span as an
   MLIR `Location` (`FileLineColLoc` or a fused loc).
4. MLIR diagnostics (verify/pass failures) carry that `Location`; the kernel maps
   it back to the `Span` and renders a source-pointing error — never raw
   expanded IR.

This makes "first-class MLIR" survivable: errors talk about *your* code.

---

## 8. What is deliberately NOT in the kernel

`defn`, `defmacro`-sugar beyond the primitive, `if`-with-elif, `when`, `cond`,
`and`/`or`, `let*`/`letrec` sugar, arithmetic operators that emit ops, `print`,
`defstruct`, `defdialect`, `defpattern`, `passes`, `emit`, `require-dialect`,
type constructors like `vector`/`memref`, threading macros, pattern-match `case`.

All of these are in `prelude.coil`. If that file can express them over the
primitives above, the design's central claim holds.
