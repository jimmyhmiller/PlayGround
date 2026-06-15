# A Low-Level Lisp with First-Class MLIR

> Working name: **`coil`** (rename later). This is a design document, not yet an
> implementation. It supersedes the two earlier experiments
> (`claude-experiments/lispier`, `claude-experiments/mlir-lisp`, and the Zig
> ports) by keeping what worked and replacing the two things that did not: the
> macro system and the lisp→MLIR mapping.

## 0. What we are trying to fix

`lispier` already proves the hard parts are possible: it reads s-expressions,
expands macros, emits MLIR through `melior`, runs a real pass pipeline, and JITs
GPT-2 on a GPU. The surface syntax (`(require-dialect arith)`, `{:attrs ...}`,
`(: 42 i64)`, regions as `(do ...)`) is good and worth keeping. Two things went
wrong, and they are the entire reason for this redesign.

### Problem 1 — The macro system was two incompatible halves

There are **two unrelated macro mechanisms** in `lispier`, and neither is the
one you actually want:

1. **Rust builtins** (`src/macros/builtins/*`): `defn`, `if`, `when`, `cond`,
   `+`, `print`, `defstruct`… Each is a hand-written `impl Macro` in Rust that
   pattern-matches `Value`s and rebuilds `Value`s. Powerful but you have to
   recompile the compiler to add one, and they encode MLIR layout knowledge by
   hand (e.g. `defn.rs` literally hard-codes the `func.func` attribute map and
   the `(do (block …))` shape).

2. **JIT `defmacro`** (`src/macros/jit_macro.rs`, `expander.rs`): a user macro
   is a `func.func` compiled to native code with the signature
   `unsafe extern "C" fn(*const Value) -> *mut Value`. To write a macro you have
   to manipulate the host's `Value` enum **through raw FFI pointers** from
   inside the generated MLIR. To make that work, `DynamicMacroContext` scrapes
   every `require-dialect`/`extern`/`func.func` seen so far, finds the
   `func.func` whose `sym_name` matches the macro name, re-bundles them into a
   synthetic source, and JIT-compiles that. It is brittle, unsafe, has no
   hygiene, and "macro-defining macros" are explicitly unsupported.

These two halves don't share a value model, a binding model, or a phase model.
That is the "never made the macro setup well" problem: there was never *one*
macro system — there was a Rust escape hatch and a fragile FFI trick.

### Problem 2 — Macros ran blind, so the mapping leaked everywhere

Macros in `lispier` are purely **syntactic**: `Value → Value`, run *before*
anything MLIR exists. So a macro can never ask "what type does this produce?",
"does this op infer its results?", "is this value in scope / dominating?".
Everything that *needs* that information got bolted on somewhere else:

- **Type inference** is special-cased twice: once in `ir_gen.rs`
  (`operation_supports_type_inference`, `enable_result_type_inference`) and again
  as per-op hacks in the Zig `builder` (`arith.constant`, `arith.addi`, …).
- **SSA threading / naming** is split between `declare`/`def`, `%name` rewriting,
  and a symbol table in `ir_gen`.
- **Strings & globals** need a whole post-expansion `StringCollector` pass that
  invents `llvm.mlir.global`s and rewrites `__print_internal__` into
  `llvm.call @printf` with `var_callee_type` by hand.
- **`llvm.call`** needs bespoke `operandSegmentSizes` / `op_bundle_sizes`
  handling inside `build_operation`.
- **structs** hand-roll GEP offsets and a `(new T)` → `@malloc` rewrite.

Each of these is the language re-deriving something MLIR already knows, because
the layer that does the work (the macro) is too early to see the IR, and the
layer that sees the IR (`ir_gen`) is too late to be programmed in the language.

**The redesign collapses both problems into one idea:**

> Building MLIR *is* evaluation, and the program's data model *is* MLIR's IR.
> Macros are ordinary compile-time functions in the same evaluator, operating on
> the same first-class IR values — so they can see types, infer results, and
> query the context, because the context is right there.

---

## 1. Design principles

1. **MLIR is the object language, not a backend.** `Type`, `Attribute`, `Value`
   (SSA), `Operation`, `Block`, `Region`, `Dialect`, `Pass`, and `Context` are
   first-class runtime values you can bind, pass, print, and introspect.
2. **One macro system.** Macros are functions evaluated at compile time by the
   *same* interpreter that runs everything else. No Rust escape hatch required
   for the 90% case; no `*const Value` FFI trick ever.
3. **The surface is a total, bijective encoding of the generic op form.** Every
   operation — known or unknown, today's dialect or one you defined five minutes
   ago — is expressible with zero compiler changes. "High-level" forms are
   macros that emit generic ops.
4. **Let MLIR do MLIR's job.** Result types come from `InferTypeOpInterface`,
   verification from the verifier, lowering from passes, dominance from the
   builder. The language never re-implements them; it *exposes* them.
5. **Explicit phase tower.** Reader → macro/meta evaluation → IR construction →
   verification → passes → JIT/AOT. Each concern has exactly one home. The thing
   that bit us (work landing in the wrong phase) is designed out.
6. **Homoiconic at two levels.** Code is s-expression data (for macros), *and*
   IR is data (for transforms). PDL/Transform/IRDL are reflected as values too,
   so rewriting the IR is also "just programming."

---

## 2. The value universe

A single `Val` type spans both ordinary Lisp data and reflected MLIR handles.
There is **one** representation shared by the runtime and the compile-time
evaluator — which is precisely what kills Problem 1's FFI boundary.

```
Val =
  | Unit | Bool b | Int i (width,signed) | Float f | Str s
  | Sym s | Keyword k
  | List [Val] | Vec [Val] | Map {Val:Val}      ; homoiconic code/data
  | Fn closure | Macro closure                   ; callable
  ; --- reflected MLIR, all backed by C-API handles ---
  | MlirType   t
  | MlirAttr   a
  | MlirValue  v        ; an SSA value (block arg or op result)
  | MlirOp     op       ; an Operation (may be detached or inserted)
  | MlirBlock  b
  | MlirRegion r
  | MlirDialect d
  | MlirContext c
  | MlirPass / MlirPassManager / MlirModule …
```

`MlirType`, `MlirAttr`, etc. are thin wrappers over the MLIR C API objects
(exactly what `melior` already gives us). Crucially these are **not** a separate
universe reachable only through FFI — they are ordinary `Val` cases the
evaluator and macros manipulate directly.

`List`/`Vec`/`Map`/`Sym` are the homoiconic layer used for *syntax*. The MLIR
cases are the homoiconic layer used for *IR*. Macros bridge them.

---

## 3. Surface syntax: a total mapping to the generic form

MLIR's generic operation form is:

```
%results = "dialect.op"(%operands) ({regions}) <{properties}>
           {attributes} : (operand-types) -> (result-types)
           [successors]
```

The surface must encode *all* of it. The core form is `op` (everything else is
sugar that expands to it):

```clojure
(op "dialect.name"
    :operands  [a b]                 ; Vals that are MlirValue
    :results   [i32]                 ; result types, OR omit to infer
    :attrs     {:predicate (# slt)}  ; attributes (see §4)
    :regions   [ (region ...) ]      ; zero or more regions
    :successors[ ^bb1 ^bb2 ]         ; block references
    :as        x)                    ; bind the (single) result to `x`
```

But you almost never write `op`. The reader treats any symbol containing a `.`
as an operation head, giving the terse form that `lispier`/`mlir-lisp` already
liked:

```clojure
(arith.addi a b)                     ; results inferred via InferTypeOpInterface
(arith.constant {:value (: 42 i32)}) ; attr-only op
(func.return r)
```

Sugar rules (all are macros, see §6, none are compiler builtins):

| You write | Expands to |
|---|---|
| `(dialect.op args…)` | `(op "dialect.op" :operands [args…])` |
| `(dialect.op {attrs} args…)` | `…:attrs {attrs} :operands [args…]` |
| trailing `(region …)`/`(do …)` forms | `…:regions […]` |
| `(let [x e] body)` | binds `x` to the SSA result of `e` |
| `^name` | a block reference; `(block ^name [args] …)` defines one |
| `(: 42 i32)` | typed literal → `arith.constant`/attr depending on position |
| `@sym` | a `FlatSymbolRefAttr` / symbol reference |

### SSA values are real values, not text

In `lispier`, `%name` was a textual convention rewritten in several places. Here
an SSA result is a first-class `MlirValue` bound by `let`/`:as`. There is no
`%` string rewriting and no separate symbol table to keep in sync — lexical
scope *is* the SSA scope, because evaluating `(arith.addi a b)` returns the
`MlirValue` for its result.

```clojure
(defn add [(: a i32) (: b i32)] -> i32
  (let [s (arith.addi a b)]      ; s : MlirValue
    (func.return s)))
```

---

## 4. Types and attributes are first-class

Types and attributes evaluate to `MlirType` / `MlirAttr`. There are three ways
to write them, in increasing escape-hatch order:

```clojure
i32  f64  index  i1                 ; builtin type symbols
(vector 4 f32)  (memref ? ? f32)    ; type constructors (functions returning MlirType)
!llvm.ptr   !llvm.struct<(i64,i64)> ; '!'-prefixed = parse via mlirTypeParseGet
(# array<i32: 1, 2, 3>)             ; '#'-prefixed = parse via mlirAttributeParseGet
```

`!…` and `#…` are the universal escape hatch: **any** type or attribute MLIR can
parse is expressible, so a new dialect needs no language support. `(vector …)`
etc. are just library functions returning `MlirType` — defined in the standard
library *in the language*, not in the compiler.

This is the first big simplification over `lispier`: there is no enum of "known
types" baked into the parser. A type is whatever evaluates to an `MlirType`.

---

## 5. The phase tower (where each concern lives)

```
 source text
    │  reader  (tokenize + read into Val s-expressions)
    ▼
 forms : Val
    │  meta-evaluation  ──────────────────────────────────────────┐
    │   • macro expansion (functions on Val → Val, hygienic)       │ compile-time
    │   • `eval-when :compile` blocks run here                     │ world has full
    │   • dialect / type / pass *definitions* register here        │ access to MLIR
    ▼                                                              ┘ (a live Context)
 IR construction  (evaluate op-forms against a live OpBuilder+Context)
    │   • each op-form builds an Operation and inserts it
    │   • result types come from MLIR inference at *this* moment
    ▼
 module : MlirModule
    │  verify   (mlirOperationVerify — real diagnostics, mapped to source spans)
    ▼
 pass pipeline  (a Val describing passes; runnable, inspectable, composable)
    ▼
 JIT (ExecutionEngine) or AOT (emit object / LLVM IR)
```

The fix for Problem 2 is the dashed box: **the compile-time world has a live
MLIR `Context`.** A macro running in meta-evaluation can construct trial ops,
ask their result types, query interfaces, even run a verification — because MLIR
is present at compile time, not just at the end. Nothing needs to be "deferred
to `ir_gen`" anymore, because `ir_gen` and the macro layer are the same world.

Concretely, the per-op type-inference special cases, the `StringCollector`
pass, and the `llvm.call` segment-size handling all become ordinary library
code that runs in this one pipeline with full information.

---

## 6. The macro system (the heart of the redesign)

### 6.1 One mechanism: compile-time functions over `Val`

A macro is a function evaluated at **compile time** that takes the unevaluated
argument forms and returns a replacement form:

```clojure
(defmacro unless [cond & body]
  `(if (not ~cond) (do ~@body)))
```

`defmacro` defines an ordinary closure; the only difference from `defn` is *when*
it runs (meta-evaluation phase) and that its arguments arrive unevaluated. The
same evaluator runs both. There is **no** `*const Value -> *mut Value`, no
`DynamicMacroContext` source-scraping, no separate JIT path. Macro-defining
macros work for free because it is all one evaluator with one environment.

> The Rust-builtin path doesn't disappear — it becomes the *bootstrap kernel*
> (§9). But user-facing macros, and eventually most of the "builtins", are
> written in the language.

### 6.2 Quasiquote that produces IR, not just syntax

`quasiquote`/`unquote`/`unquote-splicing` work over `Val`, so they compose with
the MLIR cases. Because an op-form *evaluates* to an `MlirOp`/`MlirValue`, a
macro can choose to work at the **syntax** level (return forms) or the **IR**
level (return already-built ops). Example — a macro that needs the result type:

```clojure
(defmacro widen-to-i64 [x]
  ;; runs at compile time, with a live builder/context:
  (let [v   (build x)                 ; build the operand, get an MlirValue
        ty  (value-type v)]           ; ask MLIR its type — first-class!
    (if (type= ty i64)
      v
      `(arith.extsi ~v :results [i64]))))
```

This is impossible in `lispier` (a syntactic macro can't know `x`'s type). Here
it's trivial, because the macro shares the IR-building world.

### 6.3 Hygiene

Macro-introduced bindings are renamed by default (gensym-backed), with explicit
`(unhygienic name)` / `(datum->syntax …)` escape hatches à la Racket. This kills
the accidental-capture class of bugs that hand-written `Value` rebuilding in
`lispier` had no protection against. SSA values are hygienic automatically
because they're opaque `MlirValue`s, not names.

### 6.4 Phase separation done explicitly

Borrowing Racket's tower, but small:

- `(require mod)` — runtime dependency.
- `(require-for-syntax mod)` — macros need it at compile time.
- `(eval-when :compile body…)` — run `body` during meta-evaluation.

This replaces the awkward `require` vs `require-macros` split (and the unwritten
unification in `docs/macro-system-future.md`): there is one `require`, plus
phase annotations when you mean compile-time. The loader does a pre-pass to
build the compile-time environment before expanding, exactly as that doc
sketched — but now it's principled because macros are normal functions.

### 6.5 Worked example: `defn`, `print`, `struct` as *library* macros

The painful builtins become a few lines of in-language macro, with no compiler
knowledge of `func.func` layout, printf, or GEP:

```clojure
;; defn — emits a generic func.func, no hard-coded attr map in Rust
(defmacro defn [name params -> ret & body]
  (let [ptypes (map param-type params)
        pnames (map param-name params)]
    `(func.func {:sym_name ~(str name)
                 :function_type (fn-type ~ptypes [~ret])
                 :llvm.emit_c_interface true}
       (region (block ~pnames ~@body)))))

;; print — string interning is a library concern using the live module
(defmacro print [fmt & args]
  (let [g (intern-cstring! fmt)]      ; adds an llvm.mlir.global to the module now
    `(llvm.call {:callee @printf
                 :var_callee_type !llvm.func<i32 (ptr, ...)>}
        (llvm.mlir.addressof {:global_name ~g :results [!llvm.ptr]})
        ~@args)))
```

`intern-cstring!` is a normal compile-time function that mutates the module via
the first-class `MlirModule`/builder API — replacing the entire `StringCollector`
post-pass. Same story for `defstruct`: GEP offsets come from
`mlirLLVMStructTypeGetElementType` + the data layout, queried at compile time,
not hard-coded `* 8`.

---

## 7. First-class dialects, passes, and rewrites

### 7.1 Defining dialects (IRDL, in-language)

The `mlir-lisp` experiment already prototyped `defirdl-dialect`. Keep it, but
actually *emit IRDL ops* and register the dialect with the live context (the
"Phase 3 codegen" that experiment never finished):

```clojure
(defdialect my
  (defop my.add
    :operands [(lhs AnyInteger) (rhs AnyInteger)]
    :results  [(out AnyInteger)]
    :traits   [Pure Commutative]
    :verify   (same-type lhs rhs out)))
```

This lowers to `irdl.dialect`/`irdl.operation` ops, is registered at compile
time, and from then on `(my.add a b)` builds a real `my.add` operation that
verifies. No recompiling the host.

### 7.2 Passes and pipelines are values

```clojure
(def pipeline
  (passes
    (canonicalize)
    (convert-arith-to-llvm)
    (convert-func-to-llvm)
    (reconcile-unrealized-casts)))

(run-passes module pipeline)
```

A pipeline is a `Val` you can build, splice, print, and pass around — so
`pipeline_builder.rs` becomes library code. You can also drop to a textual
pipeline string for anything exotic (`(pass-pipeline "builtin.module(...)")`).

### 7.3 Rewrites: PDL / Transform reflected as data

```clojure
(defpattern lower-my-add
  :match   (my.add ?lhs ?rhs -> ?ty)
  :rewrite (arith.addi ?lhs ?rhs -> ?ty))
```

`?x` are pattern variables. This compiles to PDL ops and registers with the
pattern rewriter, so user dialects can be lowered without C++. The Transform
dialect is exposed the same way for scheduling rewrites. Because patterns are
`Val`s, you can generate them with macros (e.g. derive a lowering for every op
in a dialect).

---

## 8. End-to-end example

```clojure
(require-dialect arith func scf)
(link-library :c)
(extern printf (fn-type [!llvm.ptr] [i32]) :vararg true)

(defn fib [(: n i32)] -> i32
  (scf.if {:results [i32]} (arith.cmpi {:predicate (# slt)} n (: 2 i32))
    (region (scf.yield n))
    (region
      (let [a (fib (arith.subi n (: 1 i32)))
            b (fib (arith.subi n (: 2 i32)))]
        (scf.yield (arith.addi a b))))))

(defn main [] -> i32
  (let [r (fib (: 10 i32))]
    (print "fib(10) = %d\n" r)
    (func.return (: 0 i32))))

(emit :jit (run-passes *module* (passes (canonicalize)
                                        (convert-scf-to-cf)
                                        (convert-to-llvm))))
```

Everything above except the reader and the kernel ops is library code.
`defn`, `print`, `scf.if`-with-results, `fn-type`, `passes`, `emit` are all
macros/functions written in the language.

---

## 9. Bootstrapping & implementation strategy

**Recommendation: evolve `lispier`'s Rust core rather than greenfield.** It
already has the reader, `melior` bindings, JIT, and a working pass story. We
re-architect its middle, not its edges.

### Kernel (Rust, small and frozen)

The compiler ships a minimal kernel that cannot be written in itself:

- Reader (`tokenizer.rs`/`reader.rs` — keep).
- The `Val` type incl. MLIR reflected handles (extend `value.rs`).
- A **meta-evaluator**: a tree-walking interpreter over `Val` with closures,
  `let`, `if`, `quote`/`quasiquote`, and primitives that wrap the MLIR C API
  (build op, infer types, parse type/attr, intern, run passes, verify, jit).
  This is the single thing that replaces both the Rust-builtin macros *and* the
  JIT-macro FFI path.
- `op` / `block` / `region` builders calling `melior` (refactor `ir_gen.rs` into
  primitives the evaluator calls, instead of a separate tree-walk).

Everything else — `defn`, `if`/`when`/`cond`, arithmetic sugar, `print`,
`defstruct`, `defdialect`, `passes`, the standard prelude — moves **out** of
`src/macros/builtins/*` and into `.coil` library files written as macros.

### Why interpret macros instead of JIT them?

`lispier` JIT-compiled macros because there was no interpreter. Once a small
meta-evaluator exists, compile-time macro execution is just interpretation:
no FFI, no `Box::from_raw`, no source-scraping, instant, debuggable. JIT is
reserved for *runtime* code (the actual program), where it belongs. If a macro
is genuinely hot, an `eval-when` block can opt into JIT — but that's an
optimization, not the mechanism.

### Migration path

1. Add the `Val` MLIR cases + meta-evaluator alongside the existing pipeline.
2. Re-express `op`/`block`/`region`/`module` as evaluator primitives over the
   existing `melior` calls in `ir_gen.rs`.
3. Port one builtin at a time to a library macro; delete the Rust version once
   the library version passes the same example (the `examples/` corpus is the
   regression suite).
4. Replace `require-macros`/JIT-macro with `require-for-syntax` + interpreted
   macros; delete `jit_macro.rs`, `macro_compiler.rs`, and `DynamicMacroContext`.
5. Land `defdialect`/`defpattern` codegen (finish `mlir-lisp`'s Phase 3).
6. Optional stretch goal: self-host the reader and evaluator in the language.

---

## 10. What this buys us (scorecard against the two problems)

| Old pain | Redesign |
|---|---|
| Two macro systems (Rust builtins + JIT FFI) | One: interpreted compile-time functions over shared `Val` |
| Macros can't see types/SSA/context | Compile-time world has a live `Context`; macros build & query IR |
| `*const Value -> *mut Value` FFI, `Box::from_raw` | No FFI boundary; same value model both phases |
| `DynamicMacroContext` source-scraping | Normal lexical environment; `require-for-syntax` |
| No hygiene, no macro-defining-macros | Hygienic by default; macro towers are free |
| Type inference special-cased in 2 places | `InferTypeOpInterface` queried at build time |
| `StringCollector`, `llvm.call` seg-sizes, struct GEP hard-coded | Library functions over the first-class module/builder |
| Known types/ops baked into parser | `!…`/`#…` parse + `(op "any.thing")` cover everything |
| `require` vs `require-macros` split | One `require` + explicit phase annotations |
| New dialect ⇒ recompile host | `defdialect` (IRDL) + `defpattern` (PDL) at runtime |

---

## 11. Open questions / risks

1. **Compile-time MLIR context lifetime.** Macros build trial IR in a context;
   we must define whether that's the same context as the final module (cheaper,
   but mutation hazards) or a scratch context (safer, needs op cloning across
   contexts). *Leaning: same context, with a builder-scope discipline.*
2. **Hygiene × SSA.** SSA values are inherently hygienic, but symbol references
   (`@foo`), block labels (`^bb`), and dialect/op names are namespaced strings —
   hygiene rules for *those* need spelling out.
3. **Error mapping.** MLIR diagnostics must carry source spans back through macro
   expansion (Racket-style syntax objects with source location) so verifier
   errors point at user code, not expanded gunk. This is the main UX risk.
4. **`eval-when` and effects.** Compile-time code that mutates the module
   (string interning, dialect registration) needs a defined ordering and
   idempotency story.
5. **Self-hosting boundary.** How small can the frozen Rust kernel be before the
   marginal op isn't worth moving into the language? Pick a deliberate line.
6. **Performance of interpreted macros** on large programs (GPT-2-scale). Likely
   fine, but measure; `eval-when`-JIT is the escape valve.

---

## 12. Summary

The earlier projects treated MLIR as a *target* reached by a syntactic macro
pass plus a hard-coded emitter, which forced a two-headed macro system and
scattered the lisp→MLIR mapping across the parser, the expander, post-passes,
and `ir_gen`. This design treats **MLIR as the object language**: IR nodes are
first-class values, building IR is evaluation, and macros are ordinary
compile-time functions sharing that one world. That single move dissolves both
historical problems — one macro system, and a total, principled mapping where
MLIR does the inference, verification, and lowering it was built to do.
