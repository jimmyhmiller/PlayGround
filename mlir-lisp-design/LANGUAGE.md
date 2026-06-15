# coil — Comprehensive Language Design

> The authoritative, end-to-end design for **coil**, a low-level Lisp that
> exposes MLIR as a first-class object language and compiles **ahead of time** to
> native code (CPU and GPU). This document consolidates and extends the
> companion deep-dives:
>
> - `DESIGN.md` — rationale & thesis (why; what `lispier`/`mlir-lisp` got wrong)
> - `SPEC.md` — reader grammar & surface→generic-form desugaring
> - `KERNEL.md` — the frozen compiler core & MLIR primitive catalog
> - `AOT.md` — the AOT/staging execution model (supersedes ELABORATION's
>   interpreter framing)
> - `ELABORATION.md` — hygiene & anti-double-emit analysis (applies to staging)
> - `prelude.coil` — the standard library written in coil
>
> Where this document and a deep-dive disagree, **this document wins** and the
> deep-dive is to be updated.

## Table of contents

1. [Philosophy & non-goals](#1-philosophy--non-goals)
2. [The compilation model](#2-the-compilation-model)
3. [Lexical & syntactic surface](#3-lexical--syntactic-surface)
4. [The value universe & homoiconicity](#4-the-value-universe--homoiconicity)
5. [Types](#5-types)
6. [Memory & data model](#6-memory--data-model)
7. [Functions, closures & calling](#7-functions-closures--calling)
8. [Control flow & error handling](#8-control-flow--error-handling)
9. [Aggregates: structs, enums, unions, slices](#9-aggregates-structs-enums-unions-slices)
10. [Metaprogramming (staged macros)](#10-metaprogramming-staged-macros)
11. [First-class MLIR: dialects, patterns, passes](#11-first-class-mlir-dialects-patterns-passes)
12. [Modules, namespaces & separate compilation](#12-modules-namespaces--separate-compilation)
13. [Targets: CPU & GPU](#13-targets-cpu--gpu)
14. [Diagnostics](#14-diagnostics)
15. [The toolchain](#15-the-toolchain)
16. [Worked examples](#16-worked-examples)
17. [Grammar reference](#17-grammar-reference)
18. [Influences & open questions](#18-influences--open-questions)

---

## 1. Philosophy & non-goals

coil is a **structured assembler with a Lisp's metaprogramming**. It treats
MLIR not as a backend but as the object language: the things you manipulate —
types, attributes, SSA values, operations, regions, dialects, passes — are
first-class. Higher-level constructs (`defn`, `match`, `struct`, generics) are
*library* macros that emit MLIR, never privileged compiler features.

**Principles**

1. **No magic.** Every construct lowers to visible MLIR. You can always ask
   "what ops did this produce?"
2. **MLIR does MLIR's job.** Type inference (`InferTypeOpInterface`), verification
   (the verifier), lowering (passes), dominance (the builder) are exposed, never
   re-implemented in the language.
3. **AOT, native, no hidden runtime.** Programs become objects/executables. There
   is no garbage collector, no boxing, no implicit allocation, no interpreter in
   the deployed artifact.
4. **One metaprogramming model.** Macros are ordinary coil functions, *staged*
   (compiled then invoked by the compiler), sharing the compiler's `Val` ABI.
5. **The language is mostly a library.** A small frozen kernel; everything else
   is `prelude.coil` and user code.

**Non-goals**

- Not a high-level managed language. No GC, no green threads, no reflection at
  runtime (reflection is a *compile-time* facility).
- Not source-compatible with any existing Lisp. It borrows s-expressions and
  hygiene, not Clojure/CL semantics.
- Not a stable ABI across versions (yet). The C ABI *to* coil is stable per the
  `extern` contract; coil↔coil calling conventions may evolve.

---

## 2. The compilation model

coil is ahead-of-time. A compilation unit flows through fixed stages (AOT.md):

```
source ─read→ forms ─expand→ core forms ─emit→ MLIR module
       ─verify→ ─passes→ lowered module ─llvm→ LLVM IR ─codegen→ object
                                                              └→ link → exe/lib
```

- **read** (`reader.rs`): text → `Val` s-expressions (SPEC §1).
- **expand**: surface sugar + user macros → *core forms*. Pure `Val → Val`.
  User macros run here via **staging** (§10): the macro is itself compiled AOT
  and invoked by the driver — never tree-walk-interpreted.
- **emit** (`emit.rs`): a symbol-table-directed walk that *builds* the MLIR
  module through the `Backend` trait. This is ordinary codegen, not execution.
- **verify / passes**: MLIR's verifier and a user-controllable pass pipeline
  (canonicalize, your custom lowerings, `convert-*-to-llvm`, GPU lowering, …).
- **llvm / codegen / link**: standard LLVM object emission and system linking.

### The staging tower

Compile-time code (macros, `defdialect`, `eval-when :compile`) is **coil compiled
by coil**. The driver maintains a *host* compilation that produces a dynamically
loadable artifact for compile-time code, calls into it during `expand`, and
discards it. This is the Rust-proc-macro model:

```
phase 0  the frozen kernel (Rust): reader, Val, emit, the Backend, a bootstrap
         expander for the irreducible sugar
phase 1  prelude + user macros compiled AOT → host artifact
phase 2  the program expanded (calling phase-1 macros) → core forms → object
```

There is never a general interpreter of coil in the loop; "running a macro" means
"calling compiled native code that returns a `Val`".

### Compile-time vs run-time, made explicit

- `(eval-when :compile …)` runs during `expand` (phase 1/2 boundary).
- `(require-for-syntax m)` makes `m` available to macros (host artifact).
- `(require m)` is a runtime dependency (linked into the program).
- A binding is *available at runtime* unless it is `for-syntax`-only.

---

## 3. Lexical & syntactic surface

Fully specified in `SPEC.md §1`; summary here for self-containedness.

```clojure
; atoms
42  -7  0xFF  3.14  1e9        ; numbers (type comes from (: v t), not the reader)
"hi\n"  true  false  nil       ; string / bool / nil
foo  arith.addi  my/h  +  ->   ; symbols (dots & sigils are name characters)
:value                          ; keyword
@printf  ^bb1  %0               ; symbol-ref / block-ref / ssa-ref (sigil symbols)
!llvm.ptr   !llvm.struct<(i64,i64)>   ; type literal (parse escape hatch)
#llvm.linkage<internal>               ; attribute literal (parse escape hatch)

; collections
(a b c)   [1 2 3]   {:k v :k2 v2}     ; list / vector / map

; reader macros
'x  `x  ~x  ~@x                 ; quote / quasiquote / unquote / splicing
; comments
; line     #_ datum     #| block |#
```

**Operator position.** A list whose head symbol contains `.` is an **op-call**:
`(dialect.op {attrs}? operand* region*)`. This, plus the kernel `op` form, is the
*entire* surface→MLIR mapping (SPEC §4). Everything non-dotted is a macro/function
call.

---

## 4. The value universe & homoiconicity

There is one `Val` type (KERNEL §1) spanning **syntax**, **compile-time data**,
and **reflected MLIR**:

- syntax/data: `Unit Nil Bool Int Float Str Sym Keyword List Vec Map Fn Macro`
- reflected MLIR (handles over the C API): `Type Attr Value Op Block Region
  Dialect Module Pass Pipeline Context`

Two levels of homoiconicity:

1. **Code is data** (`List`/`Sym`/…) — macros transform it.
2. **IR is data** (`Op`/`Region`/…) — patterns/transforms rewrite it.

Macros bridge the two. Because the compiler and compile-time code share this one
`Val` representation, there is no marshalling boundary (the root cause of
`lispier`'s macro pain).

Note: `Fn`/`Macro`/MLIR cases exist only at **compile time**. The *runtime*
program has no `Val`; its values are native (ints, floats, pointers, structs).

---

## 5. Types

A coil **type expression evaluates (at compile time) to an `MlirType`**. There is
no separate type AST; `i32` is a prelude constant bound to an `MlirType`,
`(vector-type [4] f32)` is a function returning one, and `!any.mlir.type` is the
parse escape hatch. Typing is therefore *programmable*: a macro can compute a
type with ordinary code.

### 5.1 Built-in & constructed types

```clojure
i1 i8 i16 i32 i64  u8 u32 …  index            ; integers (signedness is a surface
f16 bf16 f32 f64                               ;   convention; MLIR ints are signless,
                                               ;   ops pick signed/unsigned variants)
(ptr T)            ; typed pointer  → !llvm.ptr (opaque) + a compile-time pointee tag
(array N T)        ; fixed array    → !llvm.array<N x T>
(vector-type [N] T); SIMD vector    → vector<NxT>
(memref-type s T)  ; memref         → memref<sxT>
(tensor-type s T)  ; tensor         → tensor<sxT>
(fn-type [A B] [R]); function type
(struct T…) (enum …) (union …)                ; aggregates (§9)
```

### 5.2 Typed values & inference

A bare integer literal is untyped until placed:

```clojure
(: 42 i32)      ; typed literal → arith.constant in operand position; IntegerAttr in attr position
(arith.addi a b); result type inferred from operands via InferTypeOpInterface
```

Inference is **MLIR's**, not coil's. A binding's type is whatever the producing
op's result type is. `defn` parameter and return types are explicit (they form
the `func.func` signature); locals are inferred.

### 5.3 Generics = compile-time parametricity (monomorphization)

coil has no runtime polymorphism. A "generic" is a macro/template parameterized
over **types** (which are compile-time values), instantiated per use — i.e. C++
templates / Zig `comptime` done with staged macros:

```clojure
(defn-generic [T] max [(: a T) (: b T)] -> T
  (if (arith.cmpi {:predicate sgt} a b) a b))

(max (: 3 i32) (: 7 i32))     ; instantiates max@i32
(max (: 1.0 f64) (: 2.0 f64)) ; instantiates max@f64
```

`defn-generic` is a prelude macro that records a template; each distinct type
argument set produces one specialized `func.func` (deduplicated). No vtables, no
boxing. Constraints (e.g. "T must be integer") are checked at instantiation via
`mlir/op-has-trait`/type predicates, with errors mapped to the call site.

### 5.4 Type aliases & predicates

```clojure
(deftype Byte u8)
(deftype Vec3 (struct [x f32] [y f32] [z f32]))
(type? x)            ; compile-time: is x an MlirType?
(integer-type? T)    ; predicate used by generic constraints
```

---

## 6. Memory & data model

coil is **manual-memory by default** — like C/Zig, not like a managed Lisp.

### 6.1 Pointers & allocation

- `(ptr T)` is a typed pointer over MLIR's opaque `!llvm.ptr`; the pointee type
  is tracked at compile time for GEP/load/store typing.
- Allocation is explicit and library-provided: `malloc`/`free` (via `extern`),
  stack via `llvm.alloca`, or arena allocators (below). No implicit heap use —
  closures, aggregates, and strings never silently allocate.

```clojure
(let [p (alloc i32)]        ; → llvm.alloca, stack
  (ptr-store! p (: 7 i32))
  (ptr-load p))             ; 7
```

### 6.2 Ownership stance

coil does **not** impose a borrow checker. Ownership is a *convention* with
opt-in library support:

- **Arenas/regions** (`with-arena`): bump-allocate, free en masse. The default
  recommended pattern for compilers/servers.
- **Refcounting** (`rc` library): `Rc<T>` as a struct with a count; `retain`/
  `release` macros emit the ops. Optional, never implicit.
- **Manual**: `malloc`/`free`.

A future optional **linear/affine checking pass** (a coil-defined MLIR analysis)
can enforce single-ownership where annotated — designed as a pass, not a language
core feature, true to principle 2.

### 6.3 Strings & data layout

- A string literal is a `(ptr i8)` to an interned, NUL-terminated global
  (`intern-cstring!`, prelude). `(slice i8)` (`{ptr,len}`) is the non-C string
  type for length-aware code.
- Aggregate layout is LLVM's, queried via the data layout (`mlir/data-layout`),
  so struct field offsets/alignment are *computed*, never hard-coded (fixing
  `lispier`'s `* 8` GEP hack).

### 6.4 No hidden control or allocation

Every allocation, copy, and indirect call is written by the programmer or a macro
the programmer invoked. This is the "low-level" contract.

---

## 7. Functions, closures & calling

### 7.1 Top-level functions

```clojure
(defn add [(: a i32) (: b i32)] -> i32
  (func.return (arith.addi a b)))           ; explicit terminator…
(defn add2 [(: a i32) (: b i32)] -> i32
  (arith.addi a b))                          ; …or implicit (last expr → func.return)
```

`defn` is a prelude macro emitting `func.func` with `:function_type` from the
signature (§ prelude.coil). Multiple return values are native MLIR multi-results:

```clojure
(defn divmod [(: a i32) (: b i32)] -> [i32 i32]
  (values (arith.divsi a b) (arith.remsi a b)))
(let [[q r] (divmod x y)] …)                 ; destructure multi-results
```

`extern` declares C functions; `:vararg true` for printf-style.

### 7.2 Closures are explicit

A "low-level" language must not hide environment allocation. A bare `fn` that
captures variables is an error unless you say how it is represented:

```clojure
(closure [captured-a captured-b]            ; explicit capture list
  [(: x i32)] -> i32
  (arith.addi x captured-a))
```

`closure` lowers to a **fat pointer**: a heap/arena/stack-allocated environment
struct `{captured-a, captured-b}` plus a function pointer taking
`(env-ptr, args…)`. Where the env lives is a parameter (`:in arena`, `:in stack`,
`:in heap`). Calling is `(call-closure c args…)`. There is no implicit boxing and
no GC keeping environments alive — lifetime is the programmer's (or the arena's).

### 7.3 Function pointers & indirect calls

`(fn-ptr add)` yields a `(ptr (fn-type …))`; `(call-indirect p args…)` emits
`llvm.call` on it. Dispatch tables, vtables, and interfaces are *built from these*
in libraries — not language primitives.

---

## 8. Control flow & error handling

### 8.1 Structured control flow → `scf`/`cf`

High-level control forms are prelude macros over the `scf` (structured) and `cf`
(unstructured) dialects:

```clojure
(if {:result i32} cond  then-expr  else-expr)      ; → scf.if with results
(when cond  body…)                                  ; → scf.if, no else
(cond  c1 e1  c2 e2  :else e3)                       ; → nested scf.if
(case  x  1 a  2 b  _ default)                       ; → cf switch / scf
(while cond  body…)                                  ; → scf.while
(for [i (range n)]  body…)                           ; → scf.for
(loop [acc (: 0 i32)  i (: 0 i32)]                   ; → scf.while/for with iter_args
  (if (i< i n) (recur (i+ acc i) (i+ i (: 1 i32))) acc))
```

`recur` maps to `scf` iteration arguments; `break`/`continue` to the relevant
terminators. Because these are macros, you can add your own control constructs.

### 8.2 Early return & labeled blocks

`(return v)` inside a function body short-circuits via `cf.br` to an exit block
(macro-managed). Labeled blocks (`^bb`) and explicit `cf.br`/`cf.cond_br` are
always available for hand-rolled CFGs.

### 8.3 Errors are values (no exceptions by default)

coil's default error model is **sum types**, not unwinding:

```clojure
(deftype (Result T E) (enum [ok T] [err E]))
(deftype (Option T)   (enum [some T] [none]))

(defn parse [(: s (slice i8))] -> (Result i32 ParseError) …)

(let [x (try (parse s))]   ; `try`/`?` propagates err: on err, return it from the fn
  (use x))
```

`try`/`?` is a macro that `match`es the `Result` and early-returns on `err`.
`match` (§9.2) does the discriminant switch. This keeps control flow visible and
costs nothing when there is no error.

**Optional unwinding.** For C++/`setjmp` interop, an opt-in library exposes LLVM
landing pads / `setjmp`-`longjmp`; it is never on by default and never implicit.

---

## 9. Aggregates: structs, enums, unions, slices

### 9.1 Structs

```clojure
(defstruct Point [x f32] [y f32])
(let [p (struct-new Point :x (: 1.0 f32) :y (: 2.0 f32))]
  (Point/x p)                ; field read   → llvm.getelementptr + llvm.load
  (Point/x! p (: 3.0 f32)))  ; field write  → getelementptr + llvm.store
```

`defstruct` (prelude) builds the `!llvm.struct<…>` type and generates hygienic
accessor macros; **field indices and offsets come from the data layout** (§6.3).
Structs are value types; `struct-new` allocates where you say (`:in stack/arena/
heap`), defaulting to stack.

### 9.2 Enums / tagged unions & `match`

```clojure
(deftype Shape
  (enum [circle f32]                 ; radius
        [rect   (struct [w f32] [h f32])]
        [point]))                    ; no payload

(defn area [(: s Shape)] -> f32
  (match s
    [circle r]      (arith.mulf pi (arith.mulf r r))
    [rect {:w w :h h}] (arith.mulf w h)
    [point]         (: 0.0 f32)))
```

An `enum` lowers to `{ i32 tag, [N x i8] payload }` (size = max variant, aligned).
`match` emits a switch on the tag plus per-arm payload reinterpretation; it is
exhaustiveness-checked at compile time (a macro error names missing variants).
This is the canonical first-class-MLIR-meets-ergonomics example: a pleasant
surface, no runtime type info, all lowering visible.

### 9.3 Unions & slices

- `(union …)` — untagged overlap, for FFI/bit tricks.
- `(slice T)` — `{ (ptr T), index len }`; `(slice-get s i)`, `(slice-len s)`,
  bounds-checking optional behind `:checked`.

---

## 10. Metaprogramming (staged macros)

The single metaprogramming mechanism. A macro is a coil function, marked to run
at compile time, **compiled and invoked by the driver** (AOT.md staging), taking
unevaluated argument forms and returning a replacement form.

### 10.1 Defining & using macros

```clojure
(defmacro unless [cond & body] `(if (not ~cond) (do ~@body)))
```

- Arguments arrive unevaluated (forms). Result is re-expanded to fixpoint.
- Quasiquote (`` ` ~ ~@ ``) templates over `Val`, composing with MLIR cases.
- **Macros may build & inspect IR** during expansion via the compiler API
  (`mlir/infer-results`, `mlir/value-type`, `build`, `with-scratch`) — the thing
  `lispier` could not do. The **anti-double-emit rule** (ELABORATION §2) governs:
  if you `build`, splice the resulting `Value`, never the original form.

### 10.2 Hygiene

Scope-set hygiene (ELABORATION §5): template-introduced identifiers get a fresh
scope and can't capture caller identifiers; `gensym`, `unhygienic`,
`datum->syntax` are the escape hatches. SSA values are inherently hygienic
(opaque handles, not names).

### 10.3 Reader macros

`(defreader \$ (fn [reader] …))` registers a reader extension producing `Val`s,
for DSLs that need custom surface syntax before s-expr reading. Used sparingly;
most extension is ordinary macros.

### 10.4 Compile-time computation

`(eval-when :compile …)` and macro bodies run arbitrary coil — but as *staged*
native code, with access to the filesystem, the module under construction, and
the dialect registry. This is how `intern-cstring!`, `defdialect`, and generic
instantiation do real work at compile time without an interpreter.

### 10.5 Generics via macros

§5.3: `defn-generic`/`deftype-generic` are macros that monomorphize. Type
arguments are compile-time `MlirType` values; specialization is deduplicated by
the (name, type-args) key.

---

## 11. First-class MLIR: dialects, patterns, passes

This is the headline capability: extend the compiler **from within the language**,
at compile time, no host recompile.

### 11.1 Defining dialects (IRDL)

```clojure
(defdialect my
  (deftype my.complex :params [(elt AnyFloat)])
  (defattr my.unit-kind :params [(name StrAttr)])
  (defop  my.add
    :operands [(lhs AnyFloat) (rhs AnyFloat)]
    :results  [(out AnyFloat)]
    :traits   [Pure Commutative SameOperandsAndResultType]
    :verify   (same-type lhs rhs out)))
```

`defdialect` emits `irdl.*` ops and registers the dialect with the live context at
compile time. Thereafter `(my.add a b)` builds a real, verifying `my.add`.

### 11.2 Rewrite patterns (PDL)

```clojure
(defpattern lower-my-add
  :benefit 2
  :match   (my.add ?lhs ?rhs -> ?ty)
  :rewrite (arith.addf ?lhs ?rhs -> ?ty))
```

`?x` are PDL variables. Patterns are `Val`s — you can **generate** them (e.g. a
macro that derives a lowering for every op in a dialect). Apply via the pattern
rewriter or schedule with the transform dialect.

### 11.3 Pass pipelines as values

```clojure
(def to-llvm
  (passes (canonicalize)
          (cse)
          (apply-patterns lower-my-add)
          (convert-scf-to-cf)
          (convert-to-llvm)))

(compile-with *module* to-llvm)
```

Pipelines compose, splice, and print. Drop to a textual pipeline string for
anything exotic (`(pass-pipeline "builtin.module(...)")`).

### 11.4 The transform dialect

For scheduling rewrites (tiling, fusion, vectorization) over named handles,
exposed as `(transform …)` forms — first-class, generatable, the same way.

---

## 12. Modules, namespaces & separate compilation

- **File = module.** A module's top-level forms are its definitions.
- **Namespaces.** Symbols may be namespaced (`my/helper`); `require` with an
  alias controls resolution. Op-call heads are dialect-qualified (`arith.addi`)
  and resolved against loaded dialects, not the symbol namespace.
- **Visibility.** `def`/`defn` are module-private unless `(export …)`ed;
  `func.func`'s `sym_visibility` follows.
- **Phasing.** `(require m)` (runtime) vs `(require-for-syntax m)` (compile-time)
  vs `(require-dialect d)` (loads an MLIR dialect at compile time). One `require`,
  phase annotations where needed — unifying `lispier`'s `require`/`require-macros`
  split (KERNEL §6).
- **Separate compilation.** Each module compiles to its own MLIR module → object;
  the system linker combines them. Cross-module inlining is a pass over a merged
  module when whole-program optimization is requested.
- **ABI.** `extern`/`export` define the stable C ABI surface; internal coil↔coil
  calling conventions are not yet frozen.

---

## 13. Targets: CPU & GPU

coil targets anything MLIR can lower, with CPU and GPU first-class (the `lispier`
lineage shipped GPT-2 on an AMD GPU).

- **CPU:** lower to LLVM, emit objects, link. Vectorization via the `vector`
  dialect; SIMD types are ordinary `vector-type`s.
- **GPU:** write kernels in the `gpu` dialect; lower through `gpu-to-rocdl`
  (ROCm) or `gpu-to-nvvm` (CUDA). Host/device split, launch configs, and memory
  transfers are explicit ops, wrapped by prelude macros (`gpu-launch`,
  `gpu-alloc`). Same language, same metaprogramming — a macro can generate a
  family of kernels.
- **Pipeline selection** is a value (§11.3); a build profile picks CPU vs GPU
  lowering. Cross-compilation = choosing the LLVM target triple in the driver.

---

## 14. Diagnostics

The make-or-break UX concern, designed in (KERNEL §7):

1. The reader attaches a `Span` (file, byte range) to every node.
2. Expansion propagates spans (Racket-style `syntax/loc`): introduced nodes
   inherit the macro-call span; passed-through nodes keep their own.
3. `emit` stamps each built `Operation` with an MLIR `Location` derived from the
   span (`FileLineColLoc`, fused for macro-expanded code).
4. Verifier/pass failures carry that `Location`; the driver maps it back to the
   span and renders a **source-pointing** error — never raw expanded IR.

Macro errors (arity, exhaustiveness, generic constraints) likewise point at the
call site. `coil expand` shows expansion steps for debugging.

---

## 15. The toolchain

A single `coil` driver:

```
coil build  <file>        compile a module/program to an object/executable
coil run    <file>        build to a temp artifact and execute it (AOT, then run)
coil emit   <file> --ir   dump MLIR (or --llvm / --asm / --obj)
coil expand <file>        show macro expansion (one step or full)
coil check  <file>        read + expand + verify, no codegen
coil repl                 compile-and-run loop (each entry is built AOT, then run)
coil fmt    <file>        canonical formatter
```

- **`run`/`repl` are still AOT**: they compile to native and execute the result;
  there is no interpreter. The REPL is a fast compile+`dlopen`+call loop.
- **Incremental builds** via a content-addressed cache keyed on (source hash,
  macro-host hash, pipeline). Compile-time host artifacts are cached too.
- **Debug info**: LLVM DWARF from the source `Location`s, so native debuggers and
  profilers see coil source.
- **LSP / editor**: served from the same reader/expander; hovering shows the
  expanded ops and inferred MLIR types.

Implementation: a frozen Rust **kernel** (reader, `Val`, emit, the `Backend`,
bootstrap expander) + `melior`-backed `MlirBackend` behind the `mlir` feature +
the prelude/stdlib in coil. Self-hosting the reader/expander is a stretch goal,
not a requirement.

---

## 16. Worked examples

### 16.1 The whole pipeline, by hand and by sugar

```clojure
(require-dialect arith func)

; high-level (sugar)
(defn add [(: a i32) (: b i32)] -> i32
  (arith.addi a b))

; exactly what it expands to (core)
(op "func.func"
    :attrs {:sym_name "add" :function_type (fn-type [i32 i32] [i32])}
    :regions [(region (block ^entry [(: a i32) (: b i32)]
                 (op "func.return" :operands [(arith.addi a b)] :results [])))])
```

### 16.2 Recursive fib with structured control flow

```clojure
(defn fib [(: n i32)] -> i32
  (if {:result i32} (i< n (: 2 i32))
    n
    (i+ (fib (i- n (: 1 i32))) (fib (i- n (: 2 i32))))))
```

### 16.3 A custom dialect + its lowering, in the language

```clojure
(defdialect my (defop my.fma :operands [(a F) (b F) (c F)] :results [(o F)]
                       :traits [Pure] :verify (all-same a b c o)))
(defpattern lower-fma :match (my.fma ?a ?b ?c -> ?t)
                      :rewrite (math.fma ?a ?b ?c -> ?t))
(defn dot3 [(: a Vec3) (: b Vec3)] -> f32
  (my.fma (Vec3/x a) (Vec3/x b)
    (my.fma (Vec3/y a) (Vec3/y b)
      (arith.mulf (Vec3/z a) (Vec3/z b)))))
(compile-with *module* (passes (apply-patterns lower-fma) (convert-to-llvm)))
```

### 16.4 Result-typed error handling

```clojure
(defn safe-div [(: a i32) (: b i32)] -> (Result i32 DivErr)
  (if (i= b (: 0 i32)) (err DivErr/divide-by-zero) (ok (arith.divsi a b))))

(defn run [] -> i32
  (let [q (try (safe-div x y))]   ; propagates DivErr out of `run`
    q))
```

### 16.5 A generated family of GPU kernels

```clojure
(eval-when :compile
  (for-each [elt [f16 f32 f64]]
    (emit-def `(defn ~(sym (str "axpy_" (type->str elt)))
                 [(: a ~elt) (: x (ptr ~elt)) (: y (ptr ~elt)) (: n index)] -> ()
                 (gpu-launch [i (: n)]
                   (ptr-store! (ptr-index y i)
                     (arith.addf (arith.mulf a (ptr-load (ptr-index x i)))
                                 (ptr-load (ptr-index y i)))))))))
```

One macro, three monomorphized kernels, lowered through the GPU pipeline.

---

## 17. Grammar reference

Reader grammar is in `SPEC.md §1` (EBNF). Core-form grammar (post-expansion, what
`emit` consumes):

```
program     = top-form* ;
top-form    = op-form | def | require | eval-when ;

op-form     = "(" "op" string op-arg* ")"            ; explicit
            | "(" dotted-sym attr-map? operand* region* ")" ;  ; terse
op-arg      = ":operands" vector | ":results" (":infer" | type-vector)
            | ":attrs" map | ":regions" vector | ":successors" vector ;

region      = "(" "region" (block+ | core-form*) ")" ;
block       = "(" "block" block-ref param-vector core-form* ")" ;
param       = "(" ":" sym type ")" ;

core-form   = op-form | let | do | sym | literal ;
let         = "(" "let" "[" (sym core-form)* "]" core-form* ")" ;
do          = "(" "do" core-form* ")" ;

type        = type-lit | type-sym | "(" type-ctor type-arg* ")" ;
```

Surface (pre-expansion) adds `defn`, `defmacro`, `defstruct`, `deftype`,
`defdialect`, `defpattern`, control-flow macros, `match`, etc. — all of which
expand into the core grammar above.

---

## 18. Influences & open questions

**Influences.** Terra (staged metaprogramming over a low-level target),
Zig (`comptime`, explicit allocation, no hidden control flow), Rust (proc-macros,
sum types, `?`), Racket (`#lang`, scope-set hygiene, syntax objects),
Clojure (s-expr surface, `require`), Carp (Lisp + manual memory), and of course
MLIR/LLVM (the object language) and the prior `lispier`/`mlir-lisp` experiments
(what to keep, what to discard).

**Open questions.**

1. **Compile-time context lifetime** — scratch vs shared MLIR context for macro
   building (ELABORATION §11.1). Leaning shared, with builder-scope discipline.
2. **coil↔coil ABI** — when to freeze a stable internal calling convention.
3. **Ownership enforcement** — keep it library/convention, or ship the optional
   linear-checking pass in v1?
4. **Self-hosting** — how much of the kernel migrates into coil, and when.
5. **Effect ordering of compile-time mutation** — `intern-cstring!`/dialect
   registration ordering & idempotency across modules (ELABORATION §4).
6. **Diagnostic fidelity through deep macro expansion** — the main UX risk;
   needs real programs to validate the span-propagation model.
7. **Package/build system** — module resolution, versioning, and the
   content-addressed cache's invalidation rules.

---

*This is a living design. The implementation (`../coil/`) currently realizes the
reader, the `Val` model, the printer, the `Backend` codegen boundary, and the
core-form→MLIR `emit` mapping; the expander, the `melior` backend, and staged
macros are the next milestones (see `../coil/README.md`).*
