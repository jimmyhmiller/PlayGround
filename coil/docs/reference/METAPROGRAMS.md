# Metaprograms in Coil

A **metaprogram** is a Coil function that runs at compile time and operates on the
program. There is no separate metalanguage — it is Coil operating on Coil, compiled
to native code and run, its output type-checked and compiled normally.

## The four kinds

Distinguished only by *what they receive* and *what they return*.

| kind | signature | receives | returns | status |
|---|---|---|---|---|
| **Macro** | `[Code…] -> Code` | its own call site | replacement code | shipped |
| **Generator** | `() -> Code` (via `meta`) | nothing | new top-level forms | shipped |
| **Checker** | `Modules -> Code` | all modules, grouped | veto/report | **shipped** |
| **Transformer** | `Program -> Program` | the whole program | a rewritten program | **shipped** |

## Applying a metaprogram

- **Macros** — *call* them: `(when c body)`. Detected by their `Code` signature.
- **Checkers / transformers** — *register* them at top level: `(checker lint-icmp)` /
  `(transform desugar-inc)`. The compiler runs them during compilation. A **checker is
  handed the program as a list of modules** — `((name form…) …)`, one record per module
  (head = module name symbol, rest = its top-level forms). The checker owns the loop and
  decides which modules to look at (e.g. skip the ones where `(primitive/code-from-user? (primitive/code-nth m 1))`
  is false). A **transformer** still gets the flat form list (it rewrites in place).
- **Dialects** — *import* a module that contains those registrations: one
  `(import "safe_dialect.coil")` applies its whole stack.
- **From the CLI, optionally** — `coil run app.coil --use lint.coil` imports a
  metaprogram module (which self-registers its `(checker …)`) **without editing the
  source**. Repeatable; works on `run` and `build`. This is how you run a linter on
  demand: `coil run app.coil --use lint-on.coil`.

All four share one substrate: the `Code` value, the `code-*` operations, type
reflection, and the compiled comptime engine. The difference is **scope** (my-call-site
vs whole-program) and **power** (produce vs reject vs rewrite). An ordered stack of
checkers/transformers is what we mean by a **dialect** (e.g. the GC dialect, a
Rust-like ownership dialect, a Scheme frontend).

## The API (the vocabulary), all shipped

- **Take Code apart:** `code-count`, `code-nth`, `code-rest`, `code-sym`,
  `code-list?`, `code-sym?`, `code-int?`, `code-keyword?`, `code-str`, `code-eq`.
- **Build Code:** quote `` ` ``, unquote `~`, splice `~@`, `code-symbol` (make a
  name from a symbol/string base plus suffix parts — an int part becomes its decimal
  digits, and a **generic instantiation** base `(Gen A B)`, the same shape
  `code-field-count` accepts, is mangled deterministically (`(Box i64)` → `Box__i64`)
  so `derive` can name a helper over a generic instance), `code-str` (same, yielding a
  string), `gensym` (fresh hygienic name).
- **Bytes ↔ names:** `(primitive/int->str N)` → the integer's decimal as a `(slice u8)`, so a
  name minted from a counter/index is one call; `(primitive/str-bytes S)` → a Code list of a
  string's byte values; `(primitive/bytes->str LIST)` → the inverse (list of ints → string).
- **Reflect on types:** `code-field-count/name/kind/type`, `code-variant-*`,
  `code-trait-*` (and value-form `field-count`, `struct?`, `field-name`, …).
  Kind tags: `0=int 1=float 2=bool 3=struct 4=sum 5=ptr 6=array 7=slice 8=other`.
- **Compute:** the compiled engine runs arbitrary Coil — including generics,
  collections, allocation and FFI.
- **Fail / branch:** `error` (abort expansion with a message), `target-arch`,
  `target-os`.

`(primitive/target-arch)` and `(primitive/target-os)` are nullary code ops that answer for the
platform being **compiled for** — read from the resolved `--target` triple, not
probed from the machine running the compiler. `target-arch` yields
`aarch64`/`x86_64`/`wasm32`/`wasm64`; `target-os` yields `linux`/`darwin`/`wasm`.
Both fall back to the host when no `--target` is given.

That distinction is the whole point, and it is easy to get wrong: a platform
constant folded at compile time from a *runtime* probe silently records the
build host, so cross-compiling bakes in the wrong answer. `src/stdlib/fs.coil` is the
worked example — its `open()` flags differ per OS (512 is `O_CREAT` on darwin
and `O_TRUNC` on Linux), and it selects them with

```coil
(defn os-pick [(linux Code) (darwin Code)] (-> Code)
  (if (primitive/code-eq (primitive/target-os) `linux) linux darwin))
(defn gen-open-flags [] (-> Code)
  `(do (const O_CREAT ~(os-pick `64 `512))
       (const O_TRUNC ~(os-pick `512 `1024))))
(meta (gen-open-flags))
```

so the flags stay true compile-time literals (no runtime branch at any use) and
are still right under cross-compilation. `scripts/compiler/oracle/gate-target-os.sh`
pins this by diffing `emit-ir` across `--target` values, which is the only place
the property is observable — a cross-built binary never runs on the builder.

New API this project added (all shipped):

- **`(primitive/report NODE MSG)`** — a located compile-time **error** at `NODE`. It **collects**:
  a checker keeps running and surfaces EVERY error in one pass; the build fails after
  printing them all (see `tests/metaprogramming/located_multi.coil` → 2 errors, then failure).
- **`(primitive/warn NODE MSG)`** — a located, **non-fatal** warning at `NODE`; collects and all
  print, the build succeeds. What a *linter* wants — `tests/metaprogramming/lint.coil` warns at
  every `icmp-*`, suggesting `< > = …`.
- **`(primitive/suggest NODE MSG REPLACEMENT)`** — a `warn` that also proposes a **rewrite**.
  `REPLACEMENT` is a `Code` value, normally built from the author's own subnodes; the
  diagnostic gains a `help: try: …` line, and `coil lint --fix` (and only it) splices it
  into the file. Any node that came from source is written back as its ORIGINAL bytes,
  so comments and formatting in untouched branches survive. A round that stops compiling
  is reverted. Comments BETWEEN nodes — the one thing no `Code` value records — are
  carried across the rewrite too, so a commented chain is fixed like any other. Demo:
  `src/examples/metaprogramming/condlint.coil` rewrites a chain of 3+ nested `if`s as a `cond` with
  `:else`. Full design + what changed on contact with reality: `docs/archive/AUTOFIX.md`.
- **`(primitive/code-macro? NODE)` → bool** — true for a node the expander produced. Checkers run
  on the EXPANDED program, so every `cond`/`when`/`case` the author wrote is already
  nested `if`s by the time a rule sees it; this is how a rule about `if` tells the two
  apart. It is the same `ctxt ≠ 0` test autofix uses internally to refuse to edit
  macro-generated code.
- **Metaprograms are fed the WHOLE program, including all imports** (their own modules
  and bundled stdlib). A checker sees imported code too — `tests/metaprogramming/imports_test.coil`
  shows the linter flag an `icmp` in an imported user module.
- **Checkers run AFTER resolve + typecheck** (the *semantic* layer; see
  `docs/design/SEMANTIC_METAPROGRAMS.md`). A checker is registered at `expand-stage3` but
  executed later, once the whole program is checked, so it reads the compiler's
  authoritative output. A checker therefore layers *policy* on a program that already
  typechecks.
- **Transforms are MODULE-SHAPED and may ADD/REMOVE top-level forms.** `(transform
  FN)` hands FN the program as `((name form…) …)` (one record per module, like a
  checker) and FN returns the same shape; every form in a returned module record is
  tagged with that module, so a transform may EMIT new top-level defns (a GC dialect's
  per-type `trace-T`, a root table, a runtime import) or drop forms. Demo:
  `tests/metaprogramming/compile-and-run/addforms.coil` emits a whole new defn.
- **`(primitive/binding-of NODE)` → the local-binding identity** a reference resolves to (an
  i64; 0 = a global const/function), recorded by the type-checker per reference.
  Two references with the same positive id name the SAME local, so a checker
  distinguishes a **shadowed** local from its outer namesake — which name-matching
  cannot. This is what a borrow/move checker keys its dataflow on. Demo:
  `tests/metaprogramming/compile-and-run/borrowlike.coil` (a use-after-free checker).
- **Generic reflection.** `code-field-type`/`code-field-kind` accept a type
  **instantiation** `(Gen A B)`, not only a bare name, and substitute the type
  parameters — `code-field-type (Pair i64 (ptr u8)) 1` → `(ptr u8)`. So a derive/
  trace generator sees concrete field types through a generic. Demo:
  `tests/metaprogramming/compile-and-run/genrefl_test.coil`.
- **`(transform FN)`** — there is ONE kind of transform, and it is semantic. It runs
  to a fixpoint: each round it reads the checked program (via `code-decl` etc.) to
  decide its rewrite, then the pipeline re-resolves + re-typechecks. It also TOLERATES
  a program that doesn't yet typecheck — then the model is empty (`code-decl` →
  `:unresolved`) and the transform rewrites purely syntactically until the program
  becomes valid (e.g. `inc`→`iadd`, where `inc` is undefined until the rewrite). The
  authoritative strict check happens once, after the fixpoint. So one primitive covers
  both type-aware rewrites and syntactic desugarings. Demos: `tests/metaprogramming/retkind*.coil`
  (rewrites a marker by the wrapped call's real return type) and `dialect.coil`/`tx_test`
  (`inc`→`iadd`).
- **`(primitive/code-decl NODE)` → a declaration record**, read from that authoritative checked
  program. `(decl MODULE fn [PARAM-TYPE…] RET)` for a function, `(decl MODULE KIND)` for
  a struct/sum/trait/const/extern, `:unresolved`, or `:ambiguous`. **Pass a resolved
  REFERENCE node and it resolves to the EXACT entity** the checker picked (via node
  identity), unambiguous even when the simple name lives in several modules. This covers
  every resolved reference: **function calls, function-pointer refs (`fnptr-of`), and
  variant constructions** (which resolve to the owning sum). A bare symbol falls back to
  a name-based lookup (which reports `:ambiguous` on a cross-module name clash). Demos:
  `dup_app.coil` (two modules both defining `probe`; each call resolves to the right one),
  `refpolicy_bad.coil` (a `fnptr-of` to a pointer-returning function, never called),
  `variantcheck_test.coil` (a `(Jus 5)` construction resolves to its sum), and
  `typecheck_test.coil` (a **type reference** `wb/Box` resolves to the right module even
  though `Box` is defined in both `wa` and `wb`). So calls, fn-ptrs, variants, AND named
  types all resolve exactly.
- **`(primitive/type-of NODE)` → the expression's inferred type** as `Code` (e.g. `i64`,
  `(ptr i64)`), or `:unknown`. This is the type the real type-checker inferred, not
  syntax — a call `(getf)` reports `f64` because `getf` returns `f64`. Demo:
  `tests/metaprogramming/nofloat*.coil` bans floating-point-typed expressions.
- **`(primitive/code-file NODE)` → the source file name** of a node, and **`(primitive/code-from-user? NODE)`
  → bool** (true for a real file, false for a bundled `<…>` source). So a linter can
  *scope itself* — `tests/metaprogramming/lint.coil` warns only where `(primitive/code-from-user? f)`, which
  skips the standard library while still linting the user's own modules. (Checkers can't
  call imported string functions — the closure doesn't include them — so `code-from-user?`
  does the check in the compiler and hands the checker a bool.)
- **`(checker FN)` / `(transform FN)`** — register a whole-program metaprogram.
- **A dialect is a single import.** A module that contains `(checker …)`/`(transform …)`
  registrations *is* a dialect — importing it applies the whole stack (import order =
  pass order; transformers run before checkers). No new syntax; the module is the
  manifest. See `tests/metaprogramming/safe_dialect.coil`.

## The engine: everything compiles

There is **one engine**. `expand-stage3` lowers the metaprogram sub-program to a normal
program (`metalower.coil`: `Code` -> opaque handle, code ops -> boundary calls),
compiles it with the ordinary pipeline (cached content-addressed under
`~/.cache/coil/metaprog`), and runs each entry as **native code**. Everything the
language can do, a metaprogram can do — generics, HashMap, `malloc`, libc FFI at
expansion time (`tests/metaprogramming/compile-and-run/arbitrary.coil`).

This covers **every** compile-time call site with one semantics: macros, `(meta …)`
generators, checkers, transforms, and `(comptime E)` / `(const …)` folding. So a helper
behaves the same whether it is reached from a macro, a generator, or a `const`.

How the metaprogram object is produced depends on the host, invisibly: the LLVM build
uses the LLVM backend, the LLVM-free `main_a64` compiler uses the arm64 backend's
`export-c`, and inside a wasm sandbox the bytecode interpreter (`interp.coil`, see
`docs/reference/BYTECODE_INTERP.md`) runs the same mono'd sub-program in-process. Same semantics,
three ways of getting native (or emulated) execution.

**Macro bodies can call macros** (the TOWER): `when`/`cond`/`try!`/`fmt` inside a
metaprogram's own body expand at definition time, type-directedly — a call to a
Code-signature function whose arguments all typecheck as Code stays a FUNCTION
call (passing code values, e.g. cond-arms' recursion); one with non-Code
arguments is surface syntax and is expanded, with the checker as the only judge.
Staging is lazy and per-qual: an engine miss stages that macro on demand, and a macro
that cannot be staged is a hard error rather than a silent fallback.

Known limits: entries are capped at 8 parameters; a comptime **result** must be
materializable as a literal (a pointer, or an aggregate that is a generic instance,
is a located error); and deep self-recursion at comptime is not TCO'd, so a runaway
comptime recursion crashes rather than reporting.

## What metaprograms can and can't do

Can: read any form's syntax; **see the whole program**, including imports and bundled
stdlib; generate code (inline + new top-level defs); **rewrite code they don't own**
(transforms, to a fixpoint, adding or removing top-level forms); **reject precisely**
— `warn`/`report` are located, collect, and surface every diagnostic in one pass;
read the compiler's authoritative model (`code-decl`, `type-of`, `binding-of`);
compute at compile time with the whole language; reflect on struct fields, sum
variants and trait method signatures; compose to a fixpoint; abort the build; branch
on target.

The first three walls in this doc's original framing are **gone** — that was the point
of the project. What remains:

1. **Intercept core special forms.** `if`, `store!`, `index`, `alloc-stack` are parsed
   as special forms, so a macro named `store!` cannot shadow them. Demoting `store!`
   to an interceptable call over a `%store!` primitive is the open design (below); it
   is the one change that would re-bless the oracle snapshots.
2. **Function-signature reflection** for auto-coercion at dialect boundaries.
3. **A comptime result that is a pointer or a generic-instance aggregate**, and
   **runaway comptime recursion**, which is unbounded and crashes rather than
   reporting.

## Still open

### Core-form demotion
**Demote `store!` first** — the parser would emit an interceptable `(primitive/store! …)` call
over a `%store!` primitive so transforms can rewrite it on idiomatic code. This is the
one change that re-blesses the oracle snapshots. Unlocks write barriers and
bounds-checking on unmodified Coil.

### Richer reflection
Function-signature reflection, for auto-coercion at dialect boundaries.

## Related work in the repo
- `src/experiments/gc-dialect/` — a precise mark-sweep GC built entirely as macros + a runtime
  library (implicit allocation + reclamation, reflection-generated tracers).
- `src/experiments/transparent-gc/` — the same idea with rooting made transparent by a transform, so
  no explicit `gc-let`/`gctype` is needed.
- Jai comparison: it confirms codegen only sees post-comptime concrete code and keeps
  generics duck-typed, and leaves the comptime↔instantiation cycle undocumented.
