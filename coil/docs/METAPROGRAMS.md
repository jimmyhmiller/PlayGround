# Metaprograms in Coil

A **metaprogram** is a Coil function that runs at compile time and operates on the
program. There is no separate metalanguage — it is Coil operating on Coil, run by
the comptime interpreter, its output type-checked and compiled normally.

## The four kinds

Distinguished only by *what they receive* and *what they return*.

| kind | signature | receives | returns | status |
|---|---|---|---|---|
| **Macro** | `[Code…] -> Code` | its own call site | replacement code | shipped |
| **Generator** | `() -> Code` (via `meta`) | nothing | new top-level forms | shipped |
| **Checker** | `Program -> Code` | the whole program | veto via `error` | **shipped** |
| **Transformer** | `Program -> Program` | the whole program | a rewritten program | **shipped** |

## Applying a metaprogram

- **Macros** — *call* them: `(when c body)`. Detected by their `Code` signature.
- **Checkers / transformers** — *register* them at top level: `(checker lint-icmp)` /
  `(transform desugar-inc)`. The compiler runs them during compilation.
- **Dialects** — *import* a module that contains those registrations: one
  `(import "safe_dialect.coil")` applies its whole stack.
- **From the CLI, optionally** — `coil run app.coil --use lint.coil` imports a
  metaprogram module (which self-registers its `(checker …)`) **without editing the
  source**. Repeatable; works on `run` and `build`. This is how you run a linter on
  demand: `coil run app.coil --use lint-on.coil`.

All four share one substrate: the `Code` value, the `code-*` operations, type
reflection, and the comptime interpreter. The difference is **scope** (my-call-site
vs whole-program) and **power** (produce vs reject vs rewrite). An ordered stack of
checkers/transformers is what we mean by a **dialect** (e.g. the GC dialect, a
Rust-like ownership dialect, a Scheme frontend).

## The API (the vocabulary), all shipped

- **Take Code apart:** `code-count`, `code-nth`, `code-rest`, `code-sym`,
  `code-list?`, `code-sym?`, `code-int?`, `code-keyword?`, `code-str`, `code-eq`.
- **Build Code:** quote `` ` ``, unquote `~`, splice `~@`, `code-symbol` (make a
  name from a string), `gensym` (fresh hygienic name).
- **Reflect on types:** `code-field-count/name/kind/type`, `code-variant-*`,
  `code-trait-*` (and value-form `field-count`, `struct?`, `field-name`, …).
  Kind tags: `0=int 1=float 2=bool 3=struct 4=sum 5=ptr 6=array 7=slice 8=other`.
- **Compute:** the comptime interpreter runs arbitrary *monomorphic* Coil.
- **Fail / branch:** `error` (abort expansion with a message), `target-arch`.

New API this project added (all shipped):

- **`(report NODE MSG)`** — a located compile-time **error** at `NODE`. It **collects**:
  a checker keeps running and surfaces EVERY error in one pass; the build fails after
  printing them all (see `metaprog-poc/located_multi.coil` → 2 errors, then failure).
- **`(warn NODE MSG)`** — a located, **non-fatal** warning at `NODE`; collects and all
  print, the build succeeds. What a *linter* wants — `metaprog-poc/lint.coil` warns at
  every `icmp-*`, suggesting `< > = …`.
- **Metaprograms are fed the WHOLE program, including all imports** (their own modules
  and bundled stdlib). A checker sees imported code too — `metaprog-poc/imports_test.coil`
  shows the linter flag an `icmp` in an imported user module.
- **`(code-file NODE)` → the source file name** of a node, and **`(code-from-user? NODE)`
  → bool** (true for a real file, false for a bundled `<…>` source). So a linter can
  *scope itself* — `metaprog-poc/lint.coil` warns only where `(code-from-user? f)`, which
  skips the standard library while still linting the user's own modules. (Checkers can't
  call imported string functions — the closure doesn't include them — so `code-from-user?`
  does the check in the compiler and hands the checker a bool.)
- **`(checker FN)` / `(transform FN)`** — register a whole-program metaprogram.
- **A dialect is a single import.** A module that contains `(checker …)`/`(transform …)`
  registrations *is* a dialect — importing it applies the whole stack (import order =
  pass order; transformers run before checkers). No new syntax; the module is the
  manifest. See `metaprog-poc/safe_dialect.coil`.

## What today's metaprograms can and can't do

Can: read any form's syntax; generate code (inline + new top-level defs); compute at
compile time; reflect on struct fields / sum variants / trait method signatures;
compose and recurse to a fixpoint; abort the build; branch on target.

Can't (the walls this project removes):

1. **See the whole program.** A macro sees only its call-site args; `meta`
   generates but isn't handed every form. → no whole-program checkers/transformers.
2. **Reject precisely.** `error` aborts at one site with a bare string. No
   multi-error, span-located reporting that collects then vetoes.
3. **Rewrite code they don't own.** A macro rewrites its own subtree; `meta` adds
   code, never rewrites existing forms.
4. **Intercept core special forms.** `if`, `store!`, `index`, `alloc-stack` are
   parsed as special forms; a macro named `store!` can't shadow them.

Narrower gaps: no function-signature reflection; no `sizeof`/layout at comptime
(codegen phase); generics don't run in the comptime interpreter.

## Plan (all changes in `selfhost/src/*.coil`; the Rust compiler is gone)

Each phase is additive and gated by the oracle (`selfhost/oracle/*.sh`). Additive
phases keep the committed snapshots byte-exact; only core-form demotion (Phase 3)
re-blesses the oracle.

### Phase 0 — foundations
- **0.1 `(code-span node)`** — a code-op exposing the reader span. `comptime.coil`.
- **0.2 `(report severity span msg)` + collector** — non-fatal, located, vetoes
  after the pass. `comptime.coil` builtin + a diagnostic list in `driver.coil`.
- **0.3 the `Program` value** — a `Code` list of all top-level forms (post-resolve).
  Built in `driver.coil`.

### Phase 1 — the whole-program hook (the core addition)
- **1.1 Checker: `(checker FN)` — ✅ SHIPPED.** `FN` is an ordinary `[(prog Code)] ->
  Code` function; `(checker FN)` registers it. After macro expansion, the driver hands
  FN the **whole program** (all top-level forms as one Code list) via the existing
  `expand-macro` machinery, and FN VETOES compilation by calling `error` (its returned
  Code is ignored). Because FN is Code-signed, the existing macro closure already
  type-checks it; the hook (`run-checkers` in `selfhost/src/expander.coil`) just invokes
  it. Imported checker libraries work (name resolved via `resolve-macro`). Demo:
  `metaprog-poc/policy_{ok,bad}.coil` — an imported `no-forbidden-op` checker vetoes a
  program that calls a banned form, anywhere, buried in any function body. Verified: full
  oracle green + rebootstrap fixpoint. **Known limitation:** the checker sees *all* loaded
  forms including imported stdlib, so a "ban X everywhere" policy also inspects library
  code — scope-to-user-module and located `(report …)`/`(code-span …)` diagnostics (v1
  vetoes with a message only) are the next refinements.
- **1.2 Transformer: `(transform FN)` — ✅ SHIPPED.** `FN` is a `[(prog Code)] -> Code`
  function; `(transform FN)` registers it. After expansion, the driver hands FN the whole
  program and **replaces** it with FN's returned `(do form…)`, then re-resolves/checks the
  result. Implemented in `expander.coil::run-transformers` (mirrors `run-checkers`, runs
  before checkers). **v1 constraint:** FN must return the *same number* of top-level forms
  (an in-place rewrite), so each output form inherits the module tag it replaces — this
  keeps module structure intact without threading it through the flat Code list. Demo:
  `metaprog-poc/tx_test.coil` + `dialect.coil` — a `desugar-inc` transformer rewrites
  `(inc E)` → `(iadd E 1)` program-wide (`inc` is otherwise undefined, so the program only
  compiles *because* of the transform). Verified: full oracle green + rebootstrap fixpoint.
  **Known limitations:** the transformer sees *all* loaded forms (incl. prelude) — a naive
  rewrite must leave forms it doesn't target untouched (the demo only rebuilds forms
  containing the target) or it corrupts library code; and it can't add/remove top-level
  forms yet (same-count v1). Scope-to-user-module and add/remove-forms are the refinements.
  **Unlocks: transparent GC rooting, GC lowering, Scheme lowering as a pass.**

### Phase 2 — ordered composition = dialects — ✅ SHIPPED (via imports)
- **2.1** No new syntax was needed: `(checker …)`/`(transform …)` registrations in an
  imported module compose, and **an imported module bundling them IS a dialect**.
  Import order = pass order; transformers run before checkers. `metaprog-poc/safe_dialect.coil`
  bundles a transform + a checker; `(import "safe_dialect.coil")` applies both.
  A named `(dialect …)` form would be pure sugar over this.

### Phase 3 — core-form demotion (à la carte)
- **3.1 Demote `store!` first** — parser emits an interceptable `(store! …)` call over
  a `%store!` primitive so transformers can rewrite it on idiomatic code. **The one
  phase that re-blesses the oracle.** Unlocks write barriers / bounds-checking on
  unmodified Coil.

### Phase 4 — richer reflection (long tail)
- **4.1** function-signature reflection (auto-coercion at dialect boundaries);
  **4.2** comptime↔mono lazy-mono service (generics at comptime);
  **4.3** a layout sub-phase (`sizeof` at comptime).

### Shortest path to something real
Phase 0 + 1.1 = whole-program read + reject = a working safety checker. 1.2 + 2.1 =
the GC and Scheme dialects. 3.1 = barriers/bounds on idiomatic code.

## Related work in the repo
- `gc-dialect-poc/` — a precise mark-sweep GC built entirely as macros + a runtime
  library (implicit allocation + reclamation, reflection-generated tracers). It needs
  1.2 (transparent rooting) to drop its explicit `gc-let`/`gctype`.
- Jai comparison and the comptime↔mono phasing cycle: see the design discussion; Jai
  confirms codegen only sees post-comptime concrete code, keeps generics duck-typed,
  and leaves the true comptime↔instantiation cycle undocumented (so 4.2's cycle guard
  is new ground).
