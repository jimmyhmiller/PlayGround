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
| **Checker** | `Program -> Diagnostics` | the whole program | located reports / veto | **building** |
| **Transformer** | `Program -> Program` | the whole program | a rewritten program | planned |

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

New API this project adds:

- **`(code-span node)` → Span** — expose the span every `Code` node already carries.
- **`(report severity span msg)`** — a non-fatal, located diagnostic sink; the
  build vetoes after a pass if any error-severity report was emitted.
- **`(checker f)` / `(transform f)`** — register a whole-program metaprogram.

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
- **1.1 Checker: `(checker f)`** where `f : Code(Program) -> Void`. Driver runs each
  registered checker after resolve (after check for typed queries), collects reports,
  vetoes on error. Sibling to the existing `run-metas`. **Unlocks: safety dialects.**
- **1.2 Transformer: `(transform f)`** where `f : Code(Program) -> Code(Program)`.
  Driver applies it and continues with the rewritten program (re-check after).
  **Unlocks: transparent GC rooting, GC lowering, Scheme lowering as a pass.**

### Phase 2 — ordered composition = dialects
- **2.1 A pass manifest** — an ordered list of registered metapasses the driver runs
  in order. The ordered stack *is* the dialect definition. Answers "can you have more
  than one?".

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
