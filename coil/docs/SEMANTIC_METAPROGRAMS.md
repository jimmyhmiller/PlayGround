# Semantic Metaprograms — Design

**Status:** design (not yet implemented). Supersedes the "Phase 4 — richer
reflection" long tail in `METAPROGRAMS.md` with a concrete architecture.

## Goal

Today a metaprogram (macro / generator / checker / transformer) sees the program
as **raw syntax** — `Sexp`/`Code` nodes with bare (unqualified) names, no AST, no
types, no def-site information. A call `(foo x y)` is a `KList` whose head is a
symbol node reading `"foo"`; the metaprogram cannot tell where `foo` is defined,
which module it came from, what its signature is, or what type `x` has.

We want metaprograms to be **fully semantic**: for *any* node in the program a
metaprogram can ask —

- **Where does this name resolve?** fully-qualified name + defining module + span.
- **What is its signature?** parameter types, return type, calling convention,
  variadicity, type parameters, trait bounds.
- **What is the type of this expression?** the checker's inferred `Type`.
- **What does this type/trait/sum look like?** fields, variants, methods — and
  disambiguated by the node's own module context (today's reflection is
  name-based and errors on cross-module name collisions).
- Whole-program relations: **callees, references, impl/trait resolution.**

The key realization from reading the compiler: **none of this is new
information.** It is all computed during normal compilation and discarded. The
work is not "compute semantics" — it is "compute it *before* metaprograms run,
persist it, and expose it as query operations." The rest of this document is how.

---

## 1. Where the data already lives

| What the metaprogram wants | Where it exists today | Exposed? |
|---|---|---|
| All function definitions (params, ret, cc) | `CtCtx.fns : (ArrayList Func)` — already in the comptime context | No |
| Struct / sum / trait definitions | `CtCtx.structs/sums/traits` | Partially (name-based, module-blind) |
| Import / export tables, current module | `CtCtx.qimports/qexports/qmodule/module_fns` | No |
| Fully-qualified name of a call/var | `resolve.coil::resolve` + `qualify-program` | No (resolver not callable from comptime) |
| Function signatures as a table | `check.coil` `Cx.sigs : (ArrayList Sig)` | No |
| Type of every expression | `check.coil` `synth-*` returns `EvalPair {e, t}` — **computed then thrown away** | No |
| Struct/sum/trait info for the checker | `Cx.structs/sums/trait_defs/methods/impls` | No |

So Phase S1 (signatures + resolution) is answerable **with data already sitting
in `CtCtx`** — no reordering required. Only expression *types* (Phase S2) require
running the checker before metaprograms and persisting its output.

---

## 2. The current pipeline and the two walls

```
read-all ─▶ load-program ─▶ expand-stage3 ─▶ resolve-program ─▶ frontend-check ─▶ mono ─▶ codegen
 (Sexp)      (imports,        (macros +         (Sexp→AST,          (typecheck;
             exports,          checker/          qualify names       synth returns
             TaggedForm)       transformer       via imports)        (Expr,Type),
                               hooks HERE)                            types discarded)
```

Metaprograms run at **`expand-stage3`** — after import loading and macro
expansion, but **before** parse-to-AST, name qualification, and typechecking.
That placement creates two walls:

**Wall A — the phase-ordering paradox (transformers).** A transformer *rewrites*
the program. To hand it types we must typecheck first. But a transformer can be
*what makes the program typecheck* — e.g. the shipped `desugar-inc` demo rewrites
`(inc E)` → `(iadd E 1)` where `inc` is otherwise undefined. So "typecheck, then
give the transformer a fully-typed tree" is **not universally valid**. Any design
must let semantic queries return `:unresolved` / `:unknown` gracefully instead of
aborting, and must be re-answerable after each rewrite.

**Wall B — the oracle byte-exact constraint.** AST and checked-program dumps are
blessed byte-for-byte by `selfhost/oracle/*.sh`. Adding a `ty` field to `Expr` or
changing dump output re-blesses the oracle. The semantic data must therefore live
in a **side table** keyed on node identity, not in the dumped AST.

---

## 3. Core architecture — a Semantic Model queried through code-ops

Two firm decisions frame everything:

### Decision 1 — metaprograms keep operating on `Code` (`Sexp`)

We do **not** switch metaprograms to a typed-AST value type. Syntax stays the
stable surface; semantics are **attached** to nodes and pulled on demand through
new query operations. This is additive (every existing macro/checker/transformer
keeps working untouched), matches the repo's "additive phases gated by the
oracle" philosophy, and mirrors how production compilers expose semantics —
Roslyn's `SemanticModel`, rust-analyzer's `Semantics`, Swift's `SourceKit` — as a
*side structure over the syntax tree*, never a replacement for it.

### Decision 2 — a compiler-built `SemanticModel` side table, demand-driven

The compiler builds a `SemanticModel`: a side structure mapping **node identity →
semantic facts** (resolved name, def-site, inferred type). Metaprograms reach it
through a family of query code-ops. A metaprogram that only wants syntax pays
nothing; one that wants semantics calls the queries.

```
                 ┌─────────────────────────────────────────┐
   Sexp nodes ──▶│  SemanticModel  (side table, not dumped) │
   (the program) │   nid → { qualified-name, def-site,      │
                 │           inferred-type, … }             │
                 │   + fn-sig index (from CtCtx.fns/Cx.sigs)│
                 │   + call graph / ref index (S3)          │
                 └─────────────────────────────────────────┘
                        ▲                         ▲
         built by resolve+check            queried by code-ops
                                     (resolve-sym, fn-sig, type-of, …)
```

### 3.1 Node identity — the linchpin

To answer "the type of *this* `Sexp` node," we must join a syntax node to the
`Expr` the parser produced and the `Type` the checker inferred. Today the only
join key is the span `(source, lo, hi, ctxt)`, copied Sexp→Expr by the parser.
Spans are **not** reliably unique: a macro that duplicates a subtree (e.g. `(when
c body)` expanding `body` once) yields two `Expr`s sharing one span+ctxt.

**Recommendation: a stable node id.** Add a non-dumped `nid : i64` to `Sexp` and
`Expr`, assigned monotonically at read time and to macro-generated nodes at
expansion, propagated through `parse-program` (Sexp.nid → Expr.nid). The
`SemanticModel` is keyed by `nid`. Because `nid` is **not** emitted by the
canonical dumpers, the AST/reader oracles stay byte-exact; quote/unquote
round-trips simply mint fresh ids for generated code (correct — generated nodes
are new).

*Interim fallback (if we want S1 before touching `Sexp`):* key by
`(source,lo,hi,ctxt)` and document the macro-duplication sharp edge. S1 (names +
signatures) rarely needs per-node identity — it resolves by symbol text in a
module context — so S1 can ship on span-keys and S2 introduces `nid`.

### 3.2 What a "Type" looks like to a metaprogram

Return types as **`Code`** — the type-expression `Sexp` (`i64`, `(ptr i64)`,
`(ArrayList Foo)`, `(-> [i64 i64] bool)`). This matches the existing convention:
`code-field-type` already returns a type as `(CCode (sx-sym …))`. To let a
metaprogram *destructure* a returned type without string-parsing, add value-level
deconstructors mirroring the existing kind tags:

- `(type-head T)` → `int|float|bool|void|ptr|ref|struct|array|slice|vec|fn|app|…`
- `(type-name T)` → the struct/app name symbol
- `(type-arg T i)` / `(type-arg-count T)` → generic arguments (`TApp`)
- `(type-int-bits T)` / `(type-signed? T)` — for `TInt`
- `(type-inner T)` — pointee/elem for `ptr/ref/slice/array`
- `(type-fn-params T)` / `(type-fn-ret T)` / `(type-fn-cc T)` — for `TFn`

These are the value-level twin of the existing `0=int 1=float … 8=other` kind
tags, generalized from "named types" to arbitrary `Type`s.

---

## 4. The new query API

All additive `code-*` / `type-*` ops. `NODE` is any `Code` value; queries that
can't answer return a sentinel keyword (`:unresolved`, `:unknown`) rather than
aborting, so metaprograms can run over not-yet-valid programs (Wall A).

### S1 — resolution & signatures (no reordering; data already in `CtCtx`)

- `(resolve-sym NODE)` → fully-qualified name symbol the head resolves to in its
  module context, or `:unresolved`. Backed by `resolve` + `CtCtx.qimports`.
- `(module-of NODE)` → the defining module symbol, or `:unresolved`.
- `(def-site NODE)` → a `Code` record `(def-site FILE LO HI MODULE KIND)` for
  where the name is defined (`KIND` ∈ `fn|struct|sum|trait|const|extern`).
- `(fn-sig NODE-or-NAME)` → a `Code` record
  `(fn-sig (params T…) (ret T) (cc C) (variadic B) (type-params N…) (bounds …))`,
  or `:unresolved`. Backed by `CtCtx.fns` (self-host) / `Cx.sigs` (checker).
- `(callee NODE)` → for a call form, the resolved callee name (sugar over
  `resolve-sym` on the head).
- **Node-context reflection** — `code-field-*` / `code-variant-*` / `code-trait-*`
  variants that take a *node* instead of a bare name, so the node's module
  context disambiguates. This directly removes today's "ambiguous type '…' —
  multiple modules define it" wall.

### S2 — expression types (requires the semantic loop, §5)

- `(type-of NODE)` → inferred `Type` as `Code`, or `:unknown`.
- `(kind-of NODE)` → the kind tag (int/float/struct/…) of `NODE`'s type,
  generalizing `code-*-kind` from named types to any expression.
- The `type-*` deconstructors from §3.2.

### S3 — whole-program relations (semantic indices)

- `(callees NODE)` → list of resolved callee names inside a form/function.
- `(refs-to NAME)` → all call/var sites that resolve to `NAME` (reverse index).
- `(impl-for TRAIT TYPE)` → the resolved impl (or `:none`) — trait/impl
  resolution for dialects that reason about dispatch.
- `(reachable-from NAME)` → transitive callee set (dead-code, effect analysis).

---

## 5. The semantic elaboration loop (resolving Wall A)

Expression types (S2) require the program to be parsed, resolved, and typechecked
*before* metaprograms observe it — but transformers rewrite the program and can
be what makes it valid. The resolution is a **best-effort, fixpoint loop**:

```
read → load → expand-macros            (call-site macros only; NOT checkers/transformers)
     │
     ▼  ┌──────────────── SEMANTIC LOOP (guarded fixpoint) ───────────────┐
     │  │ parse → resolve/qualify → typecheck(BEST-EFFORT) → SemanticModel │
     │  │ run transformers   (each queries the SemanticModel; may rewrite) │
     │  │ if program changed → re-enter loop                               │
     │  │ else → exit loop                                                 │
     │  └──────────────────────────────────────────────────────────────────┘
     ▼
   run checkers   (query the FINAL SemanticModel; strict; veto/report)
     │
     ▼
   frontend-check (STRICT, authoritative) → mono → codegen
```

Four load-bearing properties:

1. **Best-effort typecheck.** During the loop the checker must *collect-and-
   continue*: an unresolved name or type error marks that node's type `:unknown`
   / `:error` and keeps going, so transformers still get real semantics for the
   parts that do resolve. This is exactly how IDE semantic models tolerate broken
   code. The **final** `frontend-check` after the loop settles is strict and
   authoritative — it produces the real diagnostics and the program that goes to
   mono. (This is the single biggest implementation cost — see §7.)

2. **Fixpoint with a fuel guard.** Transformers re-run until the program stops
   changing, bounded by a max-iteration guard (mirrors macro-expansion fuel).
   Each iteration re-derives the `SemanticModel` from the current program (simple
   and correct; §S4 makes it incremental later).

3. **Transformers before checkers.** Preserves today's ordering: transformers
   reach fixpoint first, then checkers observe the settled, typed program and
   can't be invalidated by a later rewrite.

4. **Types are advisory inside the loop, authoritative after.** A transformer
   reads types to *decide* rewrites; correctness is still enforced by the final
   strict check. A transformer that produces ill-typed code fails the final
   check with a normal diagnostic — the loop never hides errors.

---

## 6. Delivery plan (additive, oracle-gated)

Each phase is independently shippable and gated by `selfhost/oracle/*.sh` +
rebootstrap fixpoint, in the established style.

- **S0 — node identity.** Add non-dumped `nid` to `Sexp`/`Expr`; assign at
  read/expand; propagate through `parse-program`. Oracle stays byte-exact.
  *Foundation for all per-node joins.* (Skippable for S1 if we use span-keys.)

- **S1 — resolution & signature reflection.** Expose `resolve-sym`, `module-of`,
  `def-site`, `fn-sig`, node-context `code-field*/variant*/trait*`. **No
  reordering.** This alone answers the two questions that motivated this work
  ("where does `foo` live," "what's its signature") and disambiguates cross-module
  reflection. **Ship first — largest value, lowest risk, additive to the dumps.**

  **S1.0 — checkers run post-typecheck + `(code-decl NODE)` — ✅ SHIPPED.** The
  load-bearing change is **phase ordering**: checkers now run in a dedicated phase
  *after* the whole program is resolved + typechecked, so they read the compiler's
  **authoritative** output — not a parallel resolver. Concretely, `expander.coil::
  run-expand` no longer runs checkers; it *registers* them (their names + the compiled
  metaprogram closure) via `register-checkers`, and still runs macros + transformers
  (which are syntactic, pre-resolution). Then `driver.coil::run-pipeline`, right after
  `frontend-check` produces the checked `Program`, calls `run-registered-checkers`,
  which stashes that program as the **semantic model** (`set-sem-model`) and runs each
  checker against it. Consequence: a checker can only veto programs that already
  typecheck — it layers *policy* on a valid program (exactly what a dialect wants).

  On top of that phase sits `(code-decl NODE)` — one op folding "where does it live"
  + "what's its type". Given a Code node (a symbol, or a call whose head is a symbol)
  it returns a Code record read straight from the checked program:

  ```
  (decl MODULE fn [PARAM-TYPE…] RET) ; function: real module + real signature, walkable Code
  (decl MODULE KIND)                 ; struct / sum / trait / const / extern
  :unresolved                        ; no such top-level definition
  :ambiguous                         ; defined in >1 module, can't disambiguate
  ```

  Backed by lookups over the authoritative checked `Program` (`comptime.coil`
  `cp-find-fn`/`cp-kind-decls`, keyed on the resolver's fully-qualified names) plus a
  `Type`→`Code` reconstructor (`ty->sexp`). Op code **35**; parser recognizes
  `code-decl`; no checker-typing change (defaults to `TCode`). No oracle re-bless —
  rebootstrap fixpoint + all gates green; every existing checker/transformer demo
  unchanged. Demo: `metaprog-poc/sigcheck{,_ok,_bad,_cross}.coil` — a policy checker
  that reads each callee's real declared **return type** (incl. a function imported
  from another module) and vetoes calls to pointer-returning functions (type-correct
  code the policy forbids). **Known limits (v1):** looks up by simple/module-qualified
  name, not alias-qualified (`str/split`); functions carry the full signature, other
  kinds carry module + kind only; no per-*expression* types yet (that's S2). Next S1
  ergonomics: `fn-sig`, `module-of`, `def-site` as thin wrappers, and alias-aware
  resolution.

  **Note vs the original plan:** the first cut of this used a parallel declaration
  index built at `expand-stage3` (a second, approximate resolver). That was replaced
  by the post-typecheck phase above so the metaprogram reads the *one* authoritative
  result.

  **S1.1 — semantic transformers (`(semantic-transform FN)`) — ✅ SHIPPED.** A
  transformer that runs in the same post-typecheck phase: it reads the authoritative
  checked program (via `code-decl` etc.) to decide its rewrite, returns rewritten
  code, and the pipeline **re-resolves + re-typechecks to a fixpoint** before codegen.
  This is the Wall-A resolution in practice — the transformer's *input* must typecheck
  (so it layers on a valid program), and each rewrite is re-checked. `expander.coil::
  run-semantic-transforms` drives the loop (`sem-tx-apply-once` runs the transformers
  and writes `s.out`; `sem-tx-round` re-resolves/re-checks and recurses until the
  program stabilizes or a 16-round guard trips). Syntactic `(transform FN)` still runs
  at `expand-stage3` (pre-resolution) for desugarings that *produce* typeable code
  (e.g. `inc`→`iadd`); semantic transformers are for rewrites that *need* types.
  Ordering: syntactic transforms → resolve → check → semantic transforms (fixpoint) →
  checkers → mono. Demo: `metaprog-poc/retkind{,_test}.coil` — reads each wrapped
  call's real return type and rewrites a marker to `1` (pointer) / `2` (non-pointer),
  observable in the exit code (12 = `1*10 + 2`; 0 without the transform). Verified:
  rebootstrap fixpoint + gates green; existing syntactic-transform/checker demos
  unchanged. **Implementation note:** the fixpoint must be written as a *recursion*
  with the `s.out` store isolated in its own function (`sem-tx-apply-once`) that
  returns before the next `resolve-program` reads it — a mutable-accumulator loop that
  stores `s.out` and re-reads it through a call in the same frame miscompiles.

- **S2 — the semantic loop + type map.** Reorder to parse/resolve/best-effort-
  check before transformers/checkers; build the `nid→Type` map; expose
  `type-of`, `kind-of`, the `type-*` deconstructors. The heavy phase (best-effort
  checker + fixpoint loop). Unlocks true expression typing.

- **S3 — whole-program semantic indices.** Call graph + reference index +
  trait/impl resolution (`callees`, `refs-to`, `impl-for`, `reachable-from`).
  Enables real linters/optimizers (dead code, effect analysis, the GC dialect's
  rooting decisions).

- **S4 — incremental re-derivation (perf).** Re-check only forms a transformer
  touched instead of whole-program per fixpoint iteration.

**Shortest path to real value:** S1 by itself removes both walls the user hit.
S0+S2 delivers expression types. S3 is what dialects (GC rooting, ownership,
effect systems) actually need.

---

## 7. Risks & open decisions

1. **Best-effort checker is the hard part.** `frontend-check` currently aborts on
   the first error (`Result … Diag`). S2 needs a collect-and-continue mode that
   assigns `:unknown` and proceeds. This is a genuine refactor of `check.coil`,
   not a bolt-on. *Confirm appetite before S2.*

2. **Transformer confluence/termination.** A type-driven transformer changes the
   program → changes types → possibly changes what it would rewrite. We run to a
   guarded fixpoint and document that transformers should be **monotonic** (never
   un-rewrite). Do we want to *enforce* monotonicity, or just document it?

3. **Node identity: `nid` vs span-key.** `nid` is robust but touches
   reader/parser/expander. Span-keys are zero-new-fields but wrong under macro
   duplication. Recommendation: `nid` (S0). Acceptable to ship S1 on span-keys.

4. **Scope of semantics — user code vs library.** Should `type-of` /
   `fn-sig` answer for imported stdlib nodes too? The model *can*; the cost is
   typing the whole loaded program each fixpoint iteration. `code-from-user?`
   already lets a metaprogram scope its *reporting*; the question is whether we
   also scope the *model's construction* for performance.

5. **Oracle re-bless.** S0/S1 are additive (no dump change). S2's reordering
   should yield the same final program, but the best-effort pass may change
   diagnostic ordering/messages — likely re-blesses the *diagnostic* oracles
   (not the AST ones). Plan for it.

6. **Comptime interpreter reach.** Note the existing limitation ("checkers can't
   call imported string functions — the closure doesn't include them"). The
   query ops are compiler builtins, not imported Coil functions, so they sidestep
   that wall — but any *helper* a metaprogram wants to call still needs to be in
   its closure. Worth deciding whether the semantic API should also widen the
   metaprogram closure.

---

## 8. Summary

The semantics already exist inside the compiler; they are computed and discarded.
The design is: **(1)** keep metaprograms on `Code`, **(2)** build a
`SemanticModel` side table keyed by stable node id, **(3)** expose it through
additive query code-ops, and **(4)** for expression types, run a best-effort
resolve+check *before* metaprograms inside a guarded transformer fixpoint loop,
with a final strict check as the authority. S1 (names + signatures) ships first
with no reordering and removes the two walls immediately; S2 adds real expression
types; S3 adds the whole-program indices that dialects need.
