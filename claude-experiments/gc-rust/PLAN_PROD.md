# Plan: gc-rust → self-hosting-grade production language

> **Status: all six phases below are DONE.** This document is the original
> production plan, kept for context. The language now has strings + a string
> stdlib, `Vec`/`MapStr`/iterators, complete `match` (literals, guards, scalar
> matching), a module system (`mod`/`use`/`pub`, multi-file + `gcr.toml`
> projects), good diagnostics (precise carets, "did you mean?", multi-error),
> and wrapping integer overflow — on a precise **generational** GC. Plus
> hardening done after the plan: parser fuzzing, multi-error reporting, and the
> generational collector turned on. See the README for the current feature list
> and benchmarks, and `docs/` for the per-subsystem references.

**Target (chosen):** Self-hosting-grade. Strings + formatting, real stdlib
(`String`/`Vec`/`HashMap`/iterators), modules/imports, complete `match`, rich
diagnostics with carets, and a package manifest. The bar is: *the language can
write non-trivial real programs* — ultimately including (a meaningful subset of)
its own tooling.

**Verified starting point (as of 2026-06-15, checked against the code):**

- Pipeline works end-to-end: lex → parse → resolve → typecheck+mono → core IR →
  LLVM O2 → JIT/AOT. `fib`, `nbody_vec3`, `types.gcr` all run with correct output.
- **Type checker is real**: bidirectional + light unification, literal defaulting,
  generic instantiation, **exhaustiveness checking on `match`**. Fused into
  `lower.rs` (not a standalone pass).
- **GC is production-grade**: precise semi-space copying collector, shadow-stack
  roots, safepoints, relocates under stress. 92 GC tests green. Dormant
  generational/concurrent paths exist (`card_table.rs`, `barrier.rs`) but unused.
- **Prelude system exists**: `src/prelude.gcr` is written *in gc-rust* and
  injected; `Option`/`Result`/`Vec` already live there. Stdlib-in-the-language
  is the established pattern.
- **Runtime ABI is minimal**: `ai_gc_alloc_fixed`, `ai_gc_alloc_varlen`
  (varlen = basis for String/Vec backing), `ai_print_int/float`,
  `ai_gc_pollcheck_slow`. New stdlib needs a few new runtime entry points.
- **Working language surface**: structs, enums+payloads, generics (monomorphized,
  no boxing), traits+dispatch, methods, closures+captures, `Option`/`Result`+`?`,
  GC arrays, growable `Vec`, value types (flat, in registers), tuples.

**Verified gap inventory (from grepping `not supported`/`v0 slice`/testing):**

| Gap | Evidence | Phase |
|---|---|---|
| Strings construct-only; no `print_str`/len/concat/index/format | `print_str` → "unknown function"; only `ConstStr` exists | 1 |
| `check` subcommand only resolves, doesn't typecheck | `main.rs:81` runs `resolve_module` only | 1 |
| `match` enum-only; no guards, nested, literal/range patterns | `lower.rs:982,1010,1022` | 2 |
| `let` without initializer rejected | `lower.rs:446` | 2 |
| tuple field assignment unsupported | `lower.rs:943` | 2 |
| only direct calls to named non-generic fns (no fn-ptr values fully) | `lower.rs:1514` | 2 |
| no `HashMap`, iterators, slices, real `String` type | absent from prelude | 3 |
| no module system / imports / multi-file | single-module + prelude only | 4 |
| diagnostics are `err("string", span)`, no caret rendering | `diag.rs` minimal | 5 |
| no package manifest / dependency story | none | 6 |
| stale docs: `types.rs` cites nonexistent `src/mono.rs`; README undersells tuples, oversells strings | — | each phase fixes its own |
| thin E2E test coverage (4 compiler tests vs 92 GC) | `tests/` | cross-cutting |

---

## Phasing & dependency order

Phases 1–2 are **foundational and mostly serial** (they unblock everything).
Phases 3–6 are **heavily parallelizable** — that's where the "plan + parallel
build" fan-out happens. Each phase ends with a green-tests gate.

### Phase 1 — Strings + honest `check` (foundation) — *serial, do first*

The single biggest usability unlock. Strings already have a heap object and
varlen alloc; we need the *consumer* layer.

1a. **String runtime + type**
   - Promote `Prim::Str` to a first-class varlen heap type `String` (len + UTF-8
     bytes in the varlen tail), allocated via `ai_gc_alloc_varlen`.
   - New runtime entry points: `ai_print_str`, `ai_str_len`, `ai_str_concat`,
     `ai_str_eq`, `ai_str_from_int`/`ai_str_from_float` (for formatting).
   - String literals (`ConstStr`) lower to a varlen alloc + memcpy of the bytes.

1b. **String stdlib (prelude, in-language where possible)**
   - `str_len`, `str_concat` (`+` overload or `concat()`), `str_eq`, `str_get`
     (byte/char), `str_slice`, `to_string(i64)`, `to_string(f64)`.
   - `print` / `println` that take strings.

1c. **`println!`-style formatting** (start minimal)
   - A `format(parts, args)` builtin or a simple `concat`-based interpolation.
     Full `{}` macro can wait; ship string concat + `to_string` first.

1d. **Fix `check`**: route `check` through the typecheck/lower pass (not just
   `resolve_module`) so `gcr check` actually reports type errors. Add a
   `--no-codegen` fast path that stops after `lower_program`.

1e. Update README + `types.rs` header (drop the `mono.rs` reference), add
   string examples to the tour.

**Gate:** a `hello.gcr` that prints a string, concatenates, and formats an int
compiles and runs via both JIT and AOT. `gcr check` rejects a type-wrong program.

### Phase 2 — Complete the core language — *serial-ish, unblocks stdlib*

2a. **`match` completion** (the big one):
   - guards (`n if n > 5`) — `lower.rs:1022`
   - literal patterns (int/bool/char/string) and match-on-scalars — `lower.rs:982`
   - nested patterns — `lower.rs:1010`
   - ranges (`0..=9`) — optional, after the above
   - keep exhaustiveness checking correct as patterns get richer.
2b. `let` without initializer + definite-assignment check — `lower.rs:446`.
2c. tuple field assignment — `lower.rs:943`.
2d. First-class function values / closures as fn-ptr+env everywhere a value is
   expected (not just direct calls) — `lower.rs:1514`. Needed for iterators.

**Gate:** a program using guards, nested patterns, literal match arms, and a
stored closure passed to a higher-order fn compiles and runs.

### Phase 3 — Core stdlib — *PARALLEL: one agent per module*

All written in gc-rust in the prelude (or a multi-file std once Phase 4 lands),
leaning on Phases 1–2. Independent, fan out:

- **3a `String` methods**: split, trim, find, replace, chars, parse_int/float.
- **3b `Vec<T>` completion**: pop, insert, remove, iter, map/filter/fold,
  contains, sort (needs comparison or `Ord` trait).
- **3c `HashMap<K,V>`**: needs a `hash` story — start with `hash_i64`/`hash_str`
  runtime primitives + open-addressing table in-language. Trait `Hash`/`Eq`.
- **3d Iterators**: an `Iterator` trait + `next() -> Option<T>`, adapter combinators
  (`map`/`filter`/`take`/`enumerate`/`zip`), `for` desugars to it. Depends on 2d.
- **3e Slices / array views**: `&[T]`-equivalent without borrows (a fat pointer
  value type: ptr+len), bounds-checked.
- **3f Numeric/`?`/conversions**: `parse`, `From`/`Into`-style conversions,
  `Ord`/`PartialOrd`/`Eq` traits used by Vec/HashMap.

**Gate:** golden-output examples exercising each module; a small program that
reads data into a `Vec`, builds a `HashMap`, iterates, formats results.

### Phase 4 — Module system + multi-file — *PARALLEL with 5 & 6*

4a. `mod`/`use`/`pub`, file-based modules, a resolver that loads a module graph
   (replaces the single-module + prelude-injection model; prelude becomes
   `std` modules that are implicitly `use`d).
4b. Name resolution across modules; visibility checking.
4c. Split the monolithic prelude into `std::{string,vec,map,iter,...}`.

**Gate:** a 3-file program (`main` + 2 modules + `std`) compiles; visibility
violations are rejected.

### Phase 5 — Diagnostics — *PARALLEL with 4 & 6*

5a. A real diagnostic type: severity, primary + secondary spans, labels, notes,
   help. Source-mapped caret rendering (à la rustc/ariadne).
5b. Convert the `err("...", span)` call sites in `lower.rs`/`parser.rs`/`resolve.rs`
   to structured diagnostics with good messages and suggestions.
5c. Multiple-error reporting (don't stop at the first) where feasible.

**Gate:** a snapshot test corpus of broken programs → exact rendered diagnostics.

### Phase 6 — Packaging, tooling, hardening — *PARALLEL, then converge*

6a. **Package manifest** (`gcr.toml`): name, version, deps (path/git first),
   entry points. A simple resolver + build cache.
6b. **Test infrastructure**: golden-output runner, `gc_every_alloc` stress mode
   wired into CI (port from ai-lang), fuzz the parser, property tests on the GC.
6c. **Integer overflow semantics**: settle the open question (wrapping vs
   checked vs trap) and implement consistently — currently undefined.
6d. **Perf regression suite**: keep the 5 benchmarks (`fib`, `loop_mix`,
   `mandelbrot`, `nbody`, `binary_trees`) green vs Rust baselines in CI.
6e. **Self-hosting milestone (stretch)**: port a real tool (e.g. the lexer or a
   JSON parser) to gc-rust as the credibility proof.

**Gate (v1 done):** `gcr build` a multi-file project with stdlib + a package
dep; all benchmarks within target of Rust; stress suite green; diagnostics
render with carets; a non-trivial real program (the self-hosting demo) runs.

---

## Parallel build strategy

- **Phases 1 → 2 are the gate.** Build serially, land green, *then* fan out.
  They touch the same hot files (`lower.rs`, `codegen.rs`) and conflict if
  parallelized.
- **After Phase 2**, fan out with one agent per independent area, each on its
  own git worktree to avoid file conflicts:
  - Phase 3: 3a–3f are independent stdlib modules (mostly new `.gcr` + a few
    runtime primitives) → 6 parallel agents.
  - Phases 4, 5, 6 touch different subsystems (resolver/CLI vs diag vs
    tooling/CI) and can run alongside Phase 3.
- **Frozen contracts** so parallel agents don't block each other:
  - Runtime ABI additions (new `ai_*` entry points) — agree the signatures up
    front; they're the boundary between in-language stdlib and the runtime.
  - The `Iterator`/`Hash`/`Ord`/`Eq` trait signatures — freeze before 3b/3c/3d.
  - The diagnostic type — freeze before converting call sites.
- **Integration at the gates.** Each phase has a green-tests gate; merge
  worktrees at the gate, not continuously.

## Risks / sequencing notes

- **Iterators (3d) depend on first-class closures (2d).** Don't start 3d until 2d lands.
- **HashMap (3c) needs a hash primitive + `Hash`/`Eq` traits** — small runtime
  work + a trait-design decision; sequence the trait freeze before 3b/3c.
- **Module system (Phase 4) changes how the prelude is delivered** — coordinate
  with Phase 3 so stdlib modules land in the new layout, not the old monolith.
- **GC is NOT on the critical path** — it's done. Don't touch the dormant
  generational paths unless a perf gate demands it (explicit non-goal for v1).
- **Codegen completeness**: several gaps are `codegen unsupported in v0 slice`
  (`codegen.rs:475`) — each new feature needs both a lower.rs *and* codegen.rs
  arm. Budget codegen work per feature, not just frontend.

## Rough sizing

- Phase 1: ~3–5 days (serial). Phase 2: ~1 week (serial).
- Phases 3–6 in parallel: ~2–4 weeks wall-clock with fan-out (more in
  sequential effort). Self-hosting stretch (6e): open-ended.
- **Total to a credible self-hosting-grade v1: ~6–10 weeks** with parallel
  agents, dominated by stdlib breadth, modules, and diagnostics — not by
  research risk. The hard research (GC + monomorphized value types + LLVM
  integration) is already done and proven.
