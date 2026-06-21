# Handoff: debugger P1b — source-id-per-span (production multi-source fix)

Lossless checkpoint for a FRESH session to execute the source-id refactor with
fresh care (a foundation change touching pervasive `Span` structures — not to be
rushed at the tail of a long session). Read this + `docs/DEBUGGER_DESIGN.md`.

## State (all committed)

- **Debugger design**: approved (`docs/DEBUGGER_DESIGN.md`), incl. the P2 (DWARF)
  appendix.
- **P1 span-threading**: DONE + self-reviewed (5-lens workflow) + all 4 majors
  fixed (commits up to `858b2ec2f`). SpanId side-table (`CoreExpr.span` u32,
  `NO_SPAN=0`; `CoreProgram.spans` interned table) threaded AST→lower→ANF→codegen;
  the file:line:col alloc-site upgrade proves it. ANF preserves spans across its
  single rebuild (`anf.rs:378`). Verified: gcrust-rt 108/0, alloc_profile 5/0
  (parallel, non-flaky), aot 3/0, all integration suites green, gc-stress gate
  green (detector armed, no trip).
- **Known residual (the reason for THIS task)**: span resolution
  (`core::CoreProgram::span_location` → `diag::resolve_location`) is an INTERIM
  offset heuristic: `off < user_src.len()` → user, else prelude. Prelude/module
  sources are lexed in their OWN 0-based offset spaces, so a prelude span at
  offset X mis-resolves to the user file whenever `X < user_src.len()`. This is
  **TOTAL for any user file larger than the prelude (~44 KB)** (gc-rust's larger
  files exceed that), partial for medium files, fine for small. Most real allocs
  (Vec/array_new/Option/String) are IN the prelude → mislabeled. It MUST be fixed
  before P2 (DWARF resolves spans → a prelude frame in a backtrace inherits a
  wrong line). Decided: source-id-per-span (Leader-concurred).

## The fix: source-id-per-span (additive, exact, multi-module-ready)

Each span names its source; resolution uses the right source text. **Additive** —
`diag::render`'s 11 error call sites stay unchanged (the source-id is consumed
ONLY by the debugger resolution; `render` keeps its 0-based-offset heuristic,
fine for the rare prelude error). No GC / hot-path impact (front-end metadata).

### Steps

1. **`src/lexer.rs`** — add `source: SourceId` to `Span` (`type SourceId = u16;`,
   default `0` = the primary/user source). `Span::new` defaults `source: 0`. Keep
   `Span` `Copy`/`Eq`/`Serialize`. (Most code ignores the field; only tagging +
   resolution touch it.)

2. **Tag spans per source** — the only real work. Two options; pick the one with
   fewer touch points after a quick look:
   - **(a) AST span-walk at merge (recommended — localized):** a
     `set_module_source(&mut Module, sid: SourceId)` that recursively sets
     `span.source = sid` on EVERY span in a parsed Module (Item/Fn/Expr/Stmt/
     Pattern/Type/FieldInit/Arm/…). Call it in `compile.rs` right after parsing
     each source, before merging items. Robust (covers all spans regardless of
     how the parser built them); cost = one boilerplate visitor.
   - **(b) parser-preserve:** make the parser carry `source` when it combines
     token spans (`Span { start: a.start, end: b.end, source: a.source }`). Fewer
     lines IF the parser combines via a central helper; more if span-combine is
     scattered. Check `src/parser.rs` for a span-combine helper first.

3. **`src/compile.rs`** — assign a `SourceId` per source: user=0, prelude=1, each
   `mod` file =2+. After parsing each source, tag it (step 2), then merge items as
   today. Build a `SourceMap = Vec<SourceEntry { path: String, text: String }>`
   indexed by `SourceId`. Thread it out: `parse_with_prelude` /
   `parse_file_with_prelude` return `(Module, SourceMap)`. (`load_file_modules`
   recursion: each loaded mod gets the next id + a SourceMap entry.)

4. **`src/core.rs`** — replace `CoreProgram.src_path`/`src_text` with
   `sources: Vec<SourceEntry>` (the SourceMap). Rewrite `span_location`:
   `s = spans[id-1]; e = sources.get(s.source)?; (label=e.path, line_col(e.text,
   s.start))`. `s.start` is the 0-based local offset for `s.source` → correct.
   (`span_label` formats it; codegen `alloc_site_id` already uses `span_label`.)
   The interned span table is UNCHANGED — it already stores the full `Span`, which
   now carries `source`.

5. **`src/main.rs`** — attach the `SourceMap` to `prog` (run + build paths)
   instead of `src_path`/`src_text`.

6. **`diag::render` — leave UNCHANGED** (and delete the now-unused
   `diag::resolve_location` the interim added, or repoint it at the SourceMap if
   you want render to be exact too — optional, not required).

### Gotchas (already handled / to watch)

- ANF already preserves `CoreExpr.span` across its rebuild → `source` rides along
  for free (it's inside `Span`). No ANF change.
- The 2 previously-missed alloc sites (ConstStr in string-literal match patterns;
  the `?`-desugar MakeVariant) + the `variant_ctor` over-intern + the test-row
  heuristic are ALREADY fixed (`858b2ec2f`).
- The AOT-build parallelism Mutex in `tests/alloc_profile.rs` is committed.

### Verification (the regression tests that prove THIS fix)

- Strengthen `alloc_profile_prelude_allocs_are_not_fabricated_locations`: use a
  **>44 KB user file** (the residual's worst case — pad with a big comment or many
  fns) and assert prelude allocs STILL resolve to `<std>:line` (not a user line).
  This is the test the interim FAILS and the source-id fix PASSES.
- Add a **multi-module** test: a `mod m;` whose fn allocates → the alloc resolves
  to `m.gcr:line:col` (falls out of source-id for free).
- Re-run: gcrust-rt, alloc_profile (parallel), aot, broad suites, + the gc-stress
  gate (`GCR_GC_VERIFY=1 GCR_STRESS_ITERS=6 cargo test --release --test
  concurrency_stress`).

## Then → P1 production-correct (re-sign-off) → P2 (DWARF line tables)

P2 is scoped in `docs/DEBUGGER_DESIGN.md` (appendix): broaden span coverage to
statement-level + inkwell `debug_info` emission, AOT-focused, line-tables-always /
full-DWARF-under-`--debug`, ride LLDB + `lldb-dap`. Build P2 ONLY after the
source-id foundation is verified (DWARF resolves spans — it must inherit correct
locations, not the residual).

## Outstanding (non-blocking, noted)

- Cross-binary test race: a full parallel `cargo test` can flake `aot_fib` because
  multiple test processes each `cargo build -p gcrust-rt` (shared staticlib). Each
  suite passes alone; `cargo build -p gcrust-rt` first avoids it. Pre-existing
  infra, orthogonal to the debugger.
- Profiling refinement (later, per Leader): real-location is correct for DWARF;
  attributing a prelude alloc to the USER call-site (inline-frame attribution) is
  a separate future profiling refinement.
