# Debugger P2 — DWARF line tables (first-increment plan)

> **STATUS: FIRST INCREMENT DONE + GREEN** (both parts implemented + verified).
> `gcr build` AOT binaries emit DWARF line tables and **lldb does source-level
> stepping of gc-rust code** — breakpoint `dw.gcr:4` halts at `frame #0 main at
> dw.gcr:4:11`, program exits correctly. Per-source DIFiles work (prelude allocs →
> `<std>`). Verified: dwarf 2/0, alloc_profile 6/0, lib 158/0, gcrust-rt 108/0,
> all 11 integration suites green, gc-stress gate green. Two real bugs caught by
> LLVM verify + fixed during the build: (1) the function debug location must be
> set BEFORE the prologue (else prologue instrs inherit the previous function's
> scope → "wrong subprogram"); (2) `CoreFn.span` was ADDED (the body-scan
> derive-first was insufficient — prelude fns' spans nest in branches) so each
> function's DISubprogram lands in its real source file. `dsymutil` runs in
> `build_executable` (macOS) so the executable is debuggable. NEXT: P3 (full DWARF
> — locals/params/lexical scopes under `--debug`; reflection pretty-printers).
> **P3 first increment is now DONE** — `gcr build --debug` emits scalar
> local/param variable DIEs and lldb `frame variable` shows them with correct
> values. See `docs/DEBUGGER_P3_PLAN.md`.


Builds on P1 (span-threading) + P1b (source-id-per-span, production-correct,
double-gated SHIP @183e130d6). Goal: `gcr build` AOT binaries carry DWARF **line
tables** so LLDB does source-level stepping (`break file:line`, `step`, `source
list`). Full DWARF (locals/params/lexical scopes) is P3 under `--debug`.

API verified present in the inkwell fork (LLVM 21, `9a3b397`):
`module.create_debug_info_builder(...) -> (DebugInfoBuilder, DICompileUnit)`,
`di.create_file(filename, directory) -> DIFile`,
`di.create_function(scope, name, linkage, file, line, ditype, is_local, is_def, scope_line, flags, is_opt) -> DISubprogram`,
`di.create_subroutine_type(file, ret, params, flags)`,
`di.create_debug_location(ctx, line, col, scope, inlined_at) -> DILocation`,
`builder.set_current_debug_location(loc)`, `fv.set_subprogram(sp)`, `di.finalize()`.

## Part 1 — broaden spans to statement level (lower)

P1 spans only the 5 construction/alloc nodes (enough for alloc-sites; too sparse
for stepping). Reuse the P1 mechanism (`Mono::intern_span` + `CoreExpr::at`; ANF
already preserves `.span`) — NO new IR shape, just more `.at()` sites:
- each `CoreStmt::{Let, Expr}`'s expr (statement position),
- block tail exprs,
- key steppable exprs: `Call`, `Return`, assignments.
Goal: every source line with code maps to ≥1 instruction's debug location, so
`step` advances line-by-line instead of jumping.

## Part 2 — emit DWARF line tables (codegen)

Per-compilation (once):
1. `module.create_debug_info_builder(allow_unresolved=true, DWARFSourceLanguage::C
   (or Rust), filename=sources[0].path basename, directory, producer="gcr",
   is_optimized=<O2?>, flags="", runtime_ver=0, split_name="",
   kind=DWARFEmissionKind::LineTablesOnly, ...)` → `(di, cu)`.
2. Module flags: `"Debug Info Version" = 3`; on macOS also a `"Dwarf Version"`
   flag, so the backend emits DWARF into the Mach-O/ELF object.
3. **Per-source `DIFile` (the requirement P1b's review surfaced):** build a
   `Vec<DIFile>` indexed by `SourceId` — `di.create_file(entry.path, dir)` for each
   `prog.sources` entry. A node's span resolves to the RIGHT DIFile via its
   `SourceId`, so a prelude/`mod` frame in a backtrace shows its real file, not the
   user file. (CU primary file = `sources[0]`, the user source.)

Per function (`define_fn`):
4. Determine the function's source = the `SourceId` of its body spans (all nodes
   in one function share one source — a function body is lexed from a single
   source; cross-source only happens across function boundaries, each with its own
   DISubprogram). Fallback: user (0) if the function has no spanned node.
   *(Open: CoreFunc has no function-level span today — derive from the first
   spanned body node, or add a `CoreFunc.span` in Part 1. Lean: derive first; add
   the field only if needed.)*
5. `sub_ty = di.create_subroutine_type(difile, None, &[], DIFlags::zero())`
   (line-tables-only: no real param types yet — P3).
6. `sp = di.create_function(cu_as_scope, mangled_name, Some(linkage), difile,
   def_line, sub_ty, is_local=true, is_def=true, scope_line=def_line,
   DIFlags::zero(), is_optimized=<O2?>)`; `fv.set_subprogram(sp)`.

Per node (`gen_expr`/`gen_stmt`):
7. For a node with a real span: `prog.span_location(span)` → `(label, line, col)`
   (NOTE: P1b renamed `span_line_col` → `span_location`, which also returns the
   source label — use the `SourceId` to pick the scope's DIFile; within one
   function the source is constant so scope = the function's `DISubprogram`).
   `loc = di.create_debug_location(ctx, line, col, sp.as_debug_info_scope(), None)`;
   `builder.set_current_debug_location(loc)` BEFORE emitting the node's
   instructions. Clear/repoint as nodes change.

Finalize:
8. `di.finalize()` after ALL functions, BEFORE `module.verify()` / optimize.

## Scope boundary

- **AOT only** (`gcr build`): DWARF lands in the emitted object; `lldb` /
  `llvm-dwarfdump` read it. JIT-code DWARF needs the LLDB JIT registration
  interface — deferred (P3+).
- **Line tables ALWAYS** (best-effort even on O2 — LLVM preserves debug locations
  through the pipeline). Full DWARF (locals) = P3 under `gcr build --debug` (-O0).

## Verify

- A test: `gcr build` a small program → `llvm-dwarfdump --debug-line <obj>` shows
  the source file + expected line rows (grep for `<file>` + line numbers).
- Multi-source: a program allocating via the prelude → dwarfdump shows BOTH the
  user file AND `<std>` (prelude) line rows (proves per-source DIFiles).
- `lldb` smoke (manual / scripted): `break set -f X -l N`, run, `source list`,
  `step` follows source lines.
- Re-gate: gc-stress (DWARF is metadata; must not perturb codegen correctness).

## Risks / watch

- **Location↔scope file consistency (rests on NO IR inlining):** LLVM asserts a
  debug location's scope subprogram file matches the location's file unless the
  location is marked inlined. Safe for the debug build: one function = one source =
  one DIFile = one DISubprogram, no IR inlining → the invariant holds faithfully.
  CAVEAT (per Leader): if **O2 inlines** a callee's instructions (ANOTHER source —
  e.g. a prelude fn inlined into a user fn) into the caller's DISubprogram, the
  callee's locations then point at a different file than their scope → the
  one-function-one-source insight breaks. Proper handling = emit a
  `DW_TAG_inlined_subroutine` and set the inlined locations' `inlined_at` — a LATER
  refinement, NOT the first increment. For now this is the explicit best-effort
  caveat of "line-tables always (even on O2)": the debug build (no/low inlining) is
  faithful; O2-inlined frames may mis-scope until the inlined_at refinement lands.
- **O2 + line-tables:** verify locations survive the O2 pipeline (they should;
  test on an O2 build, not just -O0) — and watch for the O2-inlining caveat above.
- **macOS dSYM:** `llvm-dwarfdump` may need the `.o` directly (DWARF can live in
  the object pre-link, or a `.dSYM` post-link). Test against the object.
