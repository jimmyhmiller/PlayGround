# Source-level Debugger — design proposal (DRAFT, for review)

The headline tooling-vs-JVM **breadth** gap. From the goal self-audit: gc-rust is
"true in precision, false in breadth (~35-40%)" vs the JVM; the single biggest
absent thing a developer relies on is a **debugger** (jdb / JDWP / IDE). This is
the gating piece for "tooling ≥ JVM". Status: **design only — no code.**

Built on the shared **span-threading** prerequisite, which also upgrades Target-1b
allocation-site profiling from function+type to **file:line:col** — one
foundation, two payoffs.

---

## 1. What already exists (substrate — reuse, don't rebuild)

- **Source spans ALREADY exist in the AST.** `src/ast.rs` carries `span: Span` on
  ~22 node kinds; the lexer + parser produce them; `src/diag.rs` already renders
  caret diagnostics from them. They are **dropped at lowering** — `src/core.rs`
  has *zero* spans (confirmed: the Core IR is span-free, which is exactly why 1b's
  alloc-sites are function+type, not line-precise). So span-threading is "stop
  discarding spans we already parse," not "invent source locations."
- **inkwell 0.5 has the `debug_info` module** (`DebugInfoBuilder`) — DWARF line
  tables + subprogram/variable DIEs emit straight from the LLVM backend.
- **Reflection table** (`TypeMeta`/`ValueMeta`, `gc::reflect`) — decodes a heap
  object to `Point { x: 3, y: 4 }` with source names. This is the differentiator:
  paired with DWARF it lets the debugger render *language values*, not raw words.
- **Precise frame roots** — every GC-typed local is a frame root slot
  (`FrameOrigin`, `walk_gc_frames`); `FrameOrigin` already carries the function
  `name`. Locals inspection needs only per-local *names/types/slots* added on top.
- **Safepoint mechanism + STW** (`pause_world`, the poll protocol) — the clean
  place to stop the world for consistent inspection (same discipline the heap
  explorer uses).

So the debugger is largely "thread spans through, emit DWARF from them, and pair
DWARF with the reflection table" — not green-field.

---

## 2. The shared prerequisite: span-threading (P1)

Thread the **existing** AST spans through `lower → core → codegen`:
- Add a `Span` (or a compact `SpanId` into a side table) to the Core IR nodes
  that matter for debugging (statements, calls, allocations, binds, returns).
- Lowering stops dropping spans; codegen has a current-span to attach to emitted
  instructions.

**Two payoffs from this one foundation:**
1. **DWARF line tables** (P2) — source-level stepping + line breakpoints.
2. **Line-precise alloc-site profiling** — upgrades 1b's `(function, type)` site
   key to `file:line:col`, *for free* once spans reach codegen's alloc emission
   (the 1b `AllocSite` already has the slot; today it's labelled function+type
   because Core had no span). A concrete, **independently testable** payoff that
   lands before any DWARF work — the right first proof that span-threading works.

Span-threading is the gating dependency; do it first, prove it via the
alloc-site line:col upgrade, then build DWARF on it.

---

## 3. DWARF emission (P2/P3)

From the LLVM backend via inkwell `debug_info`:
- **P2 — line table:** a compile unit + per-function subprogram DIE + line/column
  attached to each instruction (from the threaded spans). Gives LLDB/GDB
  source-level stepping (`step`/`next`/`finish`) and `break file:line`.
- **P3 — variables:** local variable DIEs with location expressions pointing at
  each local's alloca / frame slot (the slots already exist for GC roots; add
  name + type from span-threading). Gives `frame variable` / locals in the IDE.

**Render language values, not raw words.** DWARF alone shows a heap local as a
raw pointer. Pair it with the reflection table: ship **LLDB pretty-printers**
(Python formatters) that, given an object pointer, read its header `type_id`,
look up `TypeMeta`, and format `Point { x: 3, y: 4 }` — reusing the *exact* logic
`gc::dump::render_object` already implements. This is the JVM-grade inspection UX,
delivered through a mature debugger we don't have to build.

---

## 4. Debugger frontend — decision for review

- **(A) Ride LLDB/GDB via DWARF (recommended).** Emit standard DWARF → get
  breakpoints, stepping, backtraces, conditional breakpoints, watchpoints, and
  IDE integration *for free*; add reflection-paired pretty-printers for
  language-value rendering. Least to build, immediately JVM-grade UX.
- **(B) Custom in-process debugger** on the safepoint mechanism + reflection.
  Full control + native language-value rendering, but reinvents the entire
  frontend (breakpoint insertion, stepping, expression eval). Large.

Recommendation: **(A)**. Build the standard substrate (DWARF + pretty-printers);
only consider (B) for capabilities LLDB can't express (e.g. time-travel, the
"beyond JVM" stretch).

---

## 4b. DAP / IDE protocol — which subset for v1

The modern IDE debug protocol is **DAP** (Debug Adapter Protocol; VS Code et al.;
JDWP is the older Java-specific equivalent). The key realization: with the
**LLDB-ride (A)** approach we do **not implement DAP at all** — LLDB ships
`lldb-dap` (formerly `lldb-vscode`), a DAP server over LLDB. So the chain is:

```
gcr build --debug  →  DWARF + reflection pretty-printers
                   →  LLDB (+ our Python formatters)
                   →  lldb-dap  →  any DAP IDE (VS Code, etc.)
```

We get the **full** DAP capability set (launch/attach, setBreakpoints,
conditional + hit-count breakpoints, stepIn/Out/Over, continue, pause,
stackTrace, scopes, variables, evaluate, watchpoints) **for free** from lldb-dap
— gated only by what our DWARF + pretty-printers express. So the "v1 DAP subset"
question reframes to **"which debug *capabilities* does v1's DWARF + formatters
light up"**, not "which DAP messages do we hand-write":

- **v1 (after P2+P3):** breakpoints (line + function), stepping (in/out/over),
  stackTrace/backtrace, scopes + **variables rendered as language values** (the
  reflection pretty-printers), continue/pause. That is already a JVM-grade core.
- **v1.1 (P4):** conditional breakpoints, watchpoints, `evaluate` of simple
  expressions (LLDB-native; expression eval over language types is the deepest
  piece and may stay raw-LLDB initially).

Only if we chose the **custom frontend (B)** would we hand-write a DAP subset
(setBreakpoints / threads / stackTrace / scopes / variables / continue / next /
stepIn / stepOut — the ~10 core requests). That is a large surface; the LLDB-ride
avoids it entirely. **Recommendation: ride lldb-dap; implement no DAP ourselves.**

## 5. The moving-GC ↔ debugger subtlety (flag for review)

A moving GC complicates debugging: break → inspect a local holding a GC pointer →
continue → a GC relocates the object → a cached address is now stale. Mitigations,
all leveraging what we have:
- **Inspect at safepoints.** Between safepoints the GC can't run; a debugger stop
  that lands on (or is quantized to) a safepoint sees a stable heap. The poll
  protocol already defines these points.
- **Re-read through frame slots, never cache raw addresses.** The collector
  updates frame root slots on relocation; a pretty-printer that reads the local's
  *slot* (not a snapshot of the pointer) always sees the current address. DWARF
  variable locations point at the slot, so this is natural.
- This is a real design constraint, not a blocker — precise roots + safepoints
  make it tractable. Worth explicit verification (break, force GC, continue,
  re-inspect) when P3 lands.

---

## 6. Phasing (each independently reviewable; P1 has a testable payoff before DWARF)

- **P1 — span-threading.** Carry AST spans through lower→core→codegen. **Proof +
  payoff: upgrade 1b alloc-sites to file:line:col** (testable on its own, no DWARF
  yet). The gating foundation.
- **P2 — DWARF line tables.** Source stepping + `break file:line` in LLDB. Verify:
  `lldb gcr-built-binary`, set a line breakpoint, step, confirm source lines.
- **P3 — DWARF locals + reflection pretty-printers.** Inspect locals as language
  values (`Point { x: 3, y: 4 }`) via LLDB formatters reusing `render_object`.
  Verify: break, `frame variable`, confirm decoded values; break→GC→continue→
  re-inspect (the §5 moving-GC check).
- **P4 — richer debugging.** Conditional breakpoints / watchpoints (LLDB-native),
  and heap navigation at a stop (reuse the heap explorer's snapshot + heap-diff,
  and `visit_roots` for the live root set at the breakpoint).

P1 alone advances the goal (line-precise profiling) with no debugger risk; the
big-value P3 (language-value inspection) is where "≥ JVM inspection" is realized.

---

## 7. Open questions for review

1. **Frontend: LLDB-ride (A) vs custom (B)?** Recommendation A. Confirm.
2. **DWARF always vs under a flag?** Optimized (O2) code + DWARF is lossy
   (inlining/reordering). Options: emit DWARF always (accept some opt-induced
   imprecision), or a `gcr build --debug` (-O0 or opt-preserving-debuginfo) mode.
   Likely: a debug build mode for faithful stepping; DWARF line tables on O2 as
   best-effort.
3. **Span granularity:** statement-level (cheaper, good-enough stepping) vs
   expression-level (precise alloc-sites + finer stepping). Lean expression-level
   at least for the alloc emission sites (the 1b payoff), statement-level
   elsewhere if it simplifies.
4. **Pretty-printer delivery:** LLDB Python formatters reading the reflection
   metadata (from where — a side file the build emits, or the in-binary reflect
   blob?) vs a `gcr inspect`-style native renderer the debugger shells out to.
5. **Scope of P1's span model:** full `Span` on every Core node (heavier) vs a
   `SpanId` side table keyed by node (lighter, keeps Core IR lean). Lean toward a
   side table to keep Core small.

---

*Prepared by Rust-gc for Leader review. Design-first; no implementation started.
Builds on span-threading (the shared prereq with line-precise 1b profiling),
inkwell `debug_info` (DWARF), and the existing reflection table (language-value
rendering).*

---

## Appendix: P1 status + P2 implementation scope

**P1 — DONE** (committed `68daae9e5`, in adversarial review): SpanId side-table
threaded AST→lower→ANF→codegen; the file:line:col alloc-site upgrade proves it
(two same-(function,type) allocs at different lines → distinct sites). Verified
green incl. the gc-stress gate. (Details in the commit / the alloc-profile test.)

**P2 — DWARF line tables (scoped; build after P1 sign-off).** Two parts:

1. **Broaden span coverage to statement level (lower).** P1 attaches spans to the
   5 construction/alloc nodes only — enough for alloc-sites, but a *sparse* line
   table makes stepping jumpy. Reusing the P1 mechanism (`Mono::intern_span` +
   `CoreExpr::at`; `CoreStmt::{Let,Expr}` already wrap a `CoreExpr` that has the
   `.span` field; ANF already preserves it), attach spans at statement-position
   exprs (each `CoreStmt`'s expr + block tails) and the key steppable exprs
   (calls, returns, assigns) so every source line maps to instructions. No new IR
   shape — just more `.at()` call sites in `lower`.

2. **Emit DWARF via inkwell `debug_info` (codegen).** API confirmed present in
   inkwell 0.5:
   - `codegen()`: `module.create_debug_info_builder(...)` → `(DebugInfoBuilder,
     DICompileUnit)` (producer "gcr", file = `prog.sources[0].path` — the user
     source; each prelude/`mod` source needs its own `DIFile`, keyed by `SourceId`
     via the SourceMap, so spans resolve to the right DWARF file — `optimized` = true
     on the O2 paths). Add module flags: `"Debug Info Version" = 3` (+ a Dwarf
     Version flag on macOS) so the backend emits DWARF into the object.
   - `define_fn()`: `di.create_function(...)` → a `DISubprogram` (with a minimal
     `create_subroutine_type`); `fv.set_subprogram(sp)`.
   - `gen_expr`/`gen_stmt`: for a node with a real span, resolve via
     `prog.span_line_col` → `di.create_debug_location(ctx, line, col, scope, None)`
     → `builder.set_current_debug_location(loc)` before emitting that node's
     instructions. Scope = the function's `DISubprogram` (lexical blocks later).
   - `di.finalize()` after all functions, before `module.verify()`/optimize.

3. **3-tier (per the approved decisions):** P2 emits **line tables always**
   (best-effort even on O2 — LLVM preserves debug locations through the pipeline);
   **full DWARF (locals/params/lexical scopes) is P3** under `gcr build --debug`
   (-O0 / opt-preserving for faithful stepping).

4. **Scope boundary:** P2 targets **AOT binaries** (`gcr build`) — DWARF lands in
   the emitted object naturally and `lldb`/`llvm-dwarfdump` read it. JIT-code
   DWARF needs the GDB/LLDB JIT debug-registration interface — deferred (P3+).

5. **Verify:** `llvm-dwarfdump --debug-line` on a `gcr build` object shows the
   source file + line rows; an `lldb` smoke (set a line breakpoint, `source
   list`, step) follows source. A test greps the dwarfdump output for the file +
   expected lines.

*P2 is scoped, not built — it stacks on P1, so it lands only after P1 is
signed off (the foundation must be verified before DWARF is built on it).*
