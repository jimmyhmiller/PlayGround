# Deleting the comptime interpreter — ✅ DONE 2026-07-22

**The deletion is complete.** `comptime.coil`'s tree-walker (`eval`/`eval-seq`/
`eval-args`, the quasiquote evaluator, 14 interp-only helpers — ~650 lines), the
`COIL_META` and `COIL_STAGE_MACROS` flags, `parity.sh`, and the guide's two-engines
section are gone. The compiled engine is the only engine for macros, `(meta …)`
generators, checkers/transforms, and `(comptime …)`/`(const …)` folding. Macro
staging is LAZY (an engine miss stages that one qual on demand; unstageable = hard
error), measured to old-path performance: self-build equal-or-faster, corpus warm
within noise, and emitted IR byte-identical pre-vs-post deletion across the corpus.
The nollvm-verification precondition was consciously WAIVED (its rebootstrap was
already broken by the stale seed, independent of this work); `main_a64.coil` still
typechecks, and its engine path should be gated when the seed is refreshed.
Deliberate semantic changes, all located errors instead of silent behavior: pointer
results can't be materialized as compile-time constants; a non-terminating comptime
now hangs (fuel died with the interpreter) exactly like a non-terminating macro
always did under the engine. `docs/DECISIONS.md` decision 7 is DONE. The rest of
this file is the historical record of how it got there.

---

# (historical) status, architecture, and the plan to finish

_Decision 7 of `docs/DECISIONS.md` ("delete the comptime interpreter"), written up so the
remaining work can be executed as a focused session. This is the one reviewed item still
open. Everything else from the roadmap and the wasm/scry work is landed._

## The goal (mostly aesthetic — but the hybrid has one real bug, see "The comptime-UB hole")

Coil runs compile-time code (`(comptime E)`, `(const …)`, `(meta …)`, and macro bodies)
two different ways:

- **The tree-walking interpreter** (`comptime.coil`: `eval` / `eval-seq` / `eval-args`) — a
  strictly *weaker* sublanguage: no generic calls, no `sizeof`/`alignof`/`offsetof`, no
  strings, no fn pointers, no bitfield ops. It executes a form directly, with no
  compile/check/lower/build.
- **The compiled metaprogram engine** (`metaengine` / `metalower` / `metahost` / `metashim` /
  `comptime_eval`) — compiles the metaprogram (plus its dependencies) into a dylib,
  `dlopen`s it, and calls it as native code. Full language power.

The point of the deletion is **one engine, no secretly-weaker path** (closes `mac-12` and
`diag-4`). It is *largely* a purity/maintainability goal — the interpreter itself is not
broken. But the **hybrid** is: the routing between the two engines has a live correctness
hole (below), so this is not purely aesthetic. Weigh that against the risk before committing.

## The comptime-UB hole (live bug, verified)

The interpreter is currently the only thing that catches compile-time integer UB. `ct-bin-int`
(`comptime.coil:174-178`) rejects div/rem by zero; the compiled fold has no such check, and
arm64 `sdiv`-by-zero yields **0 with no trap**. The `comptime-cap-gap?` discriminator is
supposed to keep semantic errors on the interpreter, but it keys off the *first* error the
interpreter reports — so a site that is **both** a capability gap **and** a division by zero is
routed to the compiled engine, which folds it silently.

Verified against the shipped compiler (default settings, no flags):

    (defn main [] (-> i64) (comptime (idiv (sizeof i64) 0)))   ; → exit 0, folds to 0, NO error
    (defn main [] (-> i64) (comptime (idiv (sizeof i64) 2)))   ; → exit 4, correct
    (const z (idiv 1 0))                                       ; → correctly errors (interp owns it)

Consequence for the plan: **step 6 cannot be "flip `fold-expr` once readback is total."**
Readback totality is necessary but not sufficient. Removing the interpreter removes the only
comptime-UB check in the compiler, converting a diagnostic into a silently wrong constant.
The compiled fold must carry its own div/rem-by-zero guard *before* step 6 flips.

⚠ The guard must be **integer-only**. `EBin`'s `op` is ambiguous between int and float
(`parser.coil:446`: `fdiv`=3, `frem`=4 collide with `idiv`=3, `irem`=4), and `ct-bin-float`
(`comptime.coil:166-170`) has no zero check *by design* — comptime float division to infinity
is legal and must stay legal. Classify by operand type, never by op number alone.

A second, lesser gap: the interpreter has a **fuel budget** (`comptime.coil:458`, `:585`) that
catches non-terminating comptime code. A compiled thunk simply hangs. Not a wrong answer, but
it is coverage that disappears with the interpreter.

## What has already landed (do NOT redo)

| Step | id | Commit(s) | What it did |
|------|-----|-----------|-------------|
| 1 | mac-8 | `433631b1e` | Route `(comptime E)`/`(const …)` through the compiled engine. `fold-expr`'s single `EComptime` seam tries the interpreter FIRST; only on a *capability gap* (the interp's own "…isn't supported yet" wording) does it route the site through the compiled engine via `ct-fold-hook` (wired by `main`/`main_a64`'s `register-comptime-fold!`). A genuine semantic error (div-by-zero) is *intended* to be left to stand, the discriminator being the interp's wording. ⚠ **This safety claim is false — see "The comptime-UB hole" below.** |
| 2 | mac-12 | `ec0d40ecd` | `export-c` on the arm64 backend (`g-register-sigs!` / `g-export-c-sym`; by-value struct param is a located error via `g-export-needs-thunk`). `main_a64` registers `meta-build-obj-a64`, so the LLVM-free compiler builds metaprogram dylibs with the arm64 backend and the compiled engine is its default. |
| 3a | — | `fa2ec42e7` | Aggregate/string comptime readback on the compiled engine: a write-through `(ptr T)` thunk entry + walk the C struct/`(slice u8)` layout by field offset (`comptime_eval.coil`). |

The compiled engine (`comptime_eval.coil`) today: recovers the site's checked type by nid,
builds a minimal closure sub-program of what `E` calls plus a synthetic
`(defn coil.ct.thunk [] (-> T) E)` exported as C symbol `coil_ct_thunk`, monomorphizes +
builds + `dlopen`s it (its own raw build, no metashim handshake — comptime has no code ops),
runs the entry, and reads the result back. **Scalars** (int/bool/f64) come out of the return
register; **3a** added a write-through `(ptr T)` path for readable aggregates.

## Why it is not done: three entangled reroutes + a coverage gap

`eval` has **three callers**, and all four deletion targets (the evaluator, the `COIL_META`
flag — 18 refs, `metaprog-poc/compile-and-run/parity.sh`, and the `guide.coil` interp
mention) stay reachable until all three are off `eval`:

### (A) `fold-expr` readback coverage gap — `comptime_eval.coil`
The readback (`read-value at buf+off`, `comptime-fold-one`) declines **sums, `f32`,
raw-pointer aggregates, non-default-layout (non-`LC`) structs, and non-`u8` slices**
(`comptime_eval.coil:283`, `:365`). Because `fold-expr` still tries the interpreter FIRST,
this is harmless *today* (those shapes fold on the interpreter). But you cannot flip
`fold-expr` to compiled-only until the readback covers **every shape `build-content`
produces**, or you regress the monomorphic aggregate comptimes the interpreter folds now.

**Blocking prerequisite — the `type-bytes` divergence.** Sum readback needs the host reader
to walk the C layout by field offset, but the two backends disagree on sizing:
- `codegen.coil::type-bytes` (LLVM) align-8's **array elements**: `(TArray el n) →
  align8(type-bytes el) * n` (`codegen.coil:210`), and align-8's every struct field
  (`:223/:235/:246`).
- `codegen_a64.coil::g-type-bytes` (arm64) does **not** align-8 array elements: `(TArray
  elem n) → g-type-bytes(elem) * n` (`codegen_a64.coil:211`).

So `array-of-i32` is `8*n` under LLVM, `4*n` under arm64. Latent today (the self-build never
hits an array-in-aggregate at a comptime readback site), but a single host-side reader cannot
match both dylib backends until this is reconciled. **First determine which is correct**
(LLVM's native `[N x iM]` does *not* pad elements to 8, so the arm64 form is the ABI-true one
— but `type-bytes`' result also drives the compiler's own alloca/memcpy sizing, so changing
it has layout implications that must be traced, not assumed), then make them agree. This is a
**standalone correctness fix**, independently green, and worth doing regardless of the deletion.

### (B) `run-metas` + `finish-macro` — `comptime.coil:2048`, `expander.coil:291`
- **`run-metas`** (the `(meta …)` generator path) still calls `eval`. `elaborate-metas`
  (`comptime.coil:2114`) builds a **post-check `Program` closure**; the macro path does its
  metashim injection at the **pre-check `TaggedForm`/`LS`** stage. Rerouting `run-metas`
  through the compiled engine needs the metashim functions injected+checked+lowered into the
  meta sub-program (`metalower` emits `metashim.mp-*` calls for every quasiquote/code-op) —
  a representation mismatch that requires restructuring `elaborate-metas`. Gate-visible via
  `selfhost/oracle/features/meta_stage3.coil`.
- **`finish-macro`'s `eval-seq` fallback** (`expander.coil:291`) is the **hard** one. A
  compiled fast-path already exists (`metaengine.coil:404` "run the compiled entry instead of
  `eval-seq`"; `finish-macro` routes every entry through it once built — `expander.coil:735`),
  but `eval-seq` is the fallback for the **definition-time expansion tower**, which expands
  macros *before* a compiled engine can exist for them (compiling a macro needs it checked,
  checking needs its own macros expanded — chicken-and-egg). Removing the fallback requires
  an **on-demand / incremental engine build mid-tower**. This is the genuine architectural
  change, and the piece with direct fixpoint risk.

### (C) Relocate `CtVal` / `CtCtx` — 7 modules
`comptime.coil` defines `CtVal`/`CtCtx`; `metalower`, `comptime_eval`, `expander`,
`metaengine`, `metahost`, `metashim` reuse them *through* `comptime.coil`. Since `:use *`
does **not** re-export, once the evaluator is gone each of those modules needs an explicit
import of a new home module (e.g. `ctval.coil`). Mechanical, independently green, but wider
than it looks — do it as its own commit.

## Execution plan (dependency-ordered)

**Phase 1 — bankable prerequisites — ✅ ALL LANDED.**
`type-bytes` reconciliation `b115c4e03`; readback extended by `fa2ec42e7` (aggregates/strings)
then `223fe8a06` (sums, `f32`, packed/explicit layouts); `CtVal`/`CtCtx` → `ctval.coil`
`797b957b9`. Kept below for the record — do not redo.

1. ✅ `b115c4e03` — **Reconcile `type-bytes` / `g-type-bytes`.** Trace how each array/field result drives real
   layout; make the two backends agree (likely: drop the array-element `align8` in
   `codegen.coil`, matching LLVM's native array layout — but VERIFY via a struct-with-array
   repro on both backends, and expect an oracle-ref change that you can explain line-by-line).
   Fixpoint + all 7 gates green.
2. ✅ `fa2ec42e7` + `223fe8a06` — **Extend the readback** (`comptime_eval.coil` `read-value`/`comptime-fold-one`) to cover
   sums, `f32`, raw-ptr aggregates, non-`LC` structs, non-`u8` slices — additive, `fold-expr`
   still interp-first, so behaviour is unchanged and it is green. (It is dead until Phase 2's
   flip; unit-exercise it with a temporary hook or a targeted test to prove correctness.)
3. ✅ `797b957b9` — **Relocate `CtVal`/`CtCtx`** to `ctval.coil`; add explicit imports. (It was
   5 users, not the 7 estimated here.)

**Phase 1b — the comptime-UB guard — ✅ IMPLEMENTED, verified, UNCOMMITTED (one file:
`comptime_eval.coil`).** Verified independently against the pre-change compiler:

| case | before | after |
|---|---|---|
| `(comptime (idiv (sizeof i64) 0))` | exit 0, silent | `comptime: division by zero` |
| `(comptime (irem (sizeof i64) 0))` | exit 8, silent | `comptime: remainder by zero` |
| div inside a *called* fn | exit 0, silent | `comptime: division by zero` |
| computed zero `(isub 2 2)` | exit 0, silent | `comptime: division by zero` |
| `(comptime (idiv (sizeof i64) 2))` | 4 | 4 |
| f64 `(fdiv … 0.0)` → +inf | 255 | 255 |
| multi-site fold | 22 | 22 |

Gates green (gate-full 60/0, arm64 gate-run 56/0, gate-cli PASS, diag/ast/checked/mono/
resolved clean); `stage2.o == stage3.o` fixpoint holds on both rebootstraps; no reference
files changed. Design note: when BOTH operands of an ambiguous `idiv`/`fdiv`-tagged node have
indeterminate types the compiled fold DECLINES rather than guess — guarding could corrupt a
legal float divide, not guarding could silently fold a bad integer one. Never triggered across
the corpus.

3b. **Guard div/rem-by-zero in the compiled fold.** Inject a status cell + a `coil.ct.nz`
   divisor check into the comptime closure sub-program and rewrite every *integer* `idiv`/
   `irem`/`udiv`/`urem` across **all** its function bodies (the interpreter evaluates the whole
   call graph, so a division inside a callee counts) to guard the divisor only — single
   evaluation, no `ELet` synthesis, no rhs duplication. `cte-run` reads the cell after the
   thunk and returns the interpreter's exact wording. Fixes the live bug above and is a
   prerequisite for step 6. Float behaviour must be provably unchanged.

**Phase 2 — the entangled landing (must be coordinated; NOT independently green):**
4. **Reroute `run-metas`** through the compiled engine (restructure `elaborate-metas` to the
   pre-check metashim-injection representation). Verify `meta_stage3.coil`.
5. **Reroute `finish-macro`** — make the compiled fast-path total by building the engine
   on-demand/incrementally during the expansion tower; remove the `eval-seq` fallback.
6. **Flip `fold-expr`** to compiled-only. Requires readback total **and** Phase 1b's UB guard
   landed — otherwise this step converts every comptime div/rem-by-zero diagnostic into a
   silently wrong constant. Deleting `comptime-cap-gap?` (and its "supported yet" string
   match) is part of this step, not step 7.
7. **Delete** `comptime.coil`'s `eval`/`eval-seq`/`eval-args` tree-walker, the `COIL_META`
   flag (18 refs — the driver plumbing, `metahost` reentry, gate hooks), `parity.sh`, and the
   `guide.coil` interp mention. Update `docs/DECISIONS.md` decision 7 to DONE.

## Step 5 is much smaller than this doc assumed — MEASURED

`finish-macro`'s `eval-seq` fallback was instrumented and run over the corpus (164 files:
`examples/`, `lib/`, `apps/`, `metaprog-poc/`, `mini-scheme/`, `bench/`).

- **Compiler self-compile — the bootstrap path, the fixpoint-risk one: 2 expansions.**
  Both `slice.dbg-slice-get` / `slice.dbg-slice-set`.
- **Whole corpus: 418 expansions but only 10 DISTINCT macros.**
  `slice.dbg-slice-set` 153, `slice.dbg-slice-get` 153 (exactly 2 per file — bundled stdlib),
  `mandelbrot.msg` 77, `control.when` 14, `control.for` 10, `control.while` 4, `fmt.fmt` 3,
  `control.cond` 2, `tower_msg_test.msg` 1, `slice.dbg-subslice` 1.

**The cause is STAGING, not a real cycle.** Traced ordering on a trivial program:

    CTINTERP slice.dbg-slice-get engine-active=0
    CTINTERP slice.dbg-slice-set engine-active=0
    CTMARK   engine-up -> run-expand

Round 0's `check-program` fails because the closure sub-program contains ordinary functions the
macros call — `slice-get` (`lib/slice.coil:106`) whose body holds the unexpanded macro call
`(dbg-slice-get s i)`. The tower expands it on the interpreter; round 1 then checks and the
engine comes up. But `dbg-slice-get` (`lib/slice.coil:58`) does NOT depend on `slice-get`: its
body uses only builtins, `debug-checks?` and quasiquote, so it checks standalone TODAY. The
cycle is purely an artifact of ONE all-or-nothing check over the whole qual closure — a single
unexpanded call poisons the batch and nobody gets compiled.

**Dependency depth is ~0.** `or`/`and` are compiler builtins, not lib macros, so the
`dbg-slice-*` bodies have no macro dependencies; `while` (`control.coil:66`) and `for` (`:78`)
use only builtins plus code-ops the engine handles natively.

### ⚠ BUT: per-macro staging alone does NOT work — PROTOTYPED AND FALSIFIED

The obvious fix ("check each qual's closure in isolation, stand up a partial engine for the ones
that check standalone") was built and measured. **Result: zero effect.** Identical CTINTERP
output with staging on vs off — same 2 fallbacks on `examples/fib.coil`, engine still inactive.

**Root cause — the isolation does not isolate.** `closure-funcs` does not merely take the
transitive callees of the seeds: it *additionally* seeds the callees of **every impl method and
every const value in the whole program** (`comptime.coil` ~1757-1775), because
`closure-subprogram` retains every impl and const unconditionally. So `slice.slice-get` — whose
body is the unexpanded `(dbg-slice-get s i)` — is dragged into **every** sub-program no matter
how narrow the seed. Checking `dbg-slice-get` "in isolation" still checks `slice-get`, which
still fails, so no macro ever qualifies as standalone.

So the batching is not the (only) problem; the **sub-program construction** is. A working step 5
must additionally prune impls/consts from macro sub-programs, or make the check tolerate an
unexpanded macro call in a function the macro doesn't actually call.

### stage-check-noimpl — MEASURED 2026-07-21 (Linux port; the engines are identical)

The diagnostic was rebuilt (the original scratch prototype was lost with its session
directory) and run: per macro qual, build the closure sub-program with **impls and
consts pruned** (`closure-funcs` minus its impl/const seeding; `closure-subprogram`
with empty impls/consts) and `check-program` it standalone.

- **Compiler self-compile: 100% — every qual checks standalone.** The bootstrap
  path is fully stageable with pruning alone.
- **Corpus (examples/lib/metaprog-poc/mini-scheme/bench): 4264 ok / 19 fail (99.6%).**
  The 19 failures are 11 distinct quals with exactly TWO root causes:
  1. `assert.tr-emit-main`/`tr-run`/`tr-test-name` — a genuine transitive callee
     (`slice.subslice`) carries the unexpanded `(dbg-subslice …)` call. Not a
     pruning gap: staging ORDER fixes it (`slice.dbg-subslice` itself checks
     standalone, so once tier-1 macros are engine-expandable the tower can expand
     `subslice`'s body via the ENGINE and re-check).
  2. `gcauto2.*` (8 quals) — callees genuinely use trait impls (`'i64' does not
     implement 'Ord'`), so blanket impl-pruning over-prunes for them.

**Design consequence — two-tier staging.** Tier 1: per-qual check with impls/consts
pruned; every qual that passes joins the (incremental) engine — this covers the
whole bootstrap and 99.6% of the corpus, including every `dbg-slice-*`/`dbg-subslice`
that today poisons the batch. Tier 2: for the residue, after the tower expands their
callees' macro calls through the tier-1 engine, run the ordinary FULL-closure check
(impls retained) — by then the poisoning calls are gone, so it passes and no
interpreter is involved. The instrumentation diff (throwaway, never committed) is
`stagecheck-instrumentation.diff` in the working scratchpad; it is 100 lines and
trivially re-creatable from this description.

## Engine-parity divergences, MEASURED (Linux, 2026-07-21 — pre-existing, not port bugs)

`parity.sh` over the current 116-file corpus on the Linux port: **114 identical, 2
divergent**. Both divergences are engine differences the deletion must resolve, and
both reproduce with the same mechanism the doc already describes:

- **`lib/assert.coil` — RC-DIFF (interp=1, compiled=0).** Under `COIL_META=interp`
  the interpreter dies on its own capability gap (`comptime: generic call to
  'slice.slice-len' isn't supported yet`); the compiled engine compiles it fine. A
  file the WEAKER engine cannot process at all is exactly the mac-12 complaint —
  and it means parity.sh's "both engines agree" contract already has a standing
  exception that step 5/6 work must either fix or bless.
- **`metaprog-poc/minrep_test.coil` — DIAG-DIFF.** An erroring macro reports
  `macro 'minrep.simple-check' did not return Code` on the interpreter vs
  `compilation failed: checker reported error(s) above` on the compiled engine.
  Diagnostic-wording parity for failing metaprograms is part of "100% compatible".

## Platform note (Linux port, 2026-07-21)

The compiler now also self-hosts on Linux x86-64 (`docs/LINUX_PORT.md`). Two facts
matter for this plan:

- **The MObj in-memory JIT route is arm64-only** — `main.coil` registers
  `set-meta-build-mobj!` only when the host is arm64. On Linux, compiled comptime
  goes through `comptime_eval`'s **dylib route** (`cc -shared` + `dlopen`) and
  passes the full gate-cli readback battery. So step 5 Phase A's "use the in-memory
  MObj/JIT route" is an arm64 optimization, not the portable mechanism: the
  incremental engine must keep a dylib path (or grow an x86-64 MObj emitter).
- Linux is a second verification platform for every Phase-2 step:
  `selfhost/rebootstrap-linux.sh` (LLVM-backend fixpoint + linux gate-full/gate-run
  + gate-cli) and parity.sh both run there.

## Landmines

**`closure-subprogram` does NOT own its bodies.** It copies each `Func` by value, but `body` is
a `(ptr (ArrayList Expr))` **still shared with the real program**. Any in-place rewrite of a
sub-program body therefore mutates the program being compiled. Found the hard way during Phase
1b: single-site tests passed *by luck* (the mutated node was replaced by the folded literal
anyway) and it only surfaced on a two-site program, as
`UNIMPLEMENTED: codegen: unknown callable coil.ct.nz`. Any pass that rewrites a sub-program must
either deep-copy the bodies or record and undo its edits — including on the decline path. This
will bite steps 4 and 5, which both restructure sub-programs.

**Synthesized `ECall` → synthesized callee DOES resolve** through monomorphize and codegen
(proven by injecting a callee returning `x+100` and observing 104, not 4). Post-check synthesized
functions are viable; this was an open risk and is now settled.

**`rebootstrap-nollvm.sh` is broken at HEAD, unrelated to any of this work.** The committed
nollvm seed predates the `isize` surface (`cimport.coil:39`), so stage1 dies with
`unknown type 'isize'`. Reproduced in a clean HEAD worktree. Workaround: `STAGE0=./coil`.
**The nollvm seed needs a refresh** (`selfhost/refresh-seed.sh`) as its own commit — this is
exactly the seed-refresh gotcha below, already tripped. Also pre-existing: `gate-expand`,
`gate-ir`, `gate-load` fail 0-pass on the shipped `./coil` too; not in the authoritative set.

## Gotchas (learned this session — do not relearn the hard way)
- **Seed refresh.** If any change makes the compiler's own source use a surface the committed
  seed predates, the default `rebootstrap.sh` fails ("unknown …"). Bootstrap via
  `COIL_STDLIB_DIR="$PWD" STAGE0=./coil ./selfhost/rebootstrap.sh`, then refresh the seed
  (`cp /tmp/coil-rb2 selfhost/seed/coil-seed && codesign -s - --force …`) and confirm the
  DEFAULT rebootstrap is green. (Unlikely to bite here — no new language surface — but know it.)
- **Ref-regen discipline.** A behaviour-only change leaves `gate-full` byte-identical. If IR/
  dumps move, regen and verify EVERY changed reference line is an explained consequence; the
  RECIPE forbids doctoring refs to pass.
- **Fixpoint is the tripwire.** Every Phase-2 change touches the compiler's own comptime/macro
  machinery; a bug fails `stage2.o == stage3.o` immediately. Rebootstrap after every step.
- **Both compilers.** Verify the LLVM build AND the nollvm/arm64 build (the compiled engine is
  the arm64 backend's default). The teeth are in `gate-cli.sh` (sizeof→8, generic→7,
  const-sizeof→8, c-string-in-comptime→5, aggregate-comptime no-regression) + the diag
  fixtures 06/07 (comptime folds) + `meta_stage3.coil`.

## Definition of done
`comptime.coil` has no tree-walk `eval`; `COIL_META`, `parity.sh`, and the guide's interp
mention are gone; the compiled engine is the sole engine on BOTH backends; fixpoint + all 7
gates + `meta_stage3` green on both LLVM and nollvm rebootstraps; no comptime shape that folds
today regresses.

## NEXT STEPS (ordered — resume here)

1. **Commit Phase 1b** (the comptime-UB guard). Implemented and verified, sitting uncommitted in
   `selfhost/src/comptime_eval.coil`. It is a standalone bug fix and stands on its own merit
   whether or not the deletion proceeds.
2. **Refresh the nollvm seed** as its own commit, so `rebootstrap-nollvm.sh` works from a clean
   checkout again. Independent of the deletion; currently masks real nollvm verification behind
   a `STAGE0=./coil` workaround.
3. ✅ **Step 5, Phase A — incremental engine: LANDED 2026-07-21** (with Phase B, same
   commit). `MEEntry` now carries its build's quasiquote registry (each dylib's code
   indexes the registry of the metalower run that produced it), `meta-engine-setup`
   is ADDITIVE (appends entries not yet served; existing entries win), and
   `finish-macro` looks entries up un-gated by `active` so staged entries serve
   during the tower rounds. The dylib fork stays (measured cheap enough — the
   per-qual sub-programs are tiny and content-cached); the MObj/JIT route remains an
   arm64-only optimization to consider later.
4. ✅ **Step 5, Phase B — two-tier staging: LANDED 2026-07-21, gated
   `COIL_STAGE_MACROS=1` (default OFF until macOS verification).**
   `stage-macros-prepass` (`expander.coil`, hooked at `stage3-round` round 0): per
   qual not yet in the engine, build the PRUNED closure (`closure-funcs-pruned` /
   `closure-subprogram-pruned`, comptime.coil), check standalone on a PRIVATE
   resolve (sidesteps the shared-bodies landmine), and on success compile it into
   the engine additively. Best-effort: any failure leaves the qual to the ordinary
   batch path (tier 2). **Measured on Linux:**
   - emitted IR **byte-identical** staging on vs off across the whole corpus
     (the previously-untested claim — now a PASSED check, 0 diffs);
   - the compiler's own build: **.o byte-identical** with staging on vs off;
   - interp fallbacks: corpus **255 → 0**, compiler self-compile **2 → 0** — with
     staging on, `eval-seq` never fires anywhere in the corpus;
   - fixpoint + linux gate-full/gate-run + gate-cli green in BOTH modes; parity.sh
     unchanged (114/116, the 2 known pre-existing divergences).
   **macOS verification: DONE via CI (2026-07-21).** `.github/workflows/coil.yml`
   runs the full battery on a macos-15 arm64 runner (rebootstrap incl. both
   fixpoints + gate-full byte-exact + arm64 gate-run + gate-cli, parity in both
   modes with the 2 known divergences PINNED, staged-vs-default self-build .o
   byte-identical) and an ubuntu runner (rebootstrap-linux + same parity/staging
   matrix). Both green → **staging is DEFAULT ON** (`meta-staged?` defaults true;
   `COIL_STAGE_MACROS=0` is the soak kill switch, deleted with step 7). Every
   future push touching coil/ re-proves all of this on both platforms.
   **Remaining for step 5:** Phase C, after a soak — the corpus shows 0 interp
   fallbacks with staging on, but Phase C converts any unforeseen miss into a hard
   error, so let CI + real use watch the soak first. CI runner facts worth knowing:
   gate-full needs `COIL_SELF_ARGS="--target <snapshot triple>"` (the default
   triple embeds the host Darwin version), brew LLVM must be pinned to 21 (the 22
   printer spells IR differently), and `timeout` comes from coreutils' gnubin.
5. ✅ **Step 5, Phase C — LANDED 2026-07-21.** Under the compiled engine with staging
   on (the default), a `finish-macro` engine miss is a HARD ERROR naming the macro;
   `COIL_META=interp` / `COIL_STAGE_MACROS=0` keep the interpreter path (oracle +
   kill switch) until step 7. The staging prepass now runs EVERY tower round (not
   just round 0), so a macro whose pruned closure held an unexpanded call in round
   N stages in round N+1 — the engine converges toward total and the corpus builds
   entirely without `eval-seq`.
6. ✅ **Step 4 — LANDED 2026-07-21.** `(meta …)` runs on the compiled engine:
   `expander.coil::elaborate-metas-compiled` builds the generator sub-program at
   the PRE-CHECK form level (metashim injected via `meta-sub-forms`, like the macro
   engine), wraps each meta expression as a synthetic zero-param Code-returning
   entry `coil.meta.<i>`, compiles them through `meta-engine-setup`, runs each via
   `meta-engine-run-env`, and splices with the interpreter path's exact semantics
   (`(do …)` splice, then strict re-resolve + check). Routed by
   `frontend-check-routed`; the interp path remains behind the same two flags.
   **Verified:** `meta_stage3.coil` — identical stdout, exit code, and BYTE-IDENTICAL
   emitted IR, compiled vs interp; fixpoint + all gates + parity green.
7. ✅ **Step 6 — LANDED 2026-07-21.** `fold-expr` is compiled-PRIMARY: when the fold
   hook is registered and no compat flag opts out, every `(comptime …)`/`(const …)`
   site goes straight to the compiled engine (`CtFoldHook.primary`, decided at
   startup by `register-comptime-fold!` from the engine flags — comptime.coil is
   imported by metaengine and cannot read them itself). A hook decline is a located
   hard error. The decline shapes were MEASURED before flipping (both engines were
   probed per shape): scalars/f32/strings/aggregates/sums/packed/explicit fold
   identically (gate-full byte-exact proves it); Code-at-runtime keeps its exact
   error wording (now produced by `comptime-fold-one`); div/rem-by-zero comes from
   the Phase-1b guard, `(const …)` included. **One intentional tightening:** the
   interpreter used to fold NUMERIC int→ptr casts (`(comptime (cast (ptr i8) N))`,
   incl. inside aggregates); the compiled readback declines ALL pointer results —
   a compile-time constant holding a process address is the silently-wrong-constant
   class the guard exists to prevent, and the compiled engine (unlike the interp)
   can produce REAL addresses. Workaround is strictly better: fold the integer,
   cast at runtime. Nothing in the corpus used it (all gates byte-exact/green).
   `comptime-cap-gap?` survives only inside the legacy flag path and dies at step 7.
   Nested-fold errors under compiled-primary SURFACE instead of declining (they
   were previously masked behind the interp's error).
8. **Step 7 — the deletion itself. THE ONLY STEP LEFT. Two hard preconditions:**
   (a) **the nollvm seed refresh** — `rebootstrap-nollvm.sh` is still broken at HEAD
   (committed seed predates `isize`), so the LLVM-free compiler's engine-only path
   is currently UNVERIFIABLE; deleting the interpreter before the nollvm build can
   be rebootstrapped and gated would strand it. Refresh the seed on the Mac (or via
   CI artifact), get the nollvm gate green, add it to CI. (b) **soak**: steps 4/5C/6
   all landed 2026-07-21; let CI + real use run on compiled-primary before removing
   the `COIL_META=interp` / `COIL_STAGE_MACROS=0` escape hatches. Then: delete
   `eval`/`eval-seq`/`eval-args`, `build-value`/`build-content`'s interp-only
   callers, `comptime-cap-gap?`, both env flags, `parity.sh` (its job is done — the
   CI parity pin goes with it), the `guide.coil` interp mention; mark
   `docs/DECISIONS.md` decision 7 DONE. As of tonight the unification is
   FUNCTIONALLY COMPLETE: by default, macros (step 5), `(meta …)` generators
   (step 4), and `(comptime …)`/`(const …)` (step 6) all execute on the compiled
   engine; the tree-walker is dead code behind the two compat flags.

⚠ **Ordering constraint that is NOT optional:** `parity.sh` (`COIL_META=interp` vs compiled,
byte-identical IR over 112 files) is the ONLY oracle that can prove step 5 didn't change expansion
semantics — and it dies with the interpreter. **Step 5 must be landed and validated while the
interpreter still exists.** That is the real reason step 7 is last.

## Recommendation
Phase 1 and 1b are safe, valuable and bankable now. Step 5 is no longer the month-scale unknown
this doc originally assumed — the measurement above puts its surface at 10 macros with ~0
dependency depth, dominated by a single stdlib case worth 73% on its own. It is still the piece
with direct bootstrap risk (it touches the compiler's own macro machinery, so a bug fails
`stage2.o == stage3.o` immediately), so: dedicated session, one step per rebootstrap. Start by
making `dbg-slice-get`/`dbg-slice-set` build standalone and confirming the 418 → ~112 drop; that
single number validates the whole Phase B design before any of the harder plumbing.
