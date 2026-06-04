# Handoff: JS partial evaluator — soundness, fuzzing, and simple.js

This doc hands off the current state of `partial-new` (the generic online partial
evaluator + its real-JavaScript frontend) so the next session can resume without
re-deriving anything. Read `docs/adversarial-deobfuscation.md` first for the
deeper lessons; this doc is the up-to-date status and the concrete next steps.

## TL;DR

- The JS partial evaluator is sound across extensive differential fuzzing of the
  original generator (~24k programs, zero divergences). The **recursion axis**
  added this session (self-recursive functions in the generator) shifted the RNG
  stream and surfaced a batch of **pre-existing** soundness bugs in the eval core.
- This session: added **recursion generation** + a **specialization weight budget**
  (branch-explosion OOM guard, JS-client only), and **fixed five soundness bugs**
  (all reproduced, regression-tested, fuzz- and Node-verified). Divergences in the
  1–500 range went **19 → 2**.
- One coherent **soundness class remains** (and is the bulk of divergences in the
  500–2000 range): the residual fails to reproduce a JS `throw` when a may-throw
  expression sits in a **dead/discarded position** (dead store, dead array element,
  residual-function body). The direct expression-statement case is FIXED; the
  others are not. See "Remaining issue: throw preservation" below.
- Tests: **99 frontend + 12 engine, all green**. simple.js still specializes
  correctly (the fixes fold to no-ops on its inputs).

## Update (2026-06-04): effect #4 fixed + a method-call ordering bug fixed

Two fixes landed since the TL;DR above (both fuzz- and Node-verified, regression-tested):

1. **simple.js effect #4 (byte-array duplication) — FIXED.** See "Remaining issue
   #2" below for the full write-up. Escape-at-creation duplication repair in
   `Js::specialize_program`. difftrace now 13/13 identical.

2. **Method-call receiver coercibility ordering — FIXED.** A method call
   `recv.m(args)` evaluates `recv.m` (which throws if `recv` is null/undefined)
   BEFORE its arguments, but the residual emits an *impure* argument's side effects
   eagerly (as a preceding statement), moving them ahead of the receiver-throw.
   Observable: original throws at `a.push(b++)` before `b++`, so `b` stays
   `undefined` and a later `b.x=1` throws; the buggy residual ran `b++` first
   (`b=NaN`), making `b.x=1` a silent no-op and swallowing the throw. This was the
   **fuzzer's seed-20968 throw-mismatch**. Fix: a new `Instr::ChkCoercible(key)`,
   emitted by `compile` between the receiver and the arguments ONLY when an
   argument is impure (`!is_pure`), commits the receiver's coercibility throw
   (`recv.key` read) at the member-access point. It folds away for
   statically-coercible receivers and for provably-coercible dynamic receivers
   (`Math.foo(...)`; see `rexpr_coercible`). Regression test
   `method_callee_throws_before_impure_arg`. Remaining narrow gap: a *dynamic*
   receiver that is null/undefined at runtime with an impure argument, where the
   receiver expr is not provably coercible-or-nullish — left unguarded to avoid
   bloat/false-throws (pre-existing behavior, not a regression).

3. **try/catch handler leak on `break`/`continue` — FIXED.** A `break`/`continue`
   inside a `try` body jumps out of the enclosing loop/switch WITHOUT running the
   body's `PopHandler`, so the handler leaked past the loop and a later residual op
   tripped the in-try guard (`forbid_residual_in_try`) — the fuzzer's seed-1042 /
   20252 / 21217 / 21500 probe-bug panics. Fix in `src/js.rs` `compile`: track open
   handler depth per loop/switch frame (`CompileAux.handler_depth` + `breaks_hd` /
   `continues_hd`) and emit one `PopHandler` per `try` the `break`/`continue` exits,
   before its jump. (`return` already cleaned up its frame's handlers via
   `retain(frame_depth < depth)`; this closes the same gap for the non-local jumps.)
   Regression test `break_continue_out_of_try_pops_handler`.

4. **Nested-`try` foldability probe corruption — FIXED.** `try_folds` set/reset
   `probing`/`try_taint` unconditionally; when a probed body contained a nested
   `try`, the inner probe reset the OUTER probe's `probing` flag (and clobbered its
   taint), so a later residual op in the outer probe panicked instead of tainting.
   Fixed with the `residual_fn_for` save/restore pattern.

Together (3)+(4) take the try/catch probe panics from ~3-per-2000 to ~0 on bases
1/20000 and reduce them broadly.

5. **Prefix `--x`/`++x` single-evaluation — FIXED.** A prefix update's result is
   the new value read back from the place, a live expression aliasing that place
   (`(input - 0) - 1`). When the place is later reassigned — most commonly the
   loop-carried materialization of the very slot just stored — the un-snapshotted
   result re-reads the new value and re-applies the `± 1`, diverging (the fuzzer's
   seed-60510 value bug: `(--input)` in a loop condition returned 1 instead of 2
   from a double-decrement). Fix: prefix now `Instr::Snapshot`s its result, mirror
   of the postfix snapshot of the old value. Also taught the Rust residual
   interpreter to evaluate `Op::Eval` (snapshot temps) via `eval_rexpr` instead of
   blanket-panicking — it still panics on a genuinely unmodeled call, so the
   Node-only contract for pass-through programs is unchanged. Regression test
   `prefix_update_of_loop_carried_place_is_single_evaluated`.

**Known pre-existing soundness bugs still open** (verified identical on the pre-fix
binary; NOT caused by any fix here):
- **probe-vs-main-path whistle divergence for a loop inside a folded `try`** — e.g.
  fuzz seed 31079 (still a "probe-bug" panic). A `while`/`for` whose body is fully
  static unrolls in isolation, but inside an outer `try` the MAIN path's whistle (with
  accumulated `seen` history) GENERALIZES it into a residual loop, while the isolated
  `try_folds` probe (fresh `seen`) does not — so the probe says "folds" but the main
  path materializes a loop while the handler is active, tripping the in-try guard.
  Root cause is architectural: `try_folds` re-specializes the region in isolation
  (`sole_frame_state`, empty `seen`), so it cannot always predict the main path's
  generalization decisions. A real fix needs the probe to see the same generalization
  pressure as the main path, or `PushHandler` to conservatively residualize when the
  body contains a loop that could materialize.
- **function-source identity (INHERENT LIMITATION, not a fixable bug)** — e.g.
  fuzz seeds 41158, 71404. A program that string-coerces a function (`String(fn)`,
  `"" + fn`, `fn.toString()`) observes its SOURCE TEXT. A residual closure renders
  as `__rfN.bind(null, caps)`, whose `String(...)` is `function () { [native code] }`,
  while the original is the source. This is **impossible to fix in any correct
  partial evaluator**: specialization changes a function's body (folded constants,
  lowered/renamed variables), so `String(specialized_fn)` cannot equal
  `String(original_fn)` — even rendering the closure without `.bind` only changes
  `[native code]` into a different-but-still-not-original source. PE is sound *up to
  function-source identity*; programs that observe a function's source text are
  outside its contract. The fuzzer flags these as `kind=value` false positives
  (~2 per 14k programs); they are not soundness defects.

Current fuzz status with all fixes: **0 divergences attributable to these
changes** across bases 1/10000/20000/30000/40000/60000/70000 (~14k programs); the
only divergences/panics seen are the two pre-existing classes above (seed 31079
probe panic, seed 41158 `.toString()`). The seed-60510 value divergence is now
FIXED (fix #5 above).

## How to run things

```bash
# build
cargo build --release                       # engine (partial) + frontend lib
cargo build -p js-frontend --release         # frontend + js-frontend bin
cargo build -p js-frontend --release --bin fuzz
# GOTCHA: the `fuzz` bin can go STALE — cargo sometimes does NOT relink it after a
# `partial`-crate (src/js.rs) change, so it runs OLD codegen and reports
# already-fixed divergences. If a seed you just fixed still diverges only under
# `fuzz`, force a relink: `rm -f target/release/deps/fuzz-* target/release/fuzz`
# then rebuild. Always confirm a `fuzz` finding against `./target/release/js-frontend
# --js <prog>` (the main bin rebuilds reliably).

# tests (THE gate — keep green)
cargo test -p partial --release
cargo test -p js-frontend --release

# specialize a JS file, emit residual JS
./target/release/js-frontend --js path/to/file.js
./target/release/js-frontend --js file.js 5 42     # also runs residual vs reference for inputs

# the fuzzer (differential: Node(original) vs Node(residual))
./target/release/fuzz --seed N --count 5000 --batch 300
./target/release/fuzz --repro SEED                  # one program + residual + findings
./target/release/fuzz --overflow SEED               # shrink a stack-overflow seed (subprocess)
./target/release/fuzz --specialize-file f.js        # worker mode (used by overflow shrinker)

# the two oracles for simple.js
node tools/difftrace.js /Users/jimmyhmiller/Documents/Code/deob/simple.js
#   -> differential EXTERNAL-EFFECT trace (Date.now/console). First divergence = bug site.
#   NOTE: difftrace wraps Date as a logging proxy; for the REAL failure run the
#   residual with real globals (see /tmp/runres*.js patterns from last session).
```

## Architecture (one paragraph)

`src/engine.rs` is the generic online PE (driving + memoization + whistle); it is
**never changed** for JS-specific work — all JS semantics live in the client
`src/js.rs`. The frontend (`js-frontend/`) lowers real JS (SWC) to the
`partial::js` AST (`js-frontend/src/lower.rs`), specializes `function main(input)`,
and emits residual JS (`js-frontend/src/codegen.rs`). The residual is either a
single straight-line block or a `switch(__pc)` trampoline. Heap objects use a
precise abstract heap with partial escape analysis: a tracked object that flows
into unmodeled code **escapes** (its references everywhere are invalidated and it
is reconstructed in the residual). The memo (`engine.rs`) keys residual blocks by
full `State`.

## Fixed this session (all have regression tests in `js-frontend/src/lib.rs`)

1. **Coercion / wrong-type totality** (earlier in session): `eval_bin` + all
   member/index/write handlers residualize instead of `panic!` on JS coercion
   (`1 - "y"`, `1 + null`, `obj[5]`, non-array push, heap-ref-in-primitive, …).
   Helper `escape_if_ref`, `index_is_static`.

2. **JSFuck / coercion folding** (`to_string_static`/`to_number_static`/
   `to_primitive_static`/`try_fold_bin_coerced`, string-method folding
   `try_string_method`/`try_string_split`). Verified against the real `jsfuck`
   npm encoder. Unary `+` was lowered as IDENTITY — fixed to `x - 0`.

3. **`arguments` modeled** (was a silent `Global("arguments")` = undefined). Per
   function: lowerer reserves `FuncDef.arguments_slot` lazily (`lower_ident`).
   Inline path fills it with an array of actual args; a RESIDUALIZED function
   reads native `arguments` via sentinel `ARGUMENTS_VAR_ID` (codegen renders it
   `arguments`), sliced to drop bound captures
   (`Array.prototype.slice.call(arguments, ncaptured)`) — captures are bound as
   leading params, so naive native `arguments` over-counts (this WAS a real
   divergence: `7 + arguments.length` gave 11 not 10). Also fixed: inline arg
   binding now binds only `i < nparams` (extras are arguments-only).

4. **freeze-before-opaque-call** (`freeze_before_opaque_call`): before all three
   opaque-call arms in `do_call`, snapshot live frame values whose residual
   expression reads mutable state (`Get`/`Index`/`Global`) to temps. Doc
   hypothesis #2.

5. **capture-escape-on-mutation** (the simple.js self-modifying-array class): a
   residualized closure (one containing `try`) that captures an object and
   mutates it through the capture must invalidate the caller's view. `do_call`
   has_try arm now routes captures through `operand_rexpr` (which escapes Ref
   captures) instead of `materialize_value` (which left the caller's stale copy
   live). Repro: closure does `data[1][1]^=5`, main read 50 instead of 55.

## Fixed this session, round 2 — recursion-axis soundness bugs

All found by the new recursion generator's RNG shift; all were **pre-existing** in
the eval core (not caused by the recursion code). Each has a regression test in
`js-frontend/src/lib.rs` and was verified against Node.

1. **Short-circuit side effects** (`compile_short_circuit`, `js.rs`): `&&`, `||`,
   `??`, `?:` were lowered to `Expr::Opaque` (eager, all operands evaluated), so a
   side-effecting operand on the *skipped* path still ran. Now: when the
   conditionally-evaluated operand is impure, compile real short-circuit control
   flow (new `Instr::Dup`); when it is pure, keep the compact `Opaque` (so simple.js
   is unchanged). Repro: `input && (++input)` returned 1 at input=0, should be 0.

2. **Postfix `++`/`--` result coercion** (`compile_update`, `js.rs`): `x--` yielded
   the raw old value, not `ToNumber(old)`. `false--` gave `false` not 0. Fix: coerce
   the postfix result via `x - 0` (folds to a no-op for numbers).

3. **`++`/`--` store coercion** — two sites: expression position (`emit_store` in
   `js.rs::compile_update`) and statement position (`lower.rs::lower_expr_stmt`,
   which lowered `t++;` straight to `t = t + 1`). `obj++` did `obj + 1` (string
   concat → "[object Object]1") instead of `ToNumber(obj) + 1` (NaN); `"3"++` gave
   "31" not 4. Fix: store `(place - 0) ± 1`.

4. **Discarded may-throw expression dropped** (`Instr::Pop` in `js.rs::step` +
   `may_throw`): `undefined.length;` as a statement throws, but the PE
   dead-eliminated it (`return 0`). Now a discarded value that `may_throw`
   (member/index access, call/`new`, `in`/`instanceof`) is emitted for effect via
   `Op::Eval`. NOTE: this only covers the `Pop` path; the dead-store / dead-element
   / residual-fn-body variants are NOT yet handled (see remaining issue).

## The recursion axis + the weight budget (this session)

- The fuzzer now generates **self-recursive functions**: `rec_func` emits
  `r{i}(n, ...data)` with a frozen, strictly-decreasing counter `n` (base case
  `if (n <= 0) return …;` first), self-calls passing `n - 1`, called with a static
  small counter so the evaluator fully unrolls (a dynamic counter hits the
  deliberate recursion refusal). Covers both the inlined-unroll path and (when the
  body has a `try`) the residualized-recursive-function path.
- Deliberate refusals (`dynamic-depth recursion`, non-terminating static recursion,
  continue-out-of-try, **specialization budget exceeded**) are classified as a new
  `Residual::Refused` outcome: counted, never reported as a bug, never chased by the
  shrinker.
- **Specialization weight budget** (`SPEC_WEIGHT_BUDGET` in `js.rs`, a JS-client
  resource bound — the generic engine is untouched). A depth-bounded recursion that
  forks on dynamic values inside a loop can produce a finite-but-enormous residual
  (seed 339 → 24 GB). The budget caps accumulated `State::weight` (Σ residual-expr
  nodes over block entries) and refuses cleanly past it. CLI default 300M (clears
  simple.js's 141M with headroom); the fuzzer sets **100k** via the env var (max
  legit generated weight is ~1.4k → ~70x headroom), which bounds even the heaviest
  blowups (whose memory-per-weight is far above normal, e.g. seed 738) to well under
  a gigabyte. The full 2000-program run now completes at ~620 MB peak (was OOM at
  8–9 GB). CAVEAT: weight is an imperfect memory proxy — different programs vary
  ~2000x in bytes/weight — so the budget is tuned per-workload, not universal; a
  truly robust guard would be subprocess `RLIMIT_AS`.

## Also fixed this session

- **Math modeled as a built-in** (`try_builtin_static`, the same hook as
  `String.fromCharCode`): deterministic integer-result methods fold (`floor`/`ceil`/
  `round`/`trunc` are identity in the i64 model; `abs`/`sign`/`max`/`min` over static
  numbers). Float (`sqrt`, `pow`, …) and non-deterministic `random` deliberately pass
  through. This is the **stdlib-extension pattern**: add an arm to `try_builtin_static`
  (static methods on a global), `try_builtin_new` (constructors), or
  `try_builtin_method` (bound instance methods).
- **Residual-function body context leak** (seed 228): `residual_fn_for` now specializes
  the body in a CLEAN context (save/clear/restore `probing`, `halt_at`, `eager_stores`,
  residual-try stack). A function residualized while the caller was mid-probe or
  mid-`try` (e.g. `f.call(x)` inside a `try`) inherited that state and produced an
  EMPTY `__rf0`, silently dropping the body's calls/returns/throws. Test:
  `residual_function_body_kept_when_caller_is_in_a_try`.

## Remaining issue: throw preservation in dead positions (mostly fuzzer artifact)

The residual can silently skip an exception the original raises when a may-throw
expression sits in a position whose value is dropped:
- **used** value: correct (`return undefined.length` throws at runtime).
- **`Pop`-discarded** statement (`undefined.length;`): FIXED (`may_throw` + `Op::Eval`
  at the `Pop` site — emits the whole expression, so a short-circuit like
  `cond && undefined.x;` still short-circuits at runtime).
- **dead store / dead element** (`var x = undefined.length;`, `[{b: a8.length}]` with
  the value unused): still an under-throw MISS. An attempt to fix this by
  eager-emitting nullish member/index reads at `GetProp`/`GetIndex` was **reverted** —
  it hoisted the throw out of the compact short-circuit form (`cond && undefined.x`
  evaluates the operand eagerly), causing **over-throws**. The Pop-based fix is the
  safe boundary; covering dead stores needs liveness or a may-throw-effect model that
  doesn't break short-circuiting.

This remaining class is dominated by a **fuzzer artifact**: the generator/shrinker
emits reads of **undeclared variables** (`a1`, `a8`, a free `input` inside a top-level
function). In JS an undeclared read throws ReferenceError; the PE models every
unresolved identifier as a (non-throwing) `Global`. Used → the residual reproduces it
(`return (input + 1)` with `input` unbound also throws → match); only DEAD positions
diverge (seed 231 `void a1` in a truthy-ternary object). NOT cleanly fixable — the PE
can't distinguish an undeclared identifier from a real global (`Math`, `String`), and
real code never reads undeclared vars. The right fix is at the fuzzer: stop the
generator hardcoding `Expr::Var("input")` in `atom()` where it's out of scope, and
stop the shrinker reducing to a different failure (it deletes declarations, leaving
references dangling — so shrunk minimals over-represent this artifact). The lone
non-throw straggler (seed 22) is a deeply contrived nested case.

## The fuzzer (`js-frontend/src/bin/fuzz.rs` + `tools/fuzzcmp.js`)

- Seeded structured-AST generator → JS printer → batched Node oracle
  (`fuzzcmp.js` runs original vs residual `main(input)` in isolated vm contexts
  under a 1s timeout; classifies value / throw-mismatch / termination / loader).
- Generates: literals, arithmetic/coercion, arrays/objects, index/member
  read+write, for/while (bounded, terminating by construction), switch,
  try/catch, throw, break/continue, early return, `arguments` via methods,
  string methods, **closures** that capture+mutate outer vars (often `try`-
  wrapped to force residualization), and **`.apply`/`.call`** invocation.
- Shrinker reduces failures to minimal reproducers (descends into closure
  bodies). Panic shrinker + `--overflow` subprocess shrinker for uncatchable
  aborts.
- Terminating by construction; the only nontermination the oracle should see is
  an engine bug. (A past `exit 134` was a generator `u32 fuel` underflow, fixed
  with `saturating_sub`.)

**To find more bugs**: the fuzzer currently does NOT generate recursion (acyclic
call graph only) or nested/deep data mutated across handlers. Adding bounded
recursion (a counter param + base case, called with a small static N) is the
most likely next axis to surface new bugs.

## Remaining issue #1: continue/break out of a residual `try` (REFUSAL, hard)

Minimal repro: `while(g>0){ g=g-1; try{ foo(input); } catch(a){ continue; } }`.
`residualize_try` (src/js.rs ~3450) specializes body/catch as nested
`specialize()` calls bounded by `halt_at=end`. A `continue` Jmps to the loop head
(outside the region); the nested call follows it and re-enters the same `try` →
infinite recursion. Currently guarded by `Js.residual_try_stack` → a **clear,
catchable panic** (a refusal, never a wrong answer).

**Why it's architecturally hard** (confirmed): the continue target is the outer
loop-head *block*, but residual blocks are keyed by full `State` and that block
is created **only by the very re-entry that recurses** (`create_or_get` in
engine.rs). So the residual `try` cannot reference the target block id when it is
built. The real fix is to stop specializing the catch in isolation and instead
feed the catch's continuation into the OUTER specialize work queue. Scoped design
(if you still want the isolated-subprogram approach): make
`Terminator::Halt` carry `Option<usize>` (a non-local-exit target), add
loop_head/loop_end to the sub-program halt set, and have codegen set a shared
`__exit` var in the nested trampoline + jump in the outer trampoline after the
try/catch — BUT this still needs the pc→outer-block map, which is the hard part.
Gate any attempt on the residual-try tests; risk = destabilizing the sound path.

## Remaining issue #2: simple.js effect #4 — **FIXED 2026-06-03**

> **RESOLVED.** `node tools/difftrace.js .../simple.js` now reports
> `original effects: 13, residual effects: 13 — IDENTICAL external-effect traces`.
> simple.js specializes cleanly (exit 0) and the residual is *smaller* (~4k lines,
> down from ~14k — the duplicate byte-array literals are gone).
>
> **Fix (the §5 "escape at creation" plan from `docs/effect4-handoff.md`):** the
> partial evaluator was realizing ONE abstract heap object as several runtime
> objects by reconstructing it (`materialize`) at multiple control-flow merge
> points; a runtime write to one copy was invisible to a read of another. The
> repair, all in `src/js.rs`:
> - **Profile** (`Js::specialize_program`): specialize, counting per abstract
>   address how many distinct `materialize` constructions it received
>   (`dup_counts`, bumped from the `seen` set at the end of `materialize`). An
>   address built >= 2 times is a duplicated identity.
> - **Escape at creation**: flag that object's creation pc (`creation_pc` side
>   map) and re-specialize; at a flagged `NewArray`/`NewObject` the object is
>   `escape`d immediately — built once as a runtime variable at the one point that
>   provably dominates every use — so no merge point can reconstruct it. Iterates
>   to a fixpoint (escaping one object can reveal another). On simple.js: 10
>   creation sites flagged, max constructions 12 -> (benign) 2, converges in 2
>   passes.
> - Why not the flat "materialize-once memo" (§4 of the handoff): that construction
>   did not *dominate* later references and threw `undefined.value`. Escaping at
>   creation is the only point guaranteed to dominate.
> - **Soundness**: escaping at creation is observationally transparent (it only
>   changes a static literal into a runtime variable), so over-escaping can never
>   break correctness — only forgo folding. Guarded by `FORCE_ESCAPE_ALL` (escape
>   EVERY object): the `escape_at_creation_is_transparent` test and a 1500-program
>   fuzz run under it both stay equivalent. The normal fuzzer adds **0** new
>   divergences/panics (the seed-1042 try/catch panic and seed-20968 throw-mismatch
>   are **pre-existing**, in the unrelated residual-`try` path — verified identical
>   on the pre-fix binary).
>
> A true minimal repro of the cross-merge-point duplication remains impractical
> (any small dynamic program escapes the object to a single var first, as the
> handoff predicted); the `force_escape_all` transparency test is the committed
> regression guard, with simple.js's difftrace as the integration gate.

---

### Original diagnosis (kept for history)

`/Users/jimmyhmiller/Documents/Code/deob/simple.js` is a multi-layer obfuscator
(a loader VM decrypts self-modifying bytecode in place; a bytecode VM runs it;
anti-debug `Date.now` timing). It now **specializes cleanly** (exit 0, ~14k-line
residual) and is observationally equivalent through the **first 4 `Date.now`
effects**. It diverges at effect #4.

**Fully traced** (differential byte-read instrumentation — tooling left in `/tmp`:
`runseq.js`, `simple_instO.js`/`simple_instR5.js`, `seqO.txt`/`seqR5.txt`,
`keyO.txt`/`keyR.txt`):
- The decoded byte stream is **byte-perfect for 4474 of 10007 reads**.
- At read 4475 the XOR key and position BOTH match, but `v5[1][1131]` differs
  (23 vs 152, off by exactly `^143`).
- The bytecode is **self-modifying**: a residualized loader loop in `main`
  (`v1646[v1647] ^= 143`, where `v1646 == v5[v1645]`, the SAME object as the
  reader's array — verified, not an identity bug) decrypts it lazily.
- So the lazy-decrypt write for index 1131 is emitted **after** the read in the
  residualized loader, when the original does it before → an **effect-ordering
  bug** between interleaved decrypt-writes and byte-reads across the residualized
  loop. The downstream symptoms (method name `undefined`, `Date.now` undefined)
  all derive from this one wrong byte derailing the decoder.

The capture-escape fix (#5 above) did NOT fix this — simple.js's decrypt is in
`main`'s loader loop, not a has_try closure.

### UPDATED DIAGNOSIS (2026-06-03): it is an OBJECT-IDENTITY / escape-persistence
bug, not pure effect-ordering. Structural evidence from the residual (`/tmp/r.js`,
`SPEC_WEIGHT_BUDGET=999999999`):
- The original builds the byte array `v5` **once** (simple.js line 259, declared
  231) and decrypts it in place / reads it across the whole VM loop — ONE object.
- In the residual the array is **partial-static and re-materialized as fresh
  literals in ~10 different trampoline blocks**: the distinctive chunk literal
  `[50, 139, 244, 245, 246, 229, 75, 35, …]` appears **52 times**; the box
  `v1640 = {"value": v100049}` is reconstructed in 10 blocks (lines 8528,
  13700–13890), each with fresh chunk arrays.
- The decrypt target is bound INCONSISTENTLY: in some blocks `v1646 =
  v1640.value[v1645]` (the shared chunk — correct, line 13635), in others
  `v1646 = [50, 139, …]` (a FRESH copy — line 13836). A decrypt-write to a fresh
  copy is lost; a read in a block holding a different fresh copy misses it. That is
  the "off by ^143": the read's copy was never decrypted (the decrypt hit a
  different object).
- The decoder read IS fully residualized (`__rf3`:
  `v900192.value[v900193.value][v600151]`), so this is not a static-fold issue —
  it is that the runtime array the decoder reads is a DIFFERENT object than the one
  the loader decrypted, because the PE re-creates the array per block instead of
  carrying one escaped instance.

**Root cause.** A partial-static array that is mutated through a residualized
(dynamic-index) write inside a loop must escape to ONE runtime instance that is
loop-carried across all trampoline blocks. Today the array stays static in each
block's memoized state, so it re-materializes fresh (losing cross-block writes) and
different blocks hold different copies. The fix lives in the loop-header
generalization + escape interaction: once the array escapes on any path, every
block's view of it must become the same escaped runtime var (not re-materialize the
static literal). This is the partial-static-array consistency problem; the hard part
is doing it WITHOUT losing the static folding that lets the decode collapse at all
(fold while fully static; escape permanently and loop-carry the instant a
residualized write happens).

### Progress 2026-06-03 (effect-4 session): PROVEN to be a duplication/identity
bug (the older "same object, verified" note above is STALE for the current code);
one adjacent aliasing bug found+FIXED; minimal repro of the exact simple.js path
still elusive.

**Runtime instrumentation settles it: identity, not ordering.** Tagging the byte
arrays with a WeakMap id and logging reads/writes at the diverging index 1131
(`/tmp/instr3.js` pattern: wrap the `__rf3` read and the `v1646[v1647] ^= …`
writes) shows THREE distinct objects at index 1131: WRITE hits obj#1 (152→23, the
`^143` decrypt) and obj#2; the READ hits obj#3 (value 152, never decrypted). So the
decoder reads a DIFFERENT object than the loader decrypts — off by ^143 because the
read's object was never written. The byte values are concrete numbers, so the OUTER
array stays partial-static (re-materialized) while an INNER chunk is what diverges.

**Not yet minimally reproducible.** ~10 hand-crafted repros (array in a dynamic
loop; nested arrays; static-outer/dynamic-inner index; boxed + closure-captured;
array created in an init loop) ALL fold to a single shared instance, because
`escape()` escapes the whole reachable graph from the root and `materialize` (now
with shared `seen`) keeps one instance. simple.js re-materializes the array at loop
back-edges anyway — the trigger is some interaction of: the array is BOXED
(`{value:…}`) and captured by the residualized decoder closure `__rf3`; it is
created inside the outer `while (v10>=0)` init loop; the decode/decrypt run in a
deeper VM loop; and the chunk index is static while the byte index is dynamic. The
abstract OUTER array's contents change via folded (static-index) writes, so
`materialize` re-constructs it each iteration as fresh literals; a residualized
(dynamic-byte-index) write to an inner chunk goes to the runtime object and is lost
when the next iteration re-constructs.

**Exact site found (PE instrumentation).** A debug print in `materialize`
(`src/js.rs`, the loop/branch materializer) over large Ref slots shows the byte
array — **slot 6, addr 44, `Array[1197]`** — is materialized **4 times** (same
abstract addr 44, four fresh `NewArray` constructions). Those four runtime arrays
are the id=1/2/3 objects the runtime trace saw: the decrypt writes one, the decoder
reads another. The 4 calls are at four different control-flow merge points
(loop-header generalize / branch joins) that each independently re-materialize the
same loop-carried object — there is no cross-`materialize` memory that "addr 44 is
already a residual var X", so each emits a fresh literal.

**Fix direction (substantial — next session).** Persist the escape: the first time a
loop-carried object is materialized, record `addr -> var`; later materializations of
the same addr reference that var instead of re-constructing, AND the construction
must dominate all uses (var assigned before any block reads it). Equivalently: once
the byte array must be residual, escape it FULLY to one runtime instance and emit
every decrypt as a runtime `SetIndex` on it (the decode is already a runtime closure
`__rf3`, so little folding is lost). Care points: (a) dominance/scoping of the single
construction across the trampoline; (b) don't regress the static folding for arrays
that never need to be residual. The shared-`seen` fix landed this session removes
WITHIN-`materialize` duplication; this is the ACROSS-`materialize` (cross-merge-point)
version.

**ATTEMPTED + REVERTED (2026-06-03): a flat cross-`materialize` memo is NOT
dominance-safe.** Implemented exactly the "record `addr -> (var, content snapshot)`,
seed each `materialize`, reuse the var when the snapshot is unchanged, re-materialize
when changed" idea (with the memo cleared/restored around residual-function
specialization). Effect: simple.js's array-dup count dropped **52 → 5** (the memo
does dedup), but it then threw `TypeError: Cannot read properties of undefined
(reading 'value')` at effect #0 — a reused var (the `{value:…}` box) referenced in a
block whose construction does NOT dominate it. So a flat memo is unsound: reuse is
only valid when the single construction dominates every reuse site. The real fix must
be dominance-aware — either (1) compute/track dominance in the streaming trampoline
and only reuse a memoized var on dominated paths, or (2) HOIST the one construction to
a block that dominates all merge points (e.g. the loop pre-header / function entry) so
every back-edge can reference it, or (3) escape the object eagerly at its CREATION
point (`NewArray`/`NewObject`), which is the natural dominator, when it is later
mutated through a residualized write — turning it into a runtime `Var` from the start.
Option (3) is the most promising: it sidesteps post-hoc dominance entirely (the
creation site dominates by construction), at the cost of a "will this need residual
identity?" analysis (or: conservatively escape any array created in a loop that is
ever written with a dynamic index). The 112 tests + the 14k-program fuzzer are a
strong soundness gate for whichever approach — the flat memo failed simple.js
immediately, so a wrong approach is caught fast.

While hunting effect #4 a minimal **closure-captured-array aliasing** bug was found
and fixed (test `closure_captured_array_aliases_direct_mutation`, repro at
`docs/effect4-minimal-repro.js`): an array captured by a *residualized* closure AND
mutated directly was materialized as TWO copies by the loop/branch materializer
(`materialize` in `src/js.rs`) — each Ref slot used its own fresh `seen` map, so the
closure's capture and the directly-held array became separate objects. Fix: thread
ONE shared `seen` across all Ref slots and emit non-closure objects before closures
(so `.bind(null, arr)` sees an assigned `arr`); `materialize_into_seen` is the new
shared-`seen` entry. This is a real soundness fix but is NOT the simple.js bug:
simple.js's array-dup count (52) is unchanged and it still diverges at effect #4.

**The actual simple.js mechanism (now isolated).** The byte-array box
`v1640 = {"value":[chunks]}` is **loop-carried and RE-MATERIALIZED (re-constructed
as fresh static literals) at every back-edge to the loop header (`pc 8`)** — it
appears in ~10 trampoline blocks that all `__pc = 8`. Decrypts with a STATIC index
fold into the abstract array, so they ARE reflected in each re-construction; a
decrypt with a DYNAMIC index RESIDUALIZES to a runtime `SetIndex` on ONE iteration's
construction and is LOST when the next iteration re-constructs the array fresh. So
the lazy-decrypt of byte 1131 (dynamic index) is written to one construction and the
decoder reads a later, freshly-reconstructed (un-decrypted) copy → off by ^143.

The repros above (`docs/effect4-minimal-repro.js` and variants) do NOT reproduce
THIS: they keep one escaped array because the array there never reverts to a static
Ref at a loop header. To reproduce, the array must stay partial-STATIC (so the
decode folds) AND be carried into a dynamically-controlled loop where a back-edge
re-materializes it, AND get a dynamic-index write in one iteration that a later
iteration's read should observe. **The fix**: when a loop-carried partial-static
array gets a residualized (dynamic-index) write, force it to FULLY escape to a single
runtime instance that the loop header carries as a `Var` (not re-materialize the
static literal each iteration) — i.e. propagate the escape into the loop-header
generalization / memoized loop state, not just the current path. Hard part: keep the
static folding that lets the decode collapse while it is still fully static; only
escape-and-carry the instant a residualized write happens.

## Key files

- `src/js.rs` — the JS client (eval, heap, escape/materialize, residualize_try,
  do_call, coercion folding, string methods, arguments). ~4600 lines.
- `src/engine.rs` — generic PE (do not change for JS work).
- `src/residual.rs` — residual IR (`Op`, `Terminator`, `Program`).
- `js-frontend/src/lower.rs` — SWC → `partial::js` AST + arguments slots.
- `js-frontend/src/codegen.rs` — residual IR → JS (trampoline, Op::Try, sentinel).
- `js-frontend/src/lib.rs` — public API + the whole test suite (84 tests).
- `js-frontend/src/bin/fuzz.rs` — the fuzzer.
- `tools/fuzzcmp.js`, `tools/difftrace.js` — the oracles.
- `docs/adversarial-deobfuscation.md` — deeper lessons (single-eval, ordering).

## Invariants / gotchas

- The generic engine is never changed for JS-specific soundness.
- Run `node` is required for the fuzzer/difftrace oracles.
- A stub that returns a wrong value (e.g. `arguments` → undefined) is forbidden
  (Jimmy's rule): make it correct or a clear hard error.
- The number model is **i64-only** (no float/NaN); `/` and `%` are never folded.
  Coercions yielding NaN/floats are sound but residualized, not folded.
- difftrace's `Date` is a proxy; for the true simple.js failure, run the residual
  with real globals and read the stack trace.
