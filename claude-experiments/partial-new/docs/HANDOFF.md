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
  byte-unchanged (the fixes fold to no-ops on its inputs).

## How to run things

```bash
# build
cargo build --release                       # engine (partial) + frontend lib
cargo build -p js-frontend --release         # frontend + js-frontend bin
cargo build -p js-frontend --release --bin fuzz

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

## Remaining issue: throw preservation in dead/discarded positions (SOUNDNESS)

The dominant remaining divergence class (14 of 15 across seeds 1–2000): the residual
silently skips an exception the original raises. The PE correctly residualizes a
may-throw access when its value is **used** (`return undefined.length` → throws at
runtime) and now when it is **`Pop`-discarded** (fix #4). It still drops it when the
value flows into a **dead position**:
- **dead store**: `var x = undefined.length; return 0;` (x never read) → `return 0`.
- **dead array/object element**: seed 539 — `input = [{b: a8.length}, …]` where
  `input` is then unused.
- **residual-function body**: seed 228 — `function f3(){ return a3(true); }` (a3
  undefined → TypeError) residualizes to an **empty** `__rf0` body; the throwing
  call is dropped.

The unifying fix is to treat a may-throw operation as an effect that must be emitted
in program order regardless of how (or whether) its value is consumed — extend the
`may_throw`-emit logic from the `Pop` site to dead-store elimination, escaped
array/object element construction, and residual-function-body specialization. Gate
on the full test suite + the Node fuzzer; simple.js must stay unchanged. The two
non-throw-class stragglers: seed 22 (deeply contrived nested case) and the lone
function-stringification value mismatch (likely a fuzzer-oracle artifact — comparing
`[native code]` reps).

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

## Remaining issue #2: simple.js effect #4 (WRONG ANSWER, deep, no minimal repro)

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
`main`'s loader loop, not a has_try closure. **Next step**: reproduce the
cross-handler / cross-iteration decrypt-then-read ordering minimally (hand-crafted
repros so far all fold correctly; needs the array escaped + write in one residual
region + read in another with the read emitted first). Once minimal, fix like the
capture-escape bug.

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
