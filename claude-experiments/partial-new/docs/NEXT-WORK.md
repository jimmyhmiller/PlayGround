# Next work — partial-new (JS partial evaluator)

Forward-looking handoff. Read `docs/HANDOFF.md` first for the current state (what
is fixed, what the fuzzer/budget do, the soundness invariants). This doc is the
backlog: the concrete next tracks, where they live in the code, why, and how to
verify. Pick any track independently.

Standing facts you need:
- Oracle = Node(original) vs Node(residual). The in-process Rust `check` (run_reference
  vs run_residual) is BLIND to whole bug classes because the reference interpreter
  shares the evaluator's semantics — always confirm soundness with the fuzzer / a
  hand `node -e`, not just `check`.
- Test gate (keep green): `cargo test -p partial --release` (12) and
  `cargo test -p js-frontend --release` (103).
- Fuzzer: `cargo run -p js-frontend --release --bin fuzz -- --seed 1 --count 2000
  --batch 250 --no-shrink`. It sets `SPEC_WEIGHT_BUDGET=100000` itself. A full 2000
  run completes at ~620 MB. `--repro SEED` prints one program + residual + findings;
  `--seed N --count 1 --batch 1` shrinks a single seed.
- simple.js (`/Users/jimmyhmiller/Documents/Code/deob/simple.js`) is the hardest real
  input and the regression anchor: it must keep specializing (`rc=0`) and stay
  byte-unchanged (weight ~141M). Check after any eval-core change:
  `SPEC_WEIGHT_BUDGET=999999999 ./target/release/js-frontend --js <simple.js>`.

---

## Track A — Fuzzer / shrinker hygiene — DONE (2026-06-03)

**Status: complete.** The undeclared-variable artifact is gone; every remaining
divergence is now a genuine PE soundness gap (see Tracks A2 + C1). Three fixes
landed in `js-frontend/src/bin/fuzz.rs` (+ one codegen bug in
`js-frontend/src/codegen.rs` that Track A immediately surfaced):

1. **Generator no longer emits out-of-scope `input`.** `Gen::atom()` now emits
   `Expr::Var("input")` only when `input` is in scope
   (`self.vars.iter().any(|(n,_)| n == "input")`). Inside a top-level
   `func`/`rec_func` (`std::mem::take(&mut self.vars)`) it falls back to a literal.

2. **Generator scopes the `catch` binding.** The try/catch arm pushed the catch
   parameter onto `self.vars` and never removed it, so statements *after* the
   try/catch could read it out of scope (a ReferenceError the PE models as a
   non-throwing `Global`). It now `self.vars.retain(|(n,_)| n != &cv)` after the
   catch clause — body/catch `var`s stay (function-scoped), only the block-scoped
   catch binding is dropped.

3. **Shrinker rejects reductions that read an undeclared name.** `still_fails`
   now calls `reads_undeclared(p)` first: it collects every bound name
   (`declared_names` — params, `var`s, loop vars, `while` guards, catch bindings,
   function names) and every *read* reference (bare `Var`, `Update` target, `Push`
   target, and a `Call` callee that isn't a `SAFE_CALLS` builtin), and rejects the
   reduction if any read is undeclared. This stops the shrinker drifting a real
   divergence down to a `void a1` / `a8.length` / `r0(...)`-undefined artifact.
   (A plain `AssignVar` LHS is intentionally NOT a read — `undeclared = v` is a
   silent global in non-strict JS, which the PE's `Global` model already matches.)
   Caveat: `declared_names` is a *global* set, not lexically scoped, so an
   out-of-scope-but-declared read (e.g. a catch binding read elsewhere) isn't
   rejected by the shrinker — fix #2 prevents the generator from producing those,
   so it hasn't mattered. If a future generator change reintroduces block scoping,
   make the shrinker check scope-aware.

**Result.** seed-base 1/2001/5000 (6000 programs) report 7/13/7 divergences, all
real: ~20 under-throws + 1 termination outlier (dead-store may-throw, Track C1) and
6 over-throws (method-callee detach, Track A2). Re-verify with
`./target/release/fuzz --seed 1 --count 2000 --batch 250` and classify a seed with
`--seed N --count 1 --batch 1`.

**Codegen bug found + fixed en route (seed 1865).** A negative number literal used
as a computed-member base rendered as `-1[i]`, which JS parses as `-(1[i])` (member
access binds tighter than unary minus): `(-1)[i]` is `undefined`, `-(1[i])` is `NaN`.
Fixed with `index_base_js` in `codegen.rs` (parenthesize a negative `Num` Index base;
`render_get`/`Bin`/`Opaque` were already safe). Regression test
`negative_num_index_base_is_parenthesized`. simple.js has zero such bases, so it is
byte-unchanged.

---

## Track A2 — Method-callee detached by freeze → over-throw (NEW, real, higher priority)

**Why it matters.** This is an OVER-throw (the residual raises an exception the
original never does) and, for non-`.call`/`.apply` methods, a silent `this`-corruption
(a wrong *answer*) — strictly worse than the under-throws of Track C1. Six fuzzer
divergences (seeds 661, 1374, 1763, 2368, 2737, 5563) are all this one class.

**Minimal repro (seed 661).**
```js
function f0(p0) { return (null ? Math.floor(1) : (-("k" ?? 3))); }
function main(input) { input = f0.call(null, (Math.floor(false) * null)); return 0; }
// main(0): original returns 0; residual throws TypeError.
```
Residual `main`:
```js
v100000  = __rf0;
v2176128 = v100000.call;                 // <-- method callee hoisted to a temp
v500021  = Math.floor(false);
v500024  = v2176128(null, v500021 * null); // <-- called BARE: this === undefined
```
`f0.call(null, arg)` was lowered into `tmp = f0.call; tmp(null, arg)`. Calling the
detached `.call` with `this = undefined` throws `TypeError`. The same shape hits any
method whose receiver matters (1374 `"ab".toUpperCase(... opaque ...)`, 1763
`[...].concat(a2(...))`).

**Root cause (traced).** In `do_call` the callee is popped *before*
`freeze_before_opaque_call`, so the freeze never touches the *current* call's callee.
But when an **argument** expression contains its own opaque call (`Math.floor(false)`,
`a2(...)`, `input.toString(...)`), evaluating that inner call runs
`freeze_before_opaque_call` while the *outer* method callee `Get(obj, "call")` is still
sitting on the operand stack. `freeze_readers` hoists any ostack/local `Abs::Dyn(e)`
whose `e` contains a `Get`/`Index`/`Global` to a temp — so it collapses the pending
`Get(__rf0, "call")` to `Var(v2176128)`, discarding the receiver. The eventual outer
`do_call` then emits `Call(Var(v2176128), args)` = a bare call: `this` is lost.

**Why the obvious fixes are wrong.**
- *Make `freeze_readers` preserve `Get` structure* (`tmp = obj; ... tmp.m(args)`):
  weakens the freeze's property-snapshot guarantee. The whole point of freezing
  `obj.m` (fix #4, `freeze_before_opaque_call`) is that an intervening opaque call
  must not change the value read; re-reading `tmp.m` after the call would see a
  mutated property. Do NOT do this.
- *Skip freezing callees*: not knowable at freeze time (freeze can't tell which
  ostack entry becomes a callee).

**Proposed fix (bound-method desugaring).** Freeze a method access destined for a
call by snapshotting BOTH halves and preserving `this`:
```
t_recv = obj;          // snapshot receiver
t_fn   = t_recv.m;     // snapshot the method (property read, before the inner call)
...                    // intervening opaque call
t_fn.call(t_recv, args)   // or  t_recv.m(args) if t_recv is still pinned
```
This keeps the property-snapshot guarantee (`t_fn` captured pre-call) AND the correct
`this`. Carrying the receiver requires either a new `Abs`/`RExpr` "bound method"
shape, or detecting at the freeze of a `Get(obj,m)` that it is a method-access and
emitting `Get(freeze(obj), m)` into a temp paired with a frozen `freeze(obj)` so the
call site can rebuild `t_fn.call(t_recv, ...)`. Touches the load-bearing freeze /
`do_call` path — **gate on the full test suite AND a simple.js byte-diff** (simple.js
makes heavy method calls; any freeze change risks its 14k-line residual). Build a
hand `node -e` over-throw test first, then the bound-method shape, then verify
simple.js is byte-unchanged.

---

## Track B — Flesh out the standard library (the `try_builtin_*` pattern)

**Why.** Modeling stdlib both folds static computation (smaller residuals,
deobfuscation progress) AND marks those globals as known/non-throwing. `Math` was
added this session as the template (`src/js.rs`, `try_builtin_static`). The three
extension hooks (all in `src/js.rs`, near the `TextDecoder`/`Math` code):

- `try_builtin_static(obj, method, args)` — static methods on a global,
  `Obj.method(args)`. Examples done: `String.fromCharCode`, `Math.*`.
- `try_builtin_new(name, args, s)` — constructors, `new Name(args)`. Examples done:
  `TextDecoder`, `Uint8Array`. Register the name in `builtin_ctor_name`.
- `try_builtin_method(kind, data, args, s)` — bound instance methods on a modeled
  built-in value (a `HeapObj::Builtin { kind, data }`). Example: `TextDecoder.decode`.
  Also see `try_string_method` for `String.prototype` methods on a *literal* receiver.

**Hard soundness constraints (do not violate):**
- The number model is **i64 only**. Never fold anything producing a float or NaN
  (`Math.sqrt`, `/`, `%`, `**`, `parseFloat`); residualize instead.
- Never fold **non-deterministic** or **effectful** builtins (`Math.random`, `Date.*`,
  any IO). They must pass through to the runtime or observational equivalence breaks.
- String folds must be **ASCII-guarded** (see `try_string_method`'s rationale): char
  index = UTF-16 index, ASCII case-mapping = JS's. Non-ASCII → residualize.
- Anything you fold must be **Node-verified**. Add a `node -e` check and a regression
  test. Reference-interpreter `check` can't run `Global`-based programs (it panics on
  `PushGlobal`), so test folded results via `run_residual` against the constant, and
  pass-through behavior via `to_js(...).contains("Math.sqrt")` etc.

**Good candidates (deterministic, integer/string, common in obfuscators):**
- `Number(x)`, `parseInt(x, radix)` static folds (careful with radix/NaN → residualize
  on non-fold).
- `String.prototype` methods already partly done (`try_string_method`); extend the set
  / lift the literal-receiver restriction where sound.
- `Array.prototype` methods on a *static* array (`join`, `indexOf`, `includes`,
  `slice`, `concat`, `map`/`filter`/`reduce` with a foldable callback) — these need
  `&mut State` to allocate result arrays; model like `try_string_split`.
- `Object.keys`/`Object.values`/`Object.entries` on a static object.
- `JSON.stringify`/`JSON.parse` on fully-static values (big win for deobfuscation;
  watch escaping + the i64 number model).
- `String`/`Boolean` coercion calls (`String(5)` → "5") — partly covered by
  `to_string_static`; route the *call* form through `try_builtin_static`.

Each addition is local (one arm) + a Node-verified test. Keep simple.js green/byte-
unchanged after each.

---

## Track C — Remaining real PE issues (harder, lower priority)

1. **Dead-store / dead-element may-throw (under-throw).** `var x = undefined.length;`
   (x unused), `[{b: a8.length}]` (array unused): the throwing access is dropped
   because its value is never materialized. FIXED for the `Pop`-discard case
   (`Instr::Pop` + `may_throw` in `src/js.rs`). The dead-*store* case is unfixed.
   DO NOT re-try the naive "eager-emit the access at `GetProp`/`GetIndex`" — it was
   tried and reverted in an earlier session because it hoists the throw out of the
   compact `&&`/`||`/`?:` short-circuit (pure operands evaluate eagerly), causing
   OVER-throws. A correct fix needs either (a) liveness to know a store is dead, or
   (b) a may-throw *effect* model that emits at the discard/materialization boundary
   without breaking short-circuiting.
   **Now the dominant visible class (Track A is done).** ~20 of the 27 fuzzer
   divergences are this, with clean minimals — e.g. seed 533
   `input = null; input = input.b; return 0;` (`null.b` throws, the dead store to the
   unread `input` is dropped) and seed 2037 `input = null; a5 = input[5]; return 0;`.
   One outlier (seed 3024) is the same root cause with a *termination* symptom:
   `orig=throw, spec=timeout` — the dropped throw was what terminated a loop, so the
   residual spins. Use these as the repro set; pick the cleanest (533) to build the
   liveness-or-effect fix against.

2. **continue/break out of a residualized `try`/`catch`** — still a deliberate,
   catchable refusal (`residual_try_stack` in `src/js.rs`, ~line 1009 / the panic in
   `residualize_try`). Scoped design is in `docs/HANDOFF.md` ("continue-out-of-
   residual-try fix DESIGN") and the memory note — needs a non-local-exit target on
   `Terminator::Halt` + a pc→outer-block map. Risky; gate on the residual-try tests.

3. **simple.js effect #4** — the real deobfuscation endpoint (an effect-ordering bug
   in the residualized lazy-decrypt loop). Fully traced in `docs/HANDOFF.md`
   ("Remaining issue #2"). Needs a minimal cross-region decrypt-then-read repro.

4. **OOM robustness tail.** The `SPEC_WEIGHT_BUDGET` (weight = Σ residual-expr nodes
   over block entries) is an imperfect memory proxy — bytes/weight varies ~2000x
   across programs. 100k works for the current generator, but a truly robust guard is
   subprocess `RLIMIT_AS` in the fuzzer (the `--specialize-file` worker already exists
   for the overflow shrinker; extend it with a memory cap and classify OOM as a
   resource outcome).

---

## Map of what to touch

- `src/js.rs` — the evaluator/client. `try_builtin_static`/`_new`/`_method` (stdlib),
  `compile_update`/`compile_short_circuit` (coercion/short-circuit), `may_throw` +
  `Instr::Pop` (discard throws), `residual_fn_for` (residual fn bodies, now
  clean-context), `SPEC_WEIGHT_BUDGET` (budget). The generic engine in `src/engine.rs`
  is NEVER changed for JS-specific work.
- `js-frontend/src/codegen.rs` — residual IR → JS. `index_base_js` parenthesizes a
  negative-`Num` computed-member base (Track A codegen fix); `rexpr_to_js` is the
  per-node emitter.
- `js-frontend/src/lower.rs` — SWC → `partial::js` AST. Statement-position `++`/`--`
  coercion lives here (`lower_expr_stmt`).
- `js-frontend/src/bin/fuzz.rs` — generator + shrinker (Track A, done). Artifact
  guards: `Gen::atom` (in-scope `input`), the try/catch arm (`vars.retain` to scope
  the catch binding), and `reads_undeclared`/`declared_names`/`collect_reads_*`/
  `collect_decls_*` feeding `still_fails`.
- `js-frontend/src/lib.rs` — public API + the test suite (102 tests). `SPEC_STEPS`
  env prints spec_steps/blocks/weight (calibration).
- `tools/fuzzcmp.js`, `tools/difftrace.js` — the Node oracles.
