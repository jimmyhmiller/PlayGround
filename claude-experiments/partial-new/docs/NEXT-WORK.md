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
  `cargo test -p js-frontend --release` (111).
- Fuzzer: `cargo run -p js-frontend --release --bin fuzz -- --seed 1 --count 2000
  --batch 250 --no-shrink`. It sets `SPEC_WEIGHT_BUDGET=100000` itself. A full 2000
  run completes at ~620 MB. `--repro SEED` prints one program + residual + findings;
  `--seed N --count 1 --batch 1` shrinks a single seed.
- simple.js (`/Users/jimmyhmiller/Documents/Code/deob/simple.js`) is the hardest real
  input and the regression anchor: after any eval-core change it must keep specializing
  cleanly (`rc=0`, weight ~141M) and the residual must stay correct (observationally
  equivalent to the original). Check with
  `SPEC_WEIGHT_BUDGET=999999999 ./target/release/js-frontend --js <simple.js>` and, for
  behavior, run the residual against the original (see `docs/HANDOFF.md` for the
  difftrace/run-residual tooling). The residual is large, so a quick smoke check that
  it didn't change shape is fine, but the *requirement* is correctness, never identical
  output text — a change that only renames residual temps or reshapes the residual is
  fine as long as behavior is preserved.

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
`negative_num_index_base_is_parenthesized`. simple.js has zero negative-`Num` member
bases, so this change does not affect it.

---

## Track A2 — Method-callee detached by freeze → over-throw — DONE (2026-06-03)

**Status: fixed.** Five of the six over-throws (seeds 661, 1374, 1763, 2737, 5563)
were this class and are now resolved; the sixth (2368) turned out to be a *different*
bug (closure capture staleness — see Track A3). simple.js still specializes correctly;
tests 12 + 104 green; the fuzzer sweep removed exactly those 5 seeds with zero new
divergences.

**The fix (landed in `src/js.rs` + `js-frontend/src/codegen.rs`).** New
`RExpr::BoundMethod { func, recv }`. In `freeze_readers`, when an **operand-stack**
value being frozen is a top-level method access (`recv.m` / `recv[i]`) with an *atom*
receiver, snapshot the method VALUE to the freeze temp (so a value use sees the same
pre-call snapshot a plain freeze would give) but wrap it as `BoundMethod { func:
Var(temp), recv }` instead of a bare `Var`. A `BoundMethod`:
- decays to `func` for any value use (`to_rexpr`, codegen, `fmt`) — JS detaches
  `this` once a method is read into a value, so this is correct;
- is treated as ref-like (`rexpr_is_ref_like`) so it never gets wrapped in parens as
  a member base;
- when it reaches `do_call` as a callee, emits `func.call(recv, args)` — the
  universal "uncurry this" form, correct even when `m` is itself `.call`/`.apply`
  (`(f.call).call(f, a, b)` === `f.call(a, b)`).

Regression test: `frozen_method_callee_keeps_receiver`.

**Known limitation.** Only an *atom* receiver (`Var`/`Str`/`This`/literals) is
preserved. A **non-atom** receiver (`g.x.m(...)`, `arr[i].m(...)` with an intervening
freeze) and a **`Global`** receiver (`Math.abs(...)` — harmless since those methods
ignore `this`) fall back to the plain freeze (this lost). No fuzzer divergence hits
these today. The complete fix snapshots the receiver to its own temp: that needs a
third `freeze_var` `idx` band, so widen `freeze_var`'s `idx` multiplier (purely
renumbers the residual temps — verify simple.js by running it, not by comparing
output text) and snapshot the receiver alongside the method value.

<details><summary>Original analysis (kept for reference)</summary>

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
`do_call` path — **gate on the full test suite AND simple.js still specializing
correctly** (simple.js makes heavy method calls; run its residual to confirm behavior
is preserved). Build a hand `node -e` over-throw test first, then the bound-method
shape, then re-verify simple.js by running it.

</details>

---

## Track A3 — Stale closure capture across reassignment — DONE (2026-06-03)

**Status: fixed.** Root cause: the frontend boxed captured+mutated `var` LOCALS into
`{value}` cells (capture-by-reference) but NOT captured+mutated PARAMETERS — those
were declared by-value (and regular functions even hard-errored on them: "capture-by-
reference of parameters is not supported"). So a captured parameter reassigned by the
outer scope (2368) or by the closure (3098) used a stale by-value snapshot.

**The fix (all in `js-frontend/src/lower.rs`).** `boxed_locals(params, body)` now treats
parameters as boxing candidates too; the param-binding loops in `lower_module_main`,
`lower_fn`, and `lift_arrow` box a captured+mutated param (recorded in
`ctx.boxed_params`) instead of erroring; `boxed_cell_stmts` initializes a boxed
*param* cell from its incoming argument (`p = {value: p}`, the inner read taking the
raw arg before the slot is rebound) rather than from `undefined` like a hoisted local.
Regression tests `captured_param_outer_reassign_is_shared`,
`captured_param_closure_write_is_shared`. simple.js unaffected (param boxing touches
every function but its captured params, if any, weren't reassigned across a closure).

<details><summary>Original analysis (kept for reference)</summary>

**Minimal repro (seed 2368).**
```js
function main(input) {
  var a0 = function (c1p0) { return Math.abs(input.toString(true)); };
  input = false;          // reassigns the captured variable
  return a0(input);       // a0's body should now see input === false
}
// main(0): original computes Math.abs(false.toString(true)) = Math.abs("false") = NaN
//          (JSON null); residual computes (0).toString(true) → RangeError (radix 1).
```
Residual: `v500016 = v0.toString(true)` uses `v0` (main's original `input` = 0), not
the reassigned `false`. `(0).toString(true)` throws RangeError (radix must be 2–36);
`false.toString(true)` ignores the radix and returns `"false"`, no throw.

**Root cause.** The closure captures the abstract *value* of `input` at creation
(`Var(v0)`), so the later `input = false` (which updates main's slot to `Bool(false)`)
isn't reflected when `a0` is inlined. JS closures capture the variable *cell*. Fixed by
boxing captured+mutated parameters (see above).

</details>

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

Each addition is local (one arm) + a Node-verified test. Keep simple.js specializing
correctly after each.

---

## Track C — Remaining real PE issues (harder, lower priority)

1. **Dead-store / dead-element may-throw (under-throw) — DONE (2026-06-03).**
   `var x = undefined.length;` (x unused), `[0, a.b].length` (element dropped): the
   throwing access was dropped because its value was never materialized. Now fixed at
   the *evaluation point*, not via liveness:
   - **`Instr::Pop`** (discarded statement) — fixed earlier (`may_throw` + `Op::Eval`).
   - **`Instr::Store`** — a may-throw RHS (`a.b`, `a[i]` over a runtime/null base) is
     committed as a residual assignment in program order, then the slot reads back as
     that var. Covers dead stores and conditionally/late-read stores, in any frame
     (an earlier `eager_stores`-only path was frame-1). Emits the whole RHS, so
     short-circuiting stays intact (this is why the old "eager-emit at
     `GetProp`/`GetIndex`" was wrong — it hoisted out of `&&`/`?:`).
   - **`Instr::NewArray`/`NewObject`** — each may-throw element is emitted for effect
     left-to-right at construction (`emit_maythrow_elems`), single-evaluated into a
     temp. Static literals have no `Dyn` element, so they still fold. Known gap: a
     may-throw element *before* a separately-effectful element in the *same* literal
     emits after that effect (no test hits it; needs a definitely-evaluated-context
     flag for full order).

   Result: fuzzer base 1 and base 5000 are clean; ~19 seeds fixed. Regression tests
   `dead_store_may_throw_is_preserved`, `dead_literal_element_may_throw_is_preserved`.
   The 3 originally-open seeds (2368, 2791, 3098) were fixed: 2791 here (a dead
   *boxed-cell* write, `SetProp`/`SetIndex` on a static object — now also routed
   through `emit_if_maythrow`), and 2368/3098 by Track A3 (param boxing). Two more
   contexts were closed while extending the model: a may-throw **store inside a
   `try`** (the probe must taint so the `try` residualizes and catches it — now
   `Instr::Store` calls `forbid_residual_in_try`; seed 12359) and a may-throw
   **`return` value inside a `try`** (`Instr::Ret` now taints likewise; seed 31430).
   ALL of bases 1/2001/5000 (6000 programs) plus the originally-found fresh seeds are
   clean.

1b. **Wider fuzz frontier (bases 10000–40000, NOT regressions).** Mostly cleared:
   - **dead may-throw in a CALL ARGUMENT** (seed 10125) — FIXED: `do_call`'s inline
     path commits a may-throw arg via `emit_if_maythrow` before binding (args are
     evaluated even when the callee ignores them; the opaque arms already kept them).
   - **`a[i]` where the index mutates the base** (seed 10899) — FIXED: this was an
     effect-ORDER bug, not may-throw. `a0[a0--]` inside a `try` emitted the `a0--`
     reassignment without snapshotting the index base `Var(a0)` already on the operand
     stack, so the base read the post-mutation value (`NaN[NaN]` instead of
     `undefined[NaN]`). `Instr::Store`'s emit path now freezes operand-stack readers
     of the reassigned var first (a bare `Var(id)` IS frozen here, unlike
     `freeze_readers`). Tests `dead_call_argument_may_throw_is_preserved`,
     `index_base_read_before_mutating_index`.

   STILL OPEN (2 of 14000 programs; harder/different classes):
   - **`arr.push(v)` evaluation order** (seed 20968). Root cause (traced to the exact
     lowering): `Stmt::Push(arr, v)` compiles as `arr; v; PushArr` (`src/js.rs`
     ~line 660), but JS evaluates the member `arr.push` (which throws if `arr` is
     null/undefined) BEFORE the argument `v`. So when `arr` is undefined and `v` has a
     side effect (`a5++`), the residual runs `a5++` first (a5 → NaN) then throws;
     downstream the `catch` does `a5[obj]=null`, a silent no-op on `NaN` but a throw on
     `undefined` — hence the divergence. (Confirmed with `node`: original leaves
     a5=undefined, residual leaves a5=NaN.) The proper fix commits `arr`'s pushability
     exception before `v`. It is genuinely awkward: (a) you cannot just emit
     `arr.push` first — `GetProp` on a static array ESCAPES it, defeating the modeled-
     push optimization simple.js relies on; (b) the receiver here is a dynamic merge
     var that is undefined only on this path, so a "statically nullish" check doesn't
     fire; (c) a correct null-check-before-`v` for dynamic receivers changes the
     residual shape broadly. A tried snapshot-time `freeze_before_opaque_call` did
     NOT help (the `arr.push` member is never a distinct operand in the push form) and
     was reverted. Wants its own focused effort + a simple.js behavior check.
   - **`.toString()` on a residualized closure** (seed 41158): a `value` mismatch, not
     a throw — the residual closure renders as `__rfN`/native, so its source text
     differs from the original. A representation leak; likely a documented limitation
     unless residual closures carry original source.

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
  clean-context), `SPEC_WEIGHT_BUDGET` (budget), `freeze_readers`/`do_call` +
  `RExpr::BoundMethod` (Track A2 method-callee `this` preservation; `rexpr_is_ref_like`
  treats it ref-like, `rexpr_to_js`/`fmt_rexpr` decay it to `func`). The generic engine
  in `src/engine.rs` is NEVER changed for JS-specific work.
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
