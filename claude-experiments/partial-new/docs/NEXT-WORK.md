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
  `cargo test -p js-frontend --release` (102).
- Fuzzer: `cargo run -p js-frontend --release --bin fuzz -- --seed 1 --count 2000
  --batch 250 --no-shrink`. It sets `SPEC_WEIGHT_BUDGET=100000` itself. A full 2000
  run completes at ~620 MB. `--repro SEED` prints one program + residual + findings;
  `--seed N --count 1 --batch 1` shrinks a single seed.
- simple.js (`/Users/jimmyhmiller/Documents/Code/deob/simple.js`) is the hardest real
  input and the regression anchor: it must keep specializing (`rc=0`) and stay
  byte-unchanged (weight ~141M). Check after any eval-core change:
  `SPEC_WEIGHT_BUDGET=999999999 ./target/release/js-frontend --js <simple.js>`.

---

## Track A — Fuzzer / shrinker hygiene (so the divergence count means something)

**Why.** After this session's fixes, a full 1–2000 run reports ~14 divergences, but
they are dominated by an **artifact, not a PE bug**: programs that READ AN UNDECLARED
VARIABLE. In JS that throws ReferenceError; the PE models every unresolved identifier
as a (non-throwing) `Global`, which is the *correct* choice for real globals (`Math`,
`String`) that real code and simple.js depend on. The PE cannot distinguish `a1` from
`Math`, and valid code never reads undeclared vars — so this is not a soundness bug to
fix in the evaluator. It just drowns out real bugs in the fuzzer output. Two sources:

1. **Generator emits out-of-scope references.** `js-frontend/src/bin/fuzz.rs`,
   `Gen::atom()` hardcodes `Expr::Var("input")` (the `_ =>` arm: "Always include
   `input` so programs actually depend on it"). Inside a top-level `func`/`rec_func`,
   `self.vars` holds only the params (`std::mem::take(&mut self.vars)`), so `input`
   there is undeclared. **Fix:** only emit `Expr::Var("input")` when `input` is in
   scope, i.e. `self.vars.iter().any(|(n,_)| n == "input")`. (For `main` it always is;
   inside functions it usually isn't.) That removes the only out-of-scope identifier
   the *generator* itself produces — all other identifiers come from `vars_of`/
   `assignable_vars`, which are in-scope by construction.

2. **Shrinker drifts to a different failure.** `shrink()` / `reduce_*` in
   `fuzz.rs` reduce to *any* still-failing program. A very common reduction is
   "delete a `var` declaration", which leaves its references dangling → the reduced
   program "still fails" via ReferenceError on the now-undeclared variable, which is a
   *different bug* than the original. So shrunk minimals over-represent the artifact
   and mislead diagnosis (this bit me repeatedly this session — seeds 231/539 shrank
   to undeclared-var programs whose original cause was elsewhere). **Options:**
   - Cheapest: in `still_fails`, reject a reduction whose residual references an
     identifier that the program never declares (i.e. would be a ReferenceError) — or
     more simply, reject reductions that turn a previously-declared name undeclared.
   - More general: record the *kind+site* of the original divergence and only accept a
     reduction that still fails the same way. Harder; the comparator currently returns
     value/throw-mismatch/etc. but not a stable "site".

**Verify.** After the generator fix, re-run the 2000 fuzz; the divergence count should
drop toward the genuinely-real bugs (e.g. the contrived seed 22). After the shrinker
fix, the shrunk minimals should stop collapsing to `void a1` / `a8.length`-style
undeclared-var programs.

**Note:** there is also a real-but-hard PE under-throw hiding behind the artifact —
see Track C (dead-store may-throw). Track A makes it visible; it does not fix it.

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
   tried and reverted this session because it hoists the throw out of the compact
   `&&`/`||`/`?:` short-circuit (pure operands evaluate eagerly), causing OVER-throws.
   A correct fix needs either (a) liveness to know a store is dead, or (b) a may-throw
   *effect* model that emits at the discard/materialization boundary without breaking
   short-circuiting. Note this overlaps heavily with the undeclared-var artifact
   (Track A) — do Track A first so you can see how many dead-store cases are real.

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
- `js-frontend/src/lower.rs` — SWC → `partial::js` AST. Statement-position `++`/`--`
  coercion lives here (`lower_expr_stmt`).
- `js-frontend/src/bin/fuzz.rs` — generator + shrinker (Track A).
- `js-frontend/src/lib.rs` — public API + the test suite (102 tests). `SPEC_STEPS`
  env prints spec_steps/blocks/weight (calibration).
- `tools/fuzzcmp.js`, `tools/difftrace.js` — the Node oracles.
