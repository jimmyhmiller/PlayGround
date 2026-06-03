# CFG fidelity: what our analysis IR drops, how to prove it, and how to fix it

## STATUS (2026-06-02): Instrument 1 built; it already caught two silent miscompiles

Instrument 1 (the static completeness audit, "check A" below) is **built and
green**:

- `src/lower.rs::HANDLED_OP_KINDS` — authoritative list of op-kinds the lowering
  recognizes (mirrors the `lower_op` match arms).
- `src/fidelity.rs` — classifies every base-IR op-kind as `Faithful` /
  `HardError` / `KnownLossy(reason)` / `Ignored(reason)` / `StructuralWrapper`,
  with a reviewed `KNOWN_LOSSY` + `IGNORED_FAITHFUL` registry.
- `tests/fidelity.rs` — over the 1127-fixture corpus: (1) snapshot ratchet
  (`tests/fidelity_snapshot.txt`; regen with `JSIR_FIDELITY_REGEN=1`) so a new
  op-kind or a fidelity-bucket change must be reviewed; (2) an **empirical
  bail-check** that asserts each `hard-error` op-kind really bails and each
  `ignored` one really lowers (so the classification is tested, not assumed).
- `examples/fidelity_scan.rs` — dumps op-kind frequency across the corpus.

Current classification: 77 op-kinds = 43 faithful, 26 hard-error, 4 known-lossy,
2 ignored, 2 wrapper.

**Two real silent miscompiles it surfaced (now fixed to loud bails):** the param
loop in `lower_function` only handled `identifier_ref` params and *silently
skipped* every other param shape while still counting the pattern's nested
`identifier_ref`s as positional params. Result:
- `function f({a=1}){return a}` lowered to **zero params** returning `undefined`.
- `function f(a,...rest){return rest}` treated `rest` as one positional param.
Both now `Err` loudly. This is the single largest coverage chunk: **74 fixtures
use destructured (`{a,b}`) params**, so faithfully lowering them is the
highest-leverage *fix* the audit has prioritized. Bailing (vs the old
silent-skip) moved the corpus gate `agree 87 -> 80` — those 7 were structural
agreements on value-wrong CFGs; the proper recovery is faithful param lowering,
which Instrument 2 will then validate behaviorally.

**Faithful destructured-param lowering landed** (the fix the audit prioritized).
`{a,b}` / `[x,y]` / mixed params now lower through the same `bind_target`
machinery as `const {a}=props` (each param root binds a fresh positional value;
member reads emitted at entry). Defaults (`{a=1}`), rest (`...a`), and nested
patterns still bail loudly. Corpus gate: `agree 87 -> 88` (and now on correctly
lowered functions, not value-wrong CFGs that matched by luck); ~44 fixtures
moved out of the coverage gap; 0 panics.

Instrument 2 (the behavioral differential + fuzzer) is **built and green**:
`tests/differential.rs` runs Node vs CFG-interp with object/array argument
support and three program sources:
- a **destructured-param battery** (validates the new param lowering vs Node);
- a **closure red-spec battery** (`closures_are_known_lossy_today`) that PINS the
  closure-body loss: CFG-interp must refuse these for a registered reason, and
  the test fails loudly the day closure bodies become reachable (forcing the
  registry + test to be updated together);
- a **seeded fuzzer** (`fuzz_cfg_matches_node`, `JSIR_FUZZ_N` to scale): 1500/1500
  generated programs (simple+destructured params, arithmetic, if/else, bounded
  loops, object/array mutation) agree with Node, 0 bailed.

A negative control (reinstating the old silent-skip param behavior) was verified
to make the differential fail immediately (`node=n:7 ours=n:NaN`), proving the
instrument actually bites rather than passing vacuously.

**Closure bodies are now lowered into nested CFGs (interpreter path).** Additive
and analysis-untouched: the React analyses keep the opaque `MakeArray` +
`closure_mutates` view (so the corpus gate is byte-identical: 88/141/112/320, 0
panics), while `Cfg::nested` + `Cfg::closures` carry the real body and captures
*for the interpreter only*. Captured objects share their `Rc`, so a closure that
mutates a captured object is observable after the call.

- Faithful + differential-validated: **function expressions** (params + captures
  bound) and **no-param arrows**. `closures_with_captured_mutation_match_node`
  (28 pairs) is green; the fuzzer is 1000/1000.
- Still a known loss (pinned by `arrow_params_are_known_lossy_today`): **arrows
  WITH parameters**. An arrow's params are hoisted into the enclosing region and
  not reachable from the arrow op, so the nested body would have unbound free
  variables; a soundness check in `lower_nested_closure` detects this and bails
  to a known loss (the interpreter refuses to call it) rather than miscompile.

Bugs the differential caught while building this (all fixed): own-param `x` of an
arrow mistaken for a capture (fixed via def-scope vs read-scope), and the
arrow-vs-function-expression region-layout difference (`[body]` vs
`[params, body]`).

The closure `KNOWN_LOSSY` registry entries were **reworded, not retired**: the
body loss is closed for the interpreter, but (1) the React analyses still consume
the coarse view rather than the body's per-capture effects, and (2) arrow params
remain opaque. Retiring them is the next step.

## STATUS (2026-06-02, cont.): KNOWN_LOSSY is now EMPTY

Every base-IR op-kind is now either **faithfully lowered** (and behaviorally
validated by the differential + the 2000-program fuzzer) or a **loud
hard-error**. `fidelity::KNOWN_LOSSY` is `&[]`; the snapshot has zero
`KNOWN-LOSSY` rows. What closed:

- **Closures, all forms** (arrow / function expression / object method): bodies
  lowered into nested CFGs and executed with captures shared by reference. Every
  parameter shape works: positional, destructured (`{a}`/`[x,y]`), and **arrow
  parameters** (recovered from the arrow op's operands, which is where the
  hoisted param `identifier_ref`s actually live) including destructured arrow
  params. 48 hand cases + fuzzer-generated closures all agree with Node.
- **Spread** (`[...a]`, `{...o}`, `f(...a)`): the interpreter splats/merges
  faithfully via `Cfg::spread_positions` (a side-table the analyses ignore, so
  their view is unchanged). Differential-validated.

The corpus gate stayed **byte-identical** (88/141/112/320, 0 panics) through all
of it: the closure/spread faithfulness lives in interpreter-only side-tables
(`nested`/`closures`/`spread_positions`), the analyses' representation is
untouched.

## STATUS (2026-06-02, cont.): analyses migrated onto the bodies; crutches deleted

The analyses now consume the faithful nested bodies instead of the coarse
side-tables:

- **`closure_mutates` is deleted** (field, the `collect_captures` mutation scan,
  and all stores). It was computed-but-never-consumed dead weight.
- **Mutability derives closure mutation from the body**: for each closure it runs
  the same mutable-range analysis on the lowered nested body and sees which
  leading *capture* parameters are mutated; those captures are then marked
  mutated at every call site of the closure. This is the effect the old dead
  `closure_mutates` was meant to carry, now done precisely from the real body
  (transitive through nested calls, since the body analysis handles calls).

Gate: the corpus `agree` set is **identical** (88, proven by a with/without diff
of the agree list — zero REGRESS lines); the full suite is green. The inline
`react_oracle` matched-set went **4/8 → 8/8**: the four
`KNOWN_INTERMEDIATE_OBJECT_GAPS` were already closed by earlier work but masked
because that test skips without `REACT_CC`; running it under the gate surfaced
them and they were promoted into the matched set.

Still open (the larger parity item): closure **dependencies**. Traced precisely
on `capturing-function-member-expr-arguments.js` (mismatch, react=(2,1)
ours=(3,1)):

```
%3 = member %0.router          ; closure captures props.router  (FIRST-LEVEL only)
%4 = array [%3]                ; the closure value
%7 = member %0.router
%8 = member %7.location        ; props.router.location
%9 = array [%8]                ; the useCallback dep array
%10 = call useCallback(%4, %9)
```

Our scope deps = `{%3, %8}` = 2 (one extra cache slot). React's = 1. Root cause:
**our dependencies were keyed by SSA value; React's are keyed by reactive access
path.**

## STATUS (2026-06-02, cont.): path-keyed dependencies landed — agree 88 → 94

The property-path dependency port is done, in two gated steps, **zero REGRESS**:

1. **Dedup scope dependencies by access path** (`scopes.rs::access_path_key`): a
   dep's key is `(root value, [static member names])`, so two reads of
   `props.router.location` count as one dependency. **88 → 93** (+5: the
   `drop-methodcall-use{callback,memo}`, `props-method-dependency`,
   `memoize-primitive-function-calls`, `function-expression-prototype-call-mutating`
   cluster — all same-path double-counts).
2. **Capture closures at FULL property path** (`lower.rs::collect_captures` walks
   the longest member chain, `props.router.location` not first-level
   `props.router`), so a closure's dependency matches the same path read in a
   `useCallback` dep array and dedups to one. **93 → 94** (+1:
   `capturing-function-member-expr-arguments`, the fixture traced above).

Total this session: **agree 88 → 94 (27.6%)**, mismatch 140 → 134, full suite
green, differential + fuzzer still pass (the full-path capture only feeds the
analysis view; the interpreter runs the faithful nested bodies). A path-keyed
*merge-equality* refinement was tried and reverted (corpus-neutral — no fixture
exercises it yet).

The remaining closure-cluster mismatches are now distinct issues (ref-like-name
scope creation, non-reactive dep classification, useCallback set-ref patterns),
each its own gated fix on top of this path-keyed foundation.

### The original caveat: lowering fidelity is NOT React parity

`KNOWN_LOSSY` tracks **lowering fidelity** — does the CFG drop semantically
relevant data, proven by behavioral equivalence to Node. That is now zero.

Separately, the React **analyses** (mutable-range / reactive-scope / deps) still
consume the coarse `MakeArray` + property-granular-capture view of a closure
rather than the nested body's per-capture effects. This is **sound** (never
wrong, only less precise than React) and is the reason the corpus `agree` count
did not move. It is a *parity-precision* item, tracked by the corpus gate, not a
lowering loss — see `mutability.rs` (the `closure_mutates`-not-consumed note) and
PARITY.md. Closing it is the documented coordinated change (effect inference over
the nested bodies + escape-aware dependency analysis, landed together); two prior
*range-patch* attempts regressed the gate, so it is deliberately not a quick
patch. The faithful nested bodies this work added are the prerequisite that makes
it approachable.

## TL;DR

There are **two IRs** in this project and they must not be confused:

1. **Base JSIR / jshir** (`jsir-ir`, produced by `jsir_swc::source_to_ir`). This is
   the reversible IR. It contains the **whole program**, including the bodies of
   every closure. It round-trips to source byte-for-byte (test262-verified). The
   memoization codegen reprints closures from *this* IR, which is why output is
   correct.

2. **The `jsir-ssa` CFG** (`jsir_ssa::lower` → `cfg::Cfg`). This is a separate
   SSA/CFG **analysis view** we build on top of the base IR for the
   React-Compiler analyses (mutable ranges, reactive scopes, effects, types).

**The defect:** the CFG analysis view is *lossy* in ways that are invisible and
unchecked. The biggest loss is **closures: their bodies are discarded** and
replaced by an opaque `MakeArray` of their captured reads. Every analysis that
runs on the CFG is therefore blind to what happens inside a closure. This is the
root cause of the largest cluster of React-parity mismatches (over half of the
250 `mismatch` fixtures involve closures), and every attempt to patch around it
in the layers above (mutable-range hacks) has regressed.

This is not a small calibration bug. It is a **structural information loss** in
the analysis IR, and it violates this project's own rule: *incompleteness must
be a loud hard-error, never a silent lossy summary.*

---

## Proof (reproducible)

Source under test:
```js
function component(a, b) {
  let z = {a};
  let x = function () { z.a = 2; y.b; };
  x();
  return z;
}
```

### 1. The lowering says so, in a comment

`crates/jsir-ssa/src/lower.rs`, the `function_expression` / `arrow_function_expression` arm:

> // A closure is an allocation that captures its free variables. We
> // **do NOT lower the body into the CFG**; ... Represented as `MakeArray` of the
> // captures ...

### 2. The CFG our analysis sees (closure body is gone)

`echo '<source>' | cargo run -q -p jsir-ssa --example dump_cfg`:
```
func(%0, %1) {
  ^bb0:
    ...
    %3 = object {a: %2}        ; z = {a}
    %5 = array [%4]            ; the CLOSURE — just a bag of captures
    %7 = call %6()             ; x()
    %8 = read @2
    ret %8                     ; return z
}
```
There is **no `z.a = 2`** anywhere. The mutation the closure performs is invisible
to mutable-range / scope / effect analysis. The analysis believes `z` is never
mutated after creation.

### 3. The base IR *does* keep the body (round-trips byte-for-byte)

`echo '<source>' | cargo run -q -p jsir-ssa --example rt` (source_to_ir → ir_to_source)
reproduces the closure body exactly:
```js
let x = function() {
  z.a = 2;
  y.b;
};
```
So the information is *present in the base IR* and *thrown away by the CFG
lowering*. The fix does not need new front-end work; it needs the CFG lowering to
stop discarding what it already has access to.

---

## The general bug class: silent lossy lowering

The closure case is the worst instance, but the real problem is a missing
**fidelity contract**. Today `lower.rs` has three kinds of behavior per construct:

- **Faithful** — the CFG represents the construct's full data-flow (`if`, `+`,
  member load/store, object/array literals, calls, ...).
- **Hard-error** — the construct is unsupported and lowering returns `Err(...)`
  (e.g. nested destructuring defaults). This is *acceptable*: the fixture bails
  loudly and is excluded, never silently miscompiled.
- **Silent lossy summary** — the construct is *partially* represented in a way
  that drops semantically-relevant data, with no error. **This is the bug
  class.** Closures (`MakeArray` of captures, body dropped) are the prime
  example. Suspected others are listed in the audit below.

The rule we want: **every construct is either faithful or a hard-error. No silent
lossy summaries.** A silent lossy summary makes the analysis *confidently wrong*,
which is worse than bailing.

---

## How to KNOW the CFG is not wrong (the conformance check)

We currently have no automated way to detect a silent lossy lowering. We should
build one. Two complementary checks:

### A. Op-coverage / loss audit (static)

A test that, for a corpus of inputs, walks the **base IR** and the **CFG** for the
same function and asserts an explicit classification for every base-IR op kind:
`Faithful` | `HardError` | `KnownLossy(reason)`. Any base-IR op that reaches the
CFG lowering and is neither faithfully represented nor a hard-error nor an
*explicitly registered* `KnownLossy` entry is a **fidelity violation** and fails
the test. This turns "we silently drop things" into "we have a reviewed, listed
set of known losses, and nothing new slips in."

Concretely: maintain a `KNOWN_LOSSY` registry (e.g. `["closure-body",
"property-granular-capture", ...]`) with a one-line justification each. The test
asserts the set of actual losses equals the registry. Closing a loss = lowering
it faithfully + removing its registry entry.

### B. Semantic differential check (dynamic) — the strong one

The base IR is *executable* (round-trips to JS) and we already have an SSA
interpreter (`interp.rs`) plus a Node oracle. A closure that mutates a captured
value has an **observable effect**. A faithful CFG, when interpreted, must
reproduce that effect; the current opaque-`MakeArray` CFG cannot (it never runs
the body). So:

> For a battery of programs, run the program (Node / base-IR semantics) and the
> CFG interpreter on the same inputs and assert identical observable results.
> Where they diverge, the CFG has dropped semantically-relevant behavior.

This already exists for straight-line code (`tests/oracle.rs`). **Extend it with
closure-effect programs** (a closure that mutates a captured object, is called,
and the mutation is observed after the call). Today those will *fail* the
differential check — which is exactly the proof that the CFG is wrong, and the
red test we make green by lowering closure bodies. This gives us a precise,
behavioral definition of "the CFG is faithful": *it computes the same thing the
program does.*

> Note: the CFG is an *analysis* view, not used for codegen, so it does not need
> to round-trip to source. The correctness bar is **behavioral equivalence under
> interpretation**, not textual.

---

## The fix: lower closure bodies into nested CFGs

Goal: give the analysis the closure's body so it can infer the body's effects on
its captured context variables — *without* touching codegen (codegen keeps
reprinting the real closure from the base IR, so it stays correct).

### Representation

- Replace the opaque `Op::MakeArray(captures)` for a closure with a dedicated
  `Op::MakeClosure { body: FnId, captures: Vec<(Value, CaptureEffect)> }` (or keep
  `MakeArray` for escape/aliasing but attach a side-table to a nested CFG).
- `Cfg` gains `nested: Vec<Cfg>` (the lowered closure bodies). A closure's body is
  lowered with the same `lower.rs` machinery, recursively. The body's free
  variables that resolve to enclosing interned vars are its **context captures**;
  references to them inside the body become reads/writes of those context values.

### Effects per capture (what we actually need)

Run the **already-ported effect pass** (`infer_effects`) on each nested CFG. For
each context capture, classify the body's effect on it: `Read` / `Mutate` /
`Capture` / `Freeze` (this is exactly upstream `InferReferenceEffects`'
function-expression handling, and our `effects.rs` already has the vocabulary and
the `CreateFunction` / function-value `Apply` shapes that were stubbed for this).

### Wiring into the outer analysis (the part that must land together)

This is where prior shortcuts failed — the three must be coordinated:

1. **At closure creation**, the closure value captures each context value with its
   classified effect (read-captures are frozen reads; mutate-captures alias).
2. **At each closure call** (`Apply` of the closure value), apply the body's
   effects to the captures: a `Mutate` capture is mutated at the call site (this
   is what gives the captured value its reactive scope); a `Read` capture is just
   read.
3. **Escape-aware dependency/scope precision**: a closure that does not escape
   (not returned, not passed to JSX/a hook, not stored) is *not memoized as a
   value* and its read-only captures do **not** become dependencies of the
   absorbing scope. React inlines such closures; we must not let an opaque
   allocation drag its read-captures into a scope (the exact failure mode that
   regressed `capturing-func-mutate-2`: `_c(2)` → `_c(5)`).

### Why all three together

Single-piece changes provably regress (measured twice via the comparator below:
−1 and −3 `agree`). A closure fixture only flips to `agree` when the closure is
correctly absorbed into its mutated capture's scope AND its read captures are
pruned AND the scope's deps are not inflated. Land them as one gated change.

---

## Gated rollout plan

Tooling (built; reuse it — turns the 2-minute Node gate into a ~1s check):
- `examples/react_struct.rs` → `/tmp/react_ref.txt` (React's per-fixture structure,
  captured once; stable).
- `examples/cc_batch.rs` → our per-fixture `cache,blocks` in ~0.7s.
- `oracle/compare.py react_ref ours [prev_ours]` → reproduces the gate buckets
  exactly and prints `REGRESS`/`GAIN` lines vs a prior snapshot.
- `examples/dump_cfg.rs` → prints the CFG for a snippet (the proof tool above).
- `examples/mismatch_probe.rs` → OVER/UNDER/BLOCK deltas per mismatch.

Steps, each gated on `agree` non-decreasing + the differential interpreter check:

0. **Land the conformance checks first** (audit A + differential B). The closure
   differential tests are *expected to fail* now — that red is the spec.
1. **Lower closure bodies into nested CFGs** (analysis-only; codegen untouched).
   Gate: `agree`/`mismatch` unchanged (no consumer yet), differential closure
   tests still red but the body is now present in `nested`.
2. **Infer per-capture effects** on nested CFGs via `infer_effects`. Gate:
   unchanged (still no consumer).
3. **Wire creation + call + escape-aware deps together.** Gate: differential
   closure tests go green; `agree` rises on the `capturing-func*` cluster (~13
   delta-(-1) fixtures); zero `REGRESS` lines from `compare.py`.
4. Iterate to the larger closure mismatches (loops-with-closures, `jsx-outlining`,
   `useEffect` callbacks) once the foundation holds.

Snapshot `/tmp/ours_baseline.txt` **before** each change; the comparator's
`REGRESS` detection is the safety net that caught both prior shortcut regressions.

---

## Audit: other suspected silent-lossy spots to verify

Run the conformance checks against these; promote each to faithful or hard-error:

- **Closure bodies** — confirmed lossy (this doc). Highest impact.
- **Property-granular capture** — `collect_captures` captures `p.a` not `p`
  (first-level only); `p.a.b` captures `p.a`. Verify this matches React's
  dependency granularity or is a registered known loss.
- **`MakeArray`-for-everything** — closures, real arrays, and object-method values
  all become `MakeArray`, so aliasing treats them identically. Verify this does
  not conflate (e.g. a real array vs a closure).
- **JSX → `createElement` call** — JSX is desugared to a call before the CFG; the
  `Render` effect / ref-in-render distinctions React draws on `JsxExpression`
  operands may be lost. Verify against the ref-access fixtures.
- **Object methods** (`{ m() {} }`) — lowered like closures (captures only).
- **Spread** (`...x`) — modeled as a plain capture; verify iterator-mutation
  semantics aren't silently dropped.

Each entry should end up either faithfully lowered or in an explicit
`KNOWN_LOSSY` registry with a justification, so "is the CFG wrong?" has a
checkable, reviewed answer instead of a silent one.
