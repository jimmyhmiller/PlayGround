# Reaching parity with the React Compiler

A plan to take `jsir-ssa` from "memoizes the straight-line subset, matching
React's scope structure" to "matches the React Compiler across its full fixture
corpus." Grounded in what the oracle (`tests/react_oracle.rs` + the fixture
corpus harness) already measures.

## What "parity" means here

Parity = for every fixture the React Compiler compiles, we produce output that is
**structurally equivalent** (same `_c(N)` cache size, same memo scopes, same
dependency sets) and **behaviorally equivalent** (same render output, same
reference stability under unchanged deps). We do not need byte-identical text
(we may emit `createElement` where React keeps JSX), though the recommended
codegen below would get us textual parity for free.

We already have the measuring instrument: the cloned `react-compiler-e2e` CLI as
an oracle. Parity work is "drive the corpus agreement rate to ~100% and the
coverage (fixtures we attempt) to ~100%," with the oracle as the gate.

## Current baseline (measured)

Over React's `__tests__/fixtures/compiler` (.js fixtures):

| bucket | count | meaning |
|---|---:|---|
| comparable (both memoize) | ~46 | our compilable subset |
| → **agree with React** | ~21 (**46%**) | structure matches |
| react-only | ~300 | React compiles, we bail (control flow / hooks) |
| ours-only | ~87 | we memoize, React skips (often non-components) |
| neither | ~690 | error fixtures / non-components / TS-only |
| panics in ours | ~70 | real lowering bugs |

So two axes to close: **coverage** (the ~300 react-only + ~70 panics we can't
attempt) and **fidelity** (the ~25 comparable mismatches).

Perf context: on the straight-line subset we are ~20x faster wall-clock, but
that is mostly because we do less (no type inference, no validation, string
codegen) and bail on the hard 97%. Reaching parity will and should narrow that
gap; the goal is correctness-at-coverage, not speed.

## Hard requirement: memoization codegen MUST be an IR→IR transform

This is not a design choice to weigh. **Codegen must be a transform on the
reversible JSIR, not a CFG→JS "relooper."** Any plan that emits JavaScript from
the flat CFG is rejected up front.

Why this is mandatory, not optional:

- The current string-based codegen (`codegen.rs`) **cannot reach parity** by
  construction. It bails on any function with control flow
  (`cfg.blocks.len() != 1`), and the only way to make it handle control flow is
  to solve structured-control-flow recovery (the relooper problem) — a large,
  bug-prone effort that re-implements something we already have working.
- We **already** have a reversible IR (`jshir`/`jsir`) that round-trips *all*
  control flow to source byte-exactly (`hir2ast` → `ast2source`), validated
  against test262. Emitting through it is strictly less work and strictly more
  correct than any CFG→JS path. Choosing the CFG→JS path would be throwing that
  away and rebuilding it worse.

The required pipeline:

1. Build the CFG + SSA **view** from JSIR (as now) and run the analyses (mutable
   ranges, reactive scopes, deps). The analyses already handle multi-block CFGs.
2. Map each reactive scope back to JSIR statement positions (we preserve spans /
   trivia on every op, so the mapping exists).
3. Emit the memoized program by **rewriting the JSIR**: insert the `_c(N)` cache
   declaration, wrap each scope's statements in the cache-check `if`, hoist
   outputs to `let`. This is an IR-to-IR transform like our DCE pass.
4. Print with the existing `hir2ast` → `ast2source`.

This is required because it is the only path that gives all three of:
- **Control-flow codegen for free** — jshir already emits `if`/`while`/`for`/
  `try`/`switch`/early-return correctly; we only insert memo blocks around
  statement ranges. There is no relooper to write.
- **JSX-preserving output** — stop desugaring JSX and keep JSX through the
  reversible IR; the printer re-emits the JSX literal, matching React textually.
- **Reuse of a test262-verified printer** instead of the ad-hoc string builder.

The string codegen in `codegen.rs` is a temporary stopgap for the single-block
case and the existing tests only. It is **not** part of the parity path and is
to be retired once the IR→IR transform lands; do not extend it to handle control
flow.

## Workstreams, ordered by fixtures unlocked

### 1. Control-flow codegen (biggest unlock: ~300 fixtures)
Build the **required IR→IR transform** codegen described above — this is the
mechanism, not "a way" to do control flow. Reactive scopes that span branches
need React's scope-alignment rules (a scope that starts inside one branch and is
used after the join must be hoisted to the join point — see fixtures
`align-scope-starts-within-cond`, `align-scopes-iife-*`). The analysis already
produces the scope ranges; the work is the IR rewrite + alignment.

### 2. Hooks (~hundreds of fixtures)
Hooks have special semantics the analysis must encode:
- hook calls are ordered and cannot be conditional (also a validation);
- a hook's result is a **reactive root** (like props/state);
- some hook results are **stable** and never become deps (`useRef().current`,
  the `useState` setter, `useCallback`/`useMemo` results are already memo);
- a `use`-prefixed function is itself compilable as a hook.
Needs a small hook-signature model (React keys these off naming + a registry).

### 3. Type inference — `InferTypes` (precision for many fixtures)
A unification pass (React's `react_compiler_typeinference`) that assigns each
value a type: primitive / object / array / function / hook / builtin. Drives:
- which values are references (today we over-approximate `is_ref`);
- which calls are pure vs mutating (today only `createElement`/`jsx` are pure);
- constant-vs-reactive classification.
Without it we both over-scope (treat primitives as references) and under-scope.

### 4. Mutation / aliasing precision — `InferReferenceEffects` + signatures
Our mutable-range analysis is deliberately coarse ("any call may mutate its
reference args"). React has a detailed **effect system** plus a registry of
known function signatures (`Array.prototype.map` is pure-returning-fresh,
`Array.prototype.push` mutates the receiver, frozen values can't be mutated,
etc.) and **freeze** semantics (a value passed to JSX/hooks becomes frozen).
Fixtures `transitive-freeze-array`, `type-inference-array-from`,
`repro-mutate-result-of-method-call-on-frozen-value` show where we diverge.
This is the largest *analysis* effort and the main source of the remaining
comparable mismatches.

### 5. Lowering breadth (closes the ~70 panics + ours-only gaps)
Add to `lower.rs` / the converter: `new X()` (NewExpression), optional chaining
(`a?.b`, `a?.()`), spread (`{...x}`, `[...a]`, `f(...a)`), template literals,
destructuring binds (`const {a,b} = props`), `for-of`/`for-in`, labeled
statements, `switch`, sequence expressions. Each is a few lowering cases; the
panics list (`ReadVar has result`, etc.) pinpoints the exact missing forms.

### 6. Component / hook detection + scope pruning
React only compiles functions it proves are components/hooks and **prunes
scopes for non-reactive values** (`PruneNonReactiveScopes`) and for values that
escape into refs/effects. Our `ours-only` bucket (~87) is mostly us memoizing
things React decides not to. Needs the detection + the prune passes.

### 7. Constant propagation / folding
React folds constants (`constant-propagate-global-phis`) which removes
dependencies and can drop whole scopes. We already have a `ConstProp` analysis
in `jsir-analyses` — wire it in before scope inference.

### 8. JSX member-expression tags, fragments, entities, whitespace edge cases
Smaller correctness items the corpus flagged (`jsx-member-expression`,
`jsx-fragment`, `jsx-html-entity`, `jsx-string-attribute-non-ascii`). Mostly in
the JSX desugar / (future) JSX-in-JSIR handling.

## Phased roadmap

Each phase is gated by the oracle: it ships when the corpus agreement rate for
the targeted bucket reaches ~100% and no regression elsewhere.

- **Phase A — lowering breadth + detection/pruning.** Kills the ~70 panics, adds
  NewExpression/spread/optional-chaining/destructuring, adds component detection
  + scope pruning. Closes most of the `ours-only` gap and the comparable
  mismatches that are pure lowering. *Smallest effort, immediate corpus lift.*
- **Phase B — the required IR→IR codegen + control flow.** Replaces the string
  codegen with the JSIR-transform path (mandatory, per the hard requirement
  above). Unlocks the ~300 react-only control-flow fixtures and gives
  JSX-preserving output. *Highest single unlock.*
- **Phase C — type inference + effect/signature system.** Brings mutation
  precision to parity; closes the remaining comparable mismatches (freeze,
  method purity). *Largest analysis effort.*
- **Phase D — hooks.** Hook model + validation; unlocks the hook fixtures.
- **Phase E — long tail.** Constant folding wired in, JSX edge cases, remaining
  fixtures one by one against the oracle.

## Validation strategy

- **Structural oracle** (`react_oracle.rs`): per-fixture compare `_c(N)` + memo
  block count + (extend to) per-block dependency sets, normalized to canonical
  ids (prop accesses by field, scope refs by producing scope). Promote the
  fixture corpus from an example to a gated test asserting "agreement ≥ baseline,
  monotonically non-decreasing."
- **Behavioral oracle** (already have for JSX): run our memoized output under
  Node against the createElement-desugared baseline; assert same render tree for
  many prop sets **and** reference stability when deps are unchanged. Extend to
  run React's output too (via a JSX runtime) for a three-way behavioral check.
- **No-regression on the base project**: the JSIR converter must stay byte-exact
  on test262 (98.9% today) — the React work lives in `jsir-ssa` and a JSX desugar
  that only touches JSX nodes, so this is already protected; keep it that way.

## Non-goals / risks

- We do **not** aim for byte-identical text initially; structural + behavioral
  parity first, textual (JSX-preserving) parity falls out of Phase B.
- React's mutation/effect system is the deepest pass and the main risk to a clean
  parity claim; budget the most time there (Phase C).
- Keep the reversible JSIR pure. All memoization is a transform layer; never
  bake React semantics into the base IR (it must keep serving DCE, tree-shaking,
  and test262 parity).
