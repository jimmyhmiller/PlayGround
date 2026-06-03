# Porting React's two-phase mutation/aliasing model

## STATUS — LANDED (default), agree 224 -> 244

The two-phase model is now the **default and only** mutable-range analysis. The
single-pass union-find (`mutability.rs::analyze` + `Uf`) is **deleted**;
`mutability.rs` now holds only the shared `Ranges` type + `render`. The ranges
are produced by `aliasing_ranges::analyze` (Phase 1 effects via
`infer_effects::infer` -> Phase 2 abstract heap graph via `compute_ranges`).

What shipped:
- **`aliasing_ranges.rs`** — faithful Phase-2 port: `AliasingState` heap graph
  (`Capture`/`Alias`/`MaybeAlias` edges, `created_from`), the `mutate()` worklist
  (forward Capture+Alias, backward Alias+MaybeAlias, MaybeAlias⇒Conditional,
  transitive⇒captures backward, temporal edge-index gating), and the `compute_ranges`
  driver. Unit-tested in isolation.
- **`memoize_plan`** now derives per-instruction mutations from the same effect
  stream (`aliasing_ranges::mutations_by_instr`) — single source of truth,
  freeze-aware (hooks freeze, don't mutate, their args).
- **Real bug fixes found en route:**
  - `Array.push` had `callee_effect: Read` (defaulted) so `a.push(x)` was treated
    as non-mutating — **unsound**. Fixed to `Store` (faithful to upstream
    `globals.rs`). This was masked before because the union-find mutated every
    non-pure call argument by over-approximation.
  - **Method-call emission detached the receiver:** the memoizer emitted
    `const m = recv.method; m(args)` (loses `this`), so every memoized fixture with
    a method call (`.map`, `.push`, …) silently miscompiled at runtime. The corpus
    structure-only metric never caught it. Fixed: `Op::Call` with a `Member` callee
    now emits `recv.method(args)`.
  - **Control-dependence soundness bail:** a branch whose condition reads a mutated
    allocation makes the branch-selected values control-dependent on untracked
    mutable state; memoizing them once would be stale. The precise model no longer
    conflates them (the union-find did, accidentally), so `check_soundness` bails
    explicitly.

Standing: **agree 244 / 715 (34.1%), panic 0, all 10 test binaries green**
(37 gains, 17 regressions vs the old 224). Verified sound via the Node differential.

Remaining 17 regressions are three known clusters needing follow-up passes (NOT
analysis bugs in the port):
- **manual memoization** (`useMemo-simple`, `use-callback-simple`,
  `drop-methodcall-usecallback`, `useCallback-set-ref-*`): React eliminates
  `useMemo`/`useCallback` and inlines; we keep the call, so closure-arg + deps-arg
  + result should fuse into one scope. Needs `DropManualMemoization` or a
  hook-arg-array scope-exclusion.
- **closures** (`capturing-function-*`, `outlined-*`): need `CreateFunction` +
  applying the lowered body's inferred signature (Step 1 of the plan below).
- **scope-grouping convention** (`object-expression-computed-key-*`): the inclusive
  vs half-open mutable-interval membership in `scopes::infer` is a crude
  approximation of `InferReactiveScopeVariables`; the faithful overlap-based
  grouping pass would resolve it.

The original plan (Steps 0-4) and rationale follow.

---

## Why this document exists

Our reactive-scope analysis is stuck at **agree 224 / 715 (31.3%)** against the
official `babel-plugin-react-compiler` oracle. The remaining close-mismatch
clusters (`globals-*`, `capturing-function-alias-*`, `array-map-frozen-*`,
`call-with-independently-memoizable-arg`, …) all fail for **one** reason: our
mutable-range analysis (`src/mutability.rs`) is a **single-pass union-find** that
co-scopes a captured value with its container *unconditionally*, and it has no
notion of **freeze**, **conditional** vs **definite** mutation, or
**transitive** capture.

This was proven by measurement, not opinion. Attempting to patch the union-find
toward React's behavior on the `globals-Boolean` fixture (`const x = {}; const y
= Boolean(x); return [x, y]`) required, in sequence:

1. exclude primitive-coercion globals from memoizable allocations,
2. mark them read-only so they don't create a spurious mutable range,
3. gate `MakeArray`/`MakeObject` capture-aliasing on whether the container is
   actually mutated (the frozen-capture fix),
4. and *then* stop `MergeConsecutiveScopes` from folding two empty-dep scopes
   that are actually in a dependency relationship.

Each layer exposed the next, and the partial result **regressed the oracle
224 → 220 (−4 agree, +15 bails)**. Every surgical patch on the single-pass
structure either uncovers another layer or breaks the baseline. The conclusion is
architectural: we need React's **two-phase model**, ported faithfully, replacing
`mutability.rs` wholesale.

> Ground rule (from project memory, learned at ~12M tokens of cost): **port from
> the Rust source, do not reverse-engineer from fixtures.** Crude
> fixture-reverse-engineering has regressed the gate every time; faithful ports
> have been clean. The source is at
> `~/Documents/Code/open-source/react-rust-pr36173/compiler/crates/` (facebook/react
> PR #36173). The authoritative gate is `oracle/run-corpus.sh` (the official babel
> plugin, universe 715) — **NOT** the `react-compiler-e2e` Rust binary, which is an
> incomplete reimplementation that bails on ~200 fixtures and gives misleading
> numbers.

---

## The upstream architecture (what we are porting)

React's pipeline order (from `react_compiler/src/entrypoint/pipeline.rs`, the
canonical sequence):

```
analyse_functions                       // recursively infer inner-function (closure) signatures
infer_mutation_aliasing_effects         // PHASE 1: per-instruction aliasing effects
infer_mutation_aliasing_ranges          // PHASE 2: abstract heap graph -> MutableRange per identifier
infer_reactive_scope_variables          // group identifiers with overlapping ranges into scopes
align_reactive_scopes_to_block_scopes   // align scopes to block boundaries (can't memoize half a loop)
merge_overlapping_reactive_scopes       // merge scopes whose ranges overlap
propagate_scope_dependencies            // compute deps by reactive access path
prune_non_escaping_scopes               // escape analysis
```

The two phases that replace `mutability.rs` are the first two analysis passes:

### Phase 1 — `infer_mutation_aliasing_effects` (3016 lines upstream)

For each instruction, emit a list of **aliasing effects** describing how values
flow and mutate. The vocabulary (faithful port already lives in our
`src/effects.rs::AliasingEffect`):

| effect | meaning |
|---|---|
| `Create { into, value: ValueKind, reason }` | a fresh value of a given kind (Primitive/Object/Frozen/MaybeFrozen/…) |
| `CreateFrom { from, into }` | new value with the same kind as `from` |
| `CreateFunction { into, captures, function_id }` | a closure capturing values (the path we previously stubbed) |
| `Freeze { value, reason }` | mark `value` and direct aliases frozen (flows into JSX/return/hook) |
| `Capture { from, into }` | information flow `from` → `into` (e.g. `x` into `[x]`) — non-aliasing |
| `Alias { from, into }` | mutation of `into` implies mutation of `from` |
| `MaybeAlias { from, into }` | potential aliasing (downgrades mutations to conditional) |
| `Assign { from, into }` | direct `into = from` |
| `ImmutableCapture { from, into }` | data flow for escape analysis only, no mutable-range influence |
| `Mutate { value, reason }` | mutate value + direct aliases |
| `MutateConditionally { value }` | mutate only if mutable |
| `MutateTransitive { value }` / `MutateTransitiveConditionally` | mutate value + transitive captures |
| `Apply { ... }` | a call: applies a callee's **signature** effects to the args |

The crucial distinctions our union-find lacks: **Capture ≠ Alias** (capturing
`x` into `[x]` does NOT mean mutating the array mutates `x` — it's information
flow, not aliasing), **Freeze** (a value that flows to a return/JSX can no longer
be mutated, so it stops extending ranges), and **conditional/transitive**
mutation.

### Phase 2 — `infer_mutation_aliasing_ranges` (1168 lines upstream)

Builds an **abstract heap graph** and interprets the Phase-1 effects over it to
compute each identifier's `MutableRange { start, end }`. The data model
(`infer_mutation_aliasing_ranges.rs:44-110`):

```rust
enum EdgeKind { Capture, Alias, MaybeAlias }
struct Edge { index: usize, node: IdentifierId, kind: EdgeKind }
enum NodeValue { Object, Phi, Function { function_id } }
struct Node { id, edges: Vec<Edge>, value: NodeValue, /* mutation info */ }
struct AliasingState { nodes: HashMap<IdentifierId, Node> }
```

Effects build edges (`AliasingState::capture`/`assign`/`maybe_alias`/`create_from`,
lines 118-182). The heart is **`mutate()`** (line 221), a worklist that
propagates a mutation through the graph:

- **Forward** through `Capture` and `Alias` edges: `mutate(a)` where `a Capture/Alias b` ⇒ `mutate(b)`.
- **Backward** through `Alias` and `MaybeAlias` edges (MaybeAlias **downgrades to
  conditional**): `mutate(b)` where `a Alias b` ⇒ `mutate(a)`.
- Mutation has a `MutationKind` (`None`/`Conditional`/`Definite`) that only
  *raises* a node's level; the propagation extends each touched identifier's
  `mutable_range.end` to the mutation's evaluation order
  (lines 276-277, 691).

The pass walks instructions in evaluation order, applies each instruction's
effects to the graph, runs `mutate()` for each `Mutate*` effect, and finally
emits the per-identifier ranges plus the legacy `Effect` on each `Place`
(lines 446-900).

**This is what makes `globals-Boolean` work:** `[x, y]` *Captures* x (not
Aliases), x is never mutated, so the `mutate()` worklist never reaches x — x's
range stays a point, and x ends up in its own scope, separate from the array.
The union-find cannot express "capture-but-not-alias," which is the entire bug.

---

## What we already have vs. what we need

| Upstream pass | Our status | File |
|---|---|---|
| `analyse_functions` | partial: closure bodies lowered to nested CFGs; mutation derived | `lower.rs` (`Cfg::nested`/`closures`), `mutability.rs` closure pre-pass |
| `infer_mutation_aliasing_effects` | **partial port** — `AliasingEffect` vocab, `AbstractValue`, `merge_value_kinds`, `apply_effect`, per-op `compute_signature`, worklist driver. Wired only into validation bail-checks today, NOT into ranges. | `src/effects.rs`, `src/infer_effects.rs` |
| `infer_mutation_aliasing_ranges` | **MISSING** — this is the port. Our `mutability.rs` union-find is the thing it replaces. | (new) `src/aliasing_ranges.rs` |
| `infer_reactive_scope_variables` | custom version | `src/scopes.rs::infer` |
| `align/merge/propagate/prune` | custom blend | `src/scopes.rs` |
| type/shape info (ValueKind, freeze, ref/state shapes) | partial port | `src/types.rs`, `src/infer_types.rs` |

So the work is: **(A) finish Phase 1** so it emits faithful per-instruction
effects (including the `CreateFunction`/`Apply` closure path), then **(B) port
Phase 2** as a new module that consumes Phase-1 effects and produces
`MutableRange` per `Value`, then **(C) swap `mutability.rs`'s output for it** and
re-tune the downstream scope passes to consume the new ranges.

---

## Impedance: our CFG vs. upstream HIR

Port faithfully but translate the substrate honestly. Differences and the rule
for each:

| upstream HIR | our CFG | translation |
|---|---|---|
| `Place` / `IdentifierId` | `cfg::Value` (SSA) | a Node per `Value`; ranges keyed by `Value` |
| `MutableRange { start, end: EvaluationOrder }` | program points (`mutability::Point`, RPO instruction index) | reuse our linear point numbering as `EvaluationOrder` |
| `FunctionExpression { lowered_func }` / `ObjectMethod` | `MakeArray` of captures + `Cfg::closures[v].body` (nested CFG) | `CreateFunction { into: v, captures, body }`; recurse `analyse_functions` into `cfg.nested` (already done for mutation; extend to full signature) |
| `JsxExpression` | desugared to `createElement(tag, props, ...children)` `Call` | the call's signature **Freezes** its operands (JSX freeze) — we already special-case `createElement`/`jsx*` as pure; in the new model give them a `Freeze`+`Create frozen` signature |
| `MethodCall` / `PropertyCall` | `Member` load + `Call` | resolve the member to get the receiver; apply the call signature to receiver+args |
| global function signatures (`Boolean`/`Array.isArray`/`Object.keys`/…) | — | the `ShapeRegistry`/`GlobalRegistry` in `types.rs` — grows fixture-by-fixture from upstream `globals.rs` (2500 lines); each entry gives arg effects (readonly vs mutates) + return `ValueKind` |
| `Destructure` | `bind_target` member-loads | each binding is `CreateFrom`/`Capture` from the source |

The closure path is the one we previously stubbed and that the clusters need:
`CreateFunction` records the captured values + the nested body, and `Apply` of a
closure value applies the **inferred signature of the body** (which captures it
mutates / freezes / returns) to the call site. We already lower the body and
derive *mutation*; the new model also needs *capture/alias/freeze* per captured
value, which Phase 1 over the nested CFG produces.

---

## The port plan — incremental, every step gated

**Gate protocol (mandatory, every step):**
1. `cargo build -q -p jsir-ssa && cargo test -q -p jsir-ssa` — all 10 binaries green.
2. Snapshot the agree set **before** the change:
   `./oracle/run-corpus.sh --list agree | sort > /tmp/agree_before.txt`.
3. Make the change; re-run; `comm -23 /tmp/agree_before.txt /tmp/agree_after.txt`
   must be **empty** (zero REGRESS). `comm -13` shows GAINS.
4. **Revert immediately on any net regression** (this is what kept 204 → 224
   clean). Do not leave a net-negative change in "as a foundation."
5. The fast `react-compiler-e2e` corpus is a *cheap pre-filter only* — it bails on
   the target fixtures, so the official oracle is the real arbiter for this work.

### Step 0 — scaffolding, behavior unchanged
Add `src/aliasing_ranges.rs` with the data model (`EdgeKind`, `Edge`, `NodeValue`,
`Node`, `AliasingState`, `MutationKind`, `mutate()`), ported from
`infer_mutation_aliasing_ranges.rs:44-440`, keyed by `cfg::Value`. Add a
`compute_ranges(cfg, &EffectResults) -> Ranges` entry that is **not yet called**.
Unit-test the graph in isolation (capture-not-alias, alias-both-ways,
maybe-alias-conditional, transitive). Gate: corpus byte-identical (unused code).

### Step 1 — finish Phase 1 effects to feed the graph
`infer_effects.rs::compute_signature` already maps most ops. Make it emit the
**full** effect set Phase 2 needs, faithful to
`infer_mutation_aliasing_effects.rs`:
- `MakeArray`/`MakeObject` → `Create { Object }` + `Capture { elem → container }`
  (NOT alias).
- `Member` load → `CreateFrom`/`Capture` from base (path-aware).
- `StoreMember` → `Mutate { obj }` + `Capture { value → obj }`.
- `Call` → `Apply` of the callee signature; unknown callee = conservative
  (`MutateTransitiveConditionally` args); known globals from the registry;
  JSX/createElement = `Freeze` operands + `Create frozen`.
- closure `MakeArray` → `CreateFunction { captures, body }`; closure `Apply`
  applies the body's inferred signature.
Gate: still unused by ranges; corpus unchanged. Validate Phase-1 output on a few
fixtures by dumping effects (add a `DUMP_EFFECTS` probe).

### Step 2 — compute ranges from the graph, behind a flag
Wire `compute_ranges` to produce a `mutability::Ranges`-shaped result (same public
type: `range`, `is_ref`, `def`, `alias_root`, `term_point`) so it is a drop-in.
Put it behind `JSIR_ALIASING_RANGES=1`. Compare its `Ranges` to the union-find's
on the corpus (a diff harness): where they differ, those are the fixtures the new
model changes. Gate: default path (union-find) unchanged.

### Step 3 — flip the default, measure, iterate
Make `scopes.rs`/`plan()` consume the new ranges by default. Run the full oracle.
Expect movement on `globals-*` / frozen-capture / capturing-alias clusters, and
expect some downstream scope passes (`MergeConsecutiveScopes`, escape) to need
re-tuning because the ranges now differ. Iterate **one downstream pass at a time**,
gated:
- `MergeConsecutiveScopes`: with correct ranges, two values in a capture (not
  alias) relationship no longer share a range, so the spurious merge that broke
  `globals-Boolean` should not occur — verify, then align the merge rule to
  `merge_overlapping_reactive_scopes_hir.rs`.
- escape / deps: now that `Freeze` and `Capture` are explicit, port
  `prune_non_escaping_scopes` and `propagate_scope_dependencies_hir` semantics
  where ours diverge (the deps are already path-keyed from this session).

### Step 4 — retire `mutability.rs`
Once the new ranges are the default and the oracle is ≥ the old baseline with no
REGRESS, delete the union-find `mutability.rs` (or keep only as a cross-check in
tests). Update `MEMORY.md` standing.

---

## Faithful-port checklist (per pass)

For each upstream pass, port the *algorithm*, not a fixture-shaped approximation:

- [ ] `mutate()` propagation exactly: forward Capture+Alias, backward Alias+MaybeAlias, MaybeAlias⇒Conditional, level only raises, range.end extends to mutation order.
- [ ] `Freeze` stops range extension and is transitive over direct aliases.
- [ ] `Capture` is information-flow only (escape/deps), never extends the source's mutable range.
- [ ] closure (`CreateFunction`) captures get the body's inferred per-capture effect (read/mutate/freeze), via `analyse_functions` over `cfg.nested`.
- [ ] global signatures come from the `types.rs` registry (grow it from upstream `globals.rs`), never hardcoded per fixture.
- [ ] conditional vs definite mutation preserved (needed for `MaybeAlias` and `?.` paths).

## Risk and exit criteria

This is the documented high-risk pass — it replaces the core the 224 baseline
rests on. The non-negotiables:

- **Never regress the agree set.** The agree-set diff is the net; revert on any
  REGRESS line. A net-positive-with-some-bucket-shuffle is fine; a lost *agree*
  is not.
- **Land Phase 1 and Phase 2 before flipping the default.** A half-flipped state
  (new ranges, old downstream passes) is expected to wobble; keep it behind the
  flag until the whole chain is consistent.
- **Done when:** the new model is the default, oracle agree ≥ 224 with zero
  REGRESS and a measurable gain on the frozen-capture clusters, `mutability.rs`'s
  union-find is retired, and the differential fuzzer (`tests/differential.rs`) is
  still green (behavioral correctness is independent of this analysis and must not
  move).

## Source map (port from these exact files)

```
react_compiler_inference/src/
  infer_mutation_aliasing_effects.rs   (3016)  -> finish src/effects.rs + src/infer_effects.rs
  infer_mutation_aliasing_ranges.rs    (1168)  -> new src/aliasing_ranges.rs (replaces mutability.rs)
  analyse_functions.rs                  (218)  -> closure signature inference over cfg.nested
  infer_reactive_scope_variables.rs     (399)  -> src/scopes.rs::infer (align)
  merge_overlapping_reactive_scopes_hir.rs (419) -> src/scopes.rs merge fold (align)
  align_reactive_scopes_to_block_scopes_hir.rs (327) -> scope/block alignment
  propagate_scope_dependencies_hir.rs  (2293)  -> src/scopes.rs deps (path-keyed; partially done)
react_compiler_reactive_scopes/src/
  prune_non_escaping_scopes...                 -> src/scopes.rs escape (align)
react_compiler_hir/src/
  lib.rs (AliasingEffect @1325), type_config.rs (ValueKind/ValueReason), globals.rs (signatures)
```
