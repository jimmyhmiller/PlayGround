# Handoff: fixing reactive-scope analysis (the big parity lever)

## Current entry point

This file is historical and contains some stale statements. Read
`REACT_PARITY_STATUS.md` first for the current measured state and rules. Use this
file for background and fixture examples only.

## TL;DR

We are at **agree 234 / 715 (32.7%)** against the official babel-plugin oracle.
The single biggest lever left is **not the emitter** — it is the **reactive-scope
analysis**. ~281 of the ~480 remaining misses come from our scope analysis
**diverging from React's**: wrong grouping (over-merge / over-split), missing
dependencies (it doesn't track control-flow deps), and over-aggressive escape
pruning. Our scope passes are a hand-rolled "custom blend," not faithful ports.
**Port React's reactive-scope pipeline faithfully and these resolve in bulk.**

This project's own history is emphatic about the method: **faithful ports of the
Rust source move hundreds and land clean; fixture-reverse-engineering moves ones
and regresses the gate.** Do the port.

---

## 2026-06-03 session update (read this before re-bucketing)

**Worktree resolved.** The old `js-ir-parity` worktree/branch was an analysis-first
exploration that stalled at agree 31; it is DELETED (tip was `e07c6f64a`, reflog-
recoverable). Nothing needed salvaging — its docs were already byte-identical here
and its code far behind. The lead work (this branch, `partial-recursion-fuzzer`,
main checkout) is canonical. There is no js-ir worktree now.

**Selection fix landed (foundational, gate-neutral, kept).** `lower_function` used
to lower the *first* function declaration; React (`compilationMode: 'all'`) only
memoizes the *component/hook*. On fixtures with a helper before the component
(e.g. `constructor.js` = empty `Foo` then `Component`) we were analyzing the wrong
function — empty CFG → spurious "nothing escapes". New shared selector
`lower::select_function` (and `selected_function_node_id`) picks the first
top-level function whose body returns JSX or calls a hook, keyed by stable
`node_id` so the analysis (`lower.rs`) and the in-place emitter (`memoize_plan.rs`
`function_body_stmts`/`_mut`) target the SAME function. Affects ~11 fixtures, all
of which now show their REAL bail instead of an empty-CFG mask. agree unchanged
(234, zero regress) because those 11 hit deeper issues — but it is a PREREQUISITE:
none of them can ever agree while we analyze the wrong function.

**Corrected diagnosis: the 31 "nothing escapes" bucket is NOT uniformly escape-
over-prune.** `why_miss` labels by bail *message*; an empty CFG (wrong/empty
function, or a lowering gap) trivially "nothing escapes" too. `constructor.js`
was a selection bug, not escape. Re-derive buckets from probes, not the message.

**Infer + MergeOverlapping FAITHFULLY PORTED + multi-output emitter — LIVE, net
positive (agree 234 → 235, react_only 348 → 287, −61 bails). Backup in
`INFER_PORT_WIP.patch`.** `scopes.rs::infer` is now a faithful
`find_disjoint_mutable_values` union-find + `merge_overlapping_scopes`
(`MergeOverlappingReactiveScopes`: merge crossing/identical ranges, PRESERVE
nesting). Four CFG-vs-HIR impedance bugs found and fixed (each gated, each moved
the number):
  1. **Point off-by-one.** React keys "operand is not an input" off `range.start
     > 0` (upstream instr ids start at 1, only params at 0). OUR points start at
     0 for the first instruction, so `start > 0` wrongly excluded a value defined
     at point 0 (leading `const foo = …`). Fixed: test param/block-arg membership
     directly. (+24: 211 → 235.)
  2. **Method receiver buried.** `y.push(z)` lowers to `Call(Member(y,push),[z])`;
     React's HIR MethodCall lists receiver y directly. Fixed: resolve a Call's
     callee member chain to its root and include it (`member_root`).
  3. **No-result instructions skipped.** `store x.y <- v` (a mutation) has no
     result, so operand grouping was skipped. Fixed: walk RPO with a per-instr
     point counter matching `aliasing_ranges`, process operands for EVERY instr.
  4. **Multi-output emitter bail removed** (`memoize_plan.rs`). The analysis now
     produces correct multi-output scopes (React caches several values per guard,
     `$[3]=a;$[4]=b`); `emit_guard` already loops over `plan.outs`, so all-NAMED
     multi-output now emits. Anon-in-multi still bails (needs N-temp synthesis).

**Where the remaining 22 regressions / 193 mismatches live = the DOWNSTREAM
passes, not grouping.** Probed: `capturing-function-alias-computed-load` etc. have
CORRECT grouping but wrong OUTPUTS and DEPS — e.g. a scope whose value escapes only
through a closure alias isn't marked as an output; a consumer scope's dep on
`x[0]` is missing. The remaining regression buckets:
  - **5 "output has no source statement"** + **6 "scope statements not contiguous"**
    = emitter can't place the new (larger/aliased) scopes; needs anon-multi-output
    + non-contiguous handling.
  - **computed-load / dont-merge / capturing-* (mismatch)** = `PruneNonEscapingScopes`
    (faithful escape→outputs) + `PropagateScopeDependencies` (faithful deps incl.
    access paths through aliases). These two passes (1258 + 2293 lines upstream)
    are the next major work and will flip most of the 193 mismatches → agree AND
    recover the regressions.
  - **reactive-scope-grouping** (`z` split out) = `aliasing_ranges` doesn't extend
    a captured value's range through `y.push(z)`; verify `InferMutableRanges`
    capture-range propagation.

**NEXT (in order):** (a) faithful `PruneNonEscapingScopes` → replace the
`prune_non_escaping` + `outputs` filter in `analyze`; (b) faithful
`PropagateScopeDependencies` → replace the operand-derived `deps`; (c) emitter
anon-multi-output + non-contiguous; (d) aliasing capture-range extension. The
grouping foundation is now correct, so these convert mismatch→agree.

**Lesson re-confirmed: surgical over-merge patches regress.** Tried the
`globals-Boolean/Number/String` cluster (`react=(2,2) ours=(1,1)`): React makes
two separate constant scopes (`{}` and `[x,y]`) kept apart by the intervening
`y = Boolean(x)` call; we over-merge to one. The surgical fix — (A) exclude
primitive-coercion globals from `allocations`, (B) add React's merge-adjacency
rule (a bare non-transparent instruction between two scopes breaks the run) —
made it WORSE (cache 1→0): removing Boolean-as-allocation collapsed the *array*
scope's memoization through the escape graph. Reverted. The real fix is the
faithful `PruneNonEscapingScopes` + `InferReactiveScopeVariables` ports, where
escape/output and scope grouping are computed together — not a patch to one knob.
(The merge-adjacency rule itself is correct and faithful; it just can't land in
isolation without the escape side being faithful too.)

---

## Do NOT touch the emitter

The codegen is done and is the *sole* path: `codegen::compile` calls only
`memoize_plan::memoize_inplace`, which edits the original JSIR tree in place
(control flow kept verbatim, no relooper) and faithfully emits **whatever the
analysis decides**. It handles named + anonymous outputs, multi-statement scopes,
contiguous runs, and variable/phi name recovery. `build_layout` / `memoize_file`
/ `recover_regions` are dead code (delete once analysis parity is reached).

The emitter is a faithful renderer of the analysis. Every wrong cache structure
we emit is the analysis handing it the wrong scopes/deps. So fix the analysis.

(One emitter limitation interacts with this: `memoize_inplace` currently **bails
on multi-output scopes** — see `InplaceScope` single-output restriction. That is
fine for now: the multi-output bails are mostly *analysis* problems anyway, and
the emitter generalization regressed when tried because the upstream structure
was wrong. Fix the analysis first; revisit multi-output emission after.)

---

## The diagnosis, with evidence

Run `examples/inplace_probe.rs <fixture>` (dumps our scopes: deps/outputs) and
`oracle/run-corpus.sh --show <fixture>` (React's structure). The 337 `react_only`
bails break down (via `examples/why_miss.rs`):

| count | bail reason | root |
|---|---|---|
| **106** | multi-output scope (emitter bails) | scope **grouping + deps** divergence |
| 45 | reactive scope spans multiple basic blocks | scope **block-alignment** (control flow) |
| 31 | nothing escapes | **escape** analysis over-prunes |
| ~66 | lower: unsupported for-of/for/try/switch/do-while/fn-decl | **lowering** gaps (separate) |
| 11 | allocation mutated outside guard | soundness / range edge |
| 7 | scope statements not contiguous | analysis range edge |
| ~5 each | branch-condition-reads-mutated, etc. | misc |

And `mismatch = 144` (we memoize, wrong structure) is dominated by **under-count**
(~146 under vs ~76 over historically) — i.e. we **miss deps / scopes**.

So the dominant theme across **106 (multi-output) + 144 (mismatch) + 31
(nothing-escapes) ≈ 281 fixtures** is scope grouping / dependency / escape
precision. Concrete cases (all verified this session):

1. **Missing control-flow deps.** `interdependent.js`: our grouping *matches*
   React (one multi-output scope holding `a` and `b`), but React keys it on
   `props.a, props.b, props.c` while we only find `props.a, props.b`. The
   `props.c` is read inside an `if (props.c)` *inside* the scope. Our
   `PropagateScopeDependencies` does not collect dependencies read in control
   flow. React's does. → cache 7 vs 8.

2. **Over-merge (should be nested scopes).** `allocating-primitive-as-dep-nested-scope.js`:
   React emits 2 **nested** scopes (`_c(5)`, an inner `if ($[3] !== t1)` inside an
   outer `if ($[0] !== props.a || ...)`); we fuse them into one scope with two
   outputs.

3. **Over-split (should be one scope).** `alias-capture-in-method-receiver.js`:
   React emits **1** scope (`_c(1)`, sentinel, no deps); we produce two scopes.

4. **Over-pruned escape.** The 31 "nothing escapes" (`call.js` → React `_c(6)`,
   `constructor.js` → `_c(1)`, `capturing-reference-changes-type.js` → `_c(2)`):
   we compute zero escaping scopes and emit nothing; React memoizes. Our escape
   predicate prunes scopes React keeps (it has been over-corrected — earlier work
   moved it from *too loose* to now *too tight* on these).

---

## What exists now (the "custom blend" to replace)

`src/scopes.rs` (~1130 lines). The pipeline today:

- `infer(cfg, ranges) -> Vec<Scope>` — groups *allocations* into scopes by
  **interval containment** over mutable ranges. This is a crude stand-in for
  React's `InferReactiveScopeVariables` (which groups identifiers by overlapping
  ranges via union-find) + `MergeOverlappingReactiveScopes`. **This is where
  over-merge / over-split live.**
- `reactive_values(cfg, stable)` — reactivity by operand data-flow only. **Misses
  control-flow reactivity** (a value assigned under a reactive branch).
- `prune_non_escaping(...)` (line 437, ~270 lines) — a partial port of
  `PruneNonEscapingScopes`. Computes a value-level memoization graph. **The 31
  nothing-escapes show it over-prunes.**
- `analyze(cfg, ranges) -> Vec<ScopeInfo>` (line 733) — runs `infer`, prune,
  then a `MergeConsecutiveScopes` fold (line ~946, a partial
  `mergeReactiveScopesThatInvalidateTogether`), and computes deps via
  `access_path_key` (line 1071). **Deps are operand-derived; no control-flow
  deps, no block-scope alignment.**
- There is **no** `AlignReactiveScopesToBlockScopes` (→ the 45 multi-block bails),
  and dep propagation is not the faithful path-based reachability React does.

`ScopeInfo { scope, deps: Vec<Value>, outputs: Vec<Value> }` is the analysis
output the emitter consumes. `Scope { start, end, values, mutable }`.

---

## The faithful pipeline to port (React's order)

From `react_compiler/src/entrypoint/pipeline.rs`. Ranges already done
(`aliasing_ranges.rs`). The scope passes, in order, with sources and sizes:

```
infer_reactive_scope_variables        399   group identifiers by overlapping mutable range (union-find)
align_reactive_scopes_to_block_scopes 327   a scope can't start/end mid-block; snap to block boundaries  → fixes the 45 multi-block
merge_overlapping_reactive_scopes     419   merge scopes whose ranges overlap after alignment
prune_non_escaping_scopes            1258   a scope is kept only if a value escapes to a return / hook arg → fixes the 31
merge_reactive_scopes_that_invalidate 564   merge consecutive scopes with identical dep sets
prune_non_reactive_dependencies       250   drop deps that aren't reactive
propagate_scope_dependencies         2293   THE deps pass: reactive access paths INCLUDING control-flow reads → fixes interdependent / under-count
```

Upstream paths:
`~/Documents/Code/open-source/react-rust-pr36173/compiler/crates/react_compiler_inference/src/{infer_reactive_scope_variables, align_reactive_scopes_to_block_scopes_hir, merge_overlapping_reactive_scopes_hir, propagate_scope_dependencies_hir}.rs`
`.../react_compiler_reactive_scopes/src/{prune_non_escaping_scopes, merge_reactive_scopes_that_invalidate_together, prune_non_reactive_dependencies}.rs`

Impedance: upstream operates on `HIR` with `IdentifierId` + structured terminals.
We have a CFG (`cfg::Value`, basic blocks + `Term`, `cfg.joins`/`block_kinds` for
structured reconstruction). Port the *algorithm*; key off `Value`. The mutable
ranges are `mutability::Ranges` (produced by `aliasing_ranges`); a scope is a set
of `Value`s whose ranges overlap.

---

## Recommended port order (highest yield first)

Each pass is independent enough to land + gate on its own. Suggested sequence by
yield-per-effort:

1. **`PropagateScopeDependencies`** — biggest under-count fixer (the `props.c`
   class across multi-output + mismatch). Port the access-path dependency
   collection *including dependencies read inside control flow within the scope*.
   This is the dominant missing piece. (Large file, but the core is "for each
   reactive access path read inside the scope and defined before it, add a dep.")
2. **`PruneNonEscapingScopes`** (replace our partial `prune_non_escaping`) — the
   31 nothing-escapes. Faithful escape = transitively-aliased-into-return-or-hook.
3. **`InferReactiveScopeVariables` + `MergeOverlappingReactiveScopes`** — replace
   the interval-containment `infer` with real overlap union-find. Fixes
   over-merge / over-split (the nested-scope and 2-vs-1 cases).
4. **`AlignReactiveScopesToBlockScopes`** — the 45 multi-block bails ("can't
   memoize half a loop / branch"). Needs `cfg.joins`/`block_kinds`.
5. **`merge_reactive_scopes_that_invalidate_together`** — align our existing
   `MergeConsecutiveScopes` fold to the faithful rule.

Then revisit the emitter's multi-output bail (now backed by correct structure)
and the lowering gaps (for-of/for/try/switch — separate mechanical work).

---

## Gate protocol (MANDATORY — this is how prior ports stayed clean)

The corpus structure metric is unforgiving and the number *bounces* on refactors.
Per change:

1. `cargo build -q -p jsir-ssa && cargo test -q -p jsir-ssa` — all 10 binaries green.
2. **Snapshot before:** `./oracle/run-corpus.sh --list agree | sort > /tmp/agree_before.txt`.
3. Make the change; rebuild; `./oracle/run-corpus.sh --list agree | sort > /tmp/agree_after.txt`.
4. `comm -23 /tmp/agree_before.txt /tmp/agree_after.txt` must be **empty** (zero
   REGRESS). `comm -13` shows GAINS. **Revert on any net regression** — do not
   leave a net-negative "as a foundation." (This session lost time because a
   multi-output emitter change and a `convert_decl_to_assign` rewrite each
   regressed structure; both were reverted. Gate *every* change.)
5. The corpus run is ~5 min (Node per fixture). Batch related edits, then gate.
   `cargo run -q --example corpus -- --json` is the summary; `--list <bucket>` and
   `--show <fixture>` for drill-down.

**Never edit** `oracle/` (the React side is ground truth) or the harness.

---

## Tools (built this session — reuse them)

- `examples/inplace_probe.rs <fixture.js>` — dumps the JSIR tree (with node_ids),
  the CFG, per-instruction stmt/expr provenance, **the scopes (deps/outputs)**,
  and the in-place memoized output. The primary scope-debugging tool.
- `examples/why_miss.rs <list-file>` — buckets a list of fixtures by why they
  bail (lower error / inplace error / passthrough). Feed it `--list react_only`.
- `oracle/run-corpus.sh --show <fixture>` — side-by-side source / React / ours.

Diagnosis workflow: `--list react_only | why_miss` to find the dominant bucket →
`inplace_probe` + `--show` on 3 examples to compare our scopes to React's → port
the responsible pass → gate.

---

## Pitfalls / lessons (paid for already)

- **Gate metric is STRUCTURE** `(cache_size, block_count)`, computed over the
  React-memoized universe (715). Dep-*expression* correctness that doesn't change
  cache size won't move agree — but is still required for runtime correctness.
- **Don't reverse-engineer from fixtures.** Port the Rust pass. Crude
  fixture-shaped heuristics have regressed the gate every time; faithful ports
  land clean (this is in `MEMORY.md` at ~12M tokens of cost).
- **Control-flow reactivity** was tried as a quick heuristic earlier and
  regressed (over-marked deps). The *faithful* `PropagateScopeDependencies` is
  the right version — port it, don't approximate.
- **Escape is a tightrope:** earlier work moved it from too-loose (86 spurious
  scopes) to now too-tight (31 nothing-escapes). The faithful
  `PruneNonEscapingScopes` is the calibration.
- A `mutate(a)` *call* that mutates a scope value is currently NOT mapped into the
  scope's statement run (only `StoreMember` is — see `scope_node` construction in
  `memoize_plan::memoize_inplace`); and elided `const x = a` copies between scope
  statements break contiguity. These are emitter-side range/contiguity edges that
  may resurface once the analysis produces more/larger scopes — note for later.

## Definition of done

The scope analysis produces the same `(scopes, deps, outputs)` structure React
does, so `memoize_inplace` emits matching cache structure. Target: agree well past
244 (the old resynthesis path's number), then delete `build_layout` +
`memoize_file` + `recover_regions`. The multi-output + mismatch + nothing-escapes
buckets should collapse as the passes land.
