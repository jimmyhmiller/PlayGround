I have verified the key anchors on both sides. The FINDINGS are accurate against the actual code. I have everything needed to produce the doc.

# PHASE_B_REDESIGN — Porting React's Reactive-Scope Semantics into jsir-ssa

Goal: stop emitting memo blocks React would prune (the 86 `ours_only` over-memoizations), and start handling the control-flow shapes React memoizes (currently hard-errored) — by porting the *semantics* of React's pipeline, in an order where the cheap analysis-only fixes land first and the structural codegen work lands last.

All anchors below are real and verified. React repo = `/tmp/react-rust/compiler`. Our repo = `…/js-ir-parity/claude-experiments/js-ir/crates/jsir-ssa/src`.

---

## (1) React's full reactive-scope pipeline, in order

Source of ordering: `react_compiler/src/entrypoint/pipeline.rs`.

**Upstream (feeds scope inference; produces per-identifier `mutable_range` + reactive flags):**
1. `InferTypes` — type inference so a primitive-typed call result isn't treated as allocating.
2. `AnalyseFunctions` — recursively analyse nested functions.
3. `InferMutationAliasingEffects` (pipeline.rs:272) — per-instruction Capture/Mutate/Alias/Freeze effects.
4. `InferMutationAliasingRanges` (pipeline.rs:318) — propagate effects into each identifier's `mutable_range` [start,end).
5. `InferReactivePlaces` (pipeline.rs:406) — mark which places are reactive (derive from props/state/args).

**Reactive-scope sequence:**
6. `InferReactiveScopeVariables` (pipeline.rs:449; `infer_reactive_scope_variables.rs:39`) — union-find over co-mutating identifiers → assign each disjoint group a `ScopeId`, scope.range = union of member ranges.
7. `AlignReactiveScopesToBlockScopesHIR` (pipeline.rs:539; `align_…_hir.rs:86`) — widen each scope's range outward so it starts/ends at the same structured block-scope nesting level (never half a branch/loop).
8. `MergeOverlappingReactiveScopesHIR` (pipeline.rs:550; `merge_overlapping_…_hir.rs:185`) — LIFO-stack sweep that unions any two scopes whose ranges *cross* (so the result is a disjoint-or-properly-nested forest), plus "mutate an outer scope while inner active" merges.
9. `BuildReactiveScopeTerminalsHIR` (pipeline.rs:566) — split blocks and insert `Scope`/`Goto` terminals at the aligned boundaries; renumber and re-derive ranges from the terminals.
10. `FlattenReactiveLoopsHIR` (pipeline.rs:582) — flatten scopes that wrap loops.
11. `FlattenScopesWithHooksOrUseHIR` (pipeline.rs:593) — un-memoize any scope containing a hook/`use()` (can't memoize across a hook).
12. `PropagateScopeDependenciesHIR` (pipeline.rs:613; `propagate_scope_dependencies_hir.rs:34`) — compute each scope's `Vec<ReactiveScopeDependency>` = root identifier + property *path*; hoistable-non-null fixpoint + minimal-dependency derivation.
13. `BuildReactiveFunction` (pipeline.rs:624; `build_reactive_function.rs:22`) — convert the CFG (now Scope-annotated) into a tree-structured `ReactiveFunction`.
14. `PruneUnusedLabels`.
15. `PruneNonEscapingScopes` (pipeline.rs:668; `prune_non_escaping_scopes.rs:35`) — escape analysis from roots (returns + hook args); keep a scope only if it declares/reassigns a value in the memoized set; else splice the body inline (no cache).
16. `PruneNonReactiveDependencies` — drop deps that aren't reactive.
17. `PruneUnusedScopes` (pipeline.rs:694; `prune_unused_scopes.rs:26`) — demote a scope with no own declaration, no reassignment, no return into a plain `PrunedReactiveScopeBlock`.
18. `MergeReactiveScopesThatInvalidateTogether` (pipeline.rs:707; `merge_…_that_invalidate_together.rs:33`) — fuse *adjacent* scopes that invalidate together (equal deps, producer-outputs==consumer-inputs, or always-invalidating-typed output).
19. `PruneAlwaysInvalidatingScopes` (`prune_always_invalidating_scopes.rs:50`) — demote scopes whose dep is an unmemoized always-new value.
20. `PropagateEarlyReturns` (pipeline.rs:733; `propagate_early_returns.rs:34`) — rewrite `return X` inside a surviving scope into `earlyReturnValue = X; break label`, cache `earlyReturnValue`, emit a post-memo sentinel guard.
21. Cleanup: `PruneUnusedLValues`, `PromoteUsedTemporaries`, `ExtractScopeDeclarationsFromDestructuring`, `StabilizeBlockIds`, `RenameVariables`, `PruneHoistedContexts`.
22. `codegen_function` — emit `const $ = _c(N)` + per-scope `if (dep changed) { recompute; cache } else { restore }`.

---

## (2) The GAP vs our current passes, grounded in file:line

Our entire pipeline is three functions + an emitter: `mutability::analyze` (`mutability.rs:74`), `scopes::infer` (`scopes.rs:29`), `scopes::analyze` (`scopes.rs:174`), and `memoize_plan::build_layout` (`memoize_plan.rs:49`).

| React pass | Ours: HAVE / LACK / WRONG | Anchor |
|---|---|---|
| InferMutationAliasing{Effects,Ranges} | HAVE (fused, coarser): one over-approx union-find + note_mut; inclusive `[start,end]` not half-open | `mutability.rs:74`, range convention `mutability.rs:26-27` |
| InferReactiveScopeVariables | HAVE-WRONG-SHAPE: we merge overlapping *intervals* instead of union-find over co-mutating identifiers | `scopes.rs:36-49` |
| **AlignReactiveScopesToBlockScopes** | **LACK ENTIRELY** — this is why memoize_plan hard-errors on if/else/loops | `check_soundness` rejection `memoize_plan.rs:529-535, 543-549` |
| MergeOverlappingReactiveScopes (nesting-aware) | LACK the LIFO-stack/nesting + "mutate-outer" merge; we only flat-merge overlapping intervals | `scopes.rs:43-49` |
| PropagateScopeDependencies | HAVE-NARROW: one-shot operand scan, deps are flat `Vec<Value>` (no property paths, no optional chains, no hoistable-non-null fixpoint, no range-validity check) | `scopes.rs:307-317, 361-387` |
| **PruneNonEscapingScopes** | **HAVE-WRONG**: we filter on `!outputs.is_empty()` where `outputs` = "used by any later instruction OR any terminator" (`scopes.rs:377-386`, `term_operands` includes CondBr cond + all branch args `scopes.rs:395-407`). React's escape set = only values transitively aliased into a *return* value or passed to a *hook arg*. **This mismatch is the 86 `ours_only`.** | filter `memoize_plan.rs:54`; outputs `scopes.rs:377-386` |
| PruneUnusedScopes | LACK as a transform (only the outputs filter, on the wrong predicate) | `memoize_plan.rs:54` |
| MergeReactiveScopesThatInvalidateTogether | HAVE-PARTIAL: single-dep consumer→producer fold + JSX-props transient fold; React also merges equal-dep-sets and always-invalidating-typed outputs, and is *positional/adjacent* (we are global) | `scopes.rs:300-359` |
| PruneAlwaysInvalidatingScopes | LACK entirely | — |
| **PropagateEarlyReturns** | **HAVE-INVERTED**: we hard-error on the exact dep-keyed early-return shape React memoizes | `memoize_plan.rs:255` ("early return inside branch"), `memoize_plan.rs:323` ("branch arms do not reconverge") |
| FlattenScopesWithHooksOrUse | LACK (no hook notion) — correctness gap if a hook sits inside a would-be scope | — |
| codegen | HAVE and matches: OR-reduce `$[i] !== dep`, empty-deps sentinel fallback, single running slot counter | `memoize_plan.rs:807-823`, slot threading `memoize_plan.rs:801-803` |

**Two structural prerequisites we lack for the structural passes (7/8/9/20):**
- No `fallthrough`/join field and no `BlockKind` on `cfg::Block` — React's alignment is driven entirely by `terminal.fallthrough`. We must reconstruct join points (option (b): record them during `lower.rs`, exact and cheap; option (a): dominance + post-dominance).
- No single linear `EvaluationOrder` numbering *terminals* — our `Point` (`mutability.rs:22`) numbers only instructions; React compares scope ranges against `terminal.id`.

---

## (3) Re-sequenced incremental plan

The principle: **analysis-only correctness first** (drops `ours_only` and *enables* control-flow agreement without touching codegen), then the structural/codegen work. Each step is a real port, hard-errors unhandled cases, keeps the reversible JSIR pure (we only emit through the existing `Op`-tree path; no strings, no new IR leaking across the crate boundary).

### Phase 0 — Foundations (no behavior change; preconditions for everything else)

**Step 0a. Number terminals in the Point space.**
- Change: extend `mutability::analyze` (`mutability.rs:74+`) so the linear RPO numbering assigns a `Point` to each block terminator as well as each instruction (terminal point = after the block's last instruction). Add `Ranges.term_point: HashMap<BlockId, Point>`.
- Invariant: numbering is a total order consistent with RPO; a value's def Point < its terminal-use Point.
- Gate: agree unchanged, `ours_only` unchanged, panic 0, no regression. (Pure plumbing.)

**Step 0b. Reconstruct structured join/fallthrough info.**
- Change: in `lower.rs`, when lowering `if`/`for`/`while`/`switch`/ternary/logical, record the construct head block → its join (merge) block in a new `Cfg.joins: HashMap<BlockId, BlockId>` (and a `BlockKind` enum: `Block`/`Value`/`Loop` — minimum viable is `Block` vs `Loop`). This replaces React's `terminal.fallthrough`.
- Invariant: every `CondBr` head with a reconverging diamond has exactly one join recorded; the join post-dominates the head. Hard-error in `lower.rs` if a construct is lowered without a recordable join (don't silently omit).
- Gate: agree unchanged, `ours_only` unchanged, panic 0. (Metadata only.)

> Note on range convention: React's `scope.range` is half-open `[start, end)`; ours is inclusive `[start, end]` (`mutability.rs:26-27`). **Every ported comparison must convert.** Standardize internally on half-open for the new scope-range arithmetic and convert at the `Ranges` boundary. This is called out again per-step.

### Phase 1 — Escape-based pruning (THE fix for the 86 `ours_only`; analysis only)

**Step 1. Port PruneNonEscapingScopes (escape graph + memoized-set DFS).**
- Change: add `scopes::prune_non_escaping(cfg, &ranges, &mut infos)` run inside `analyze` *before* the merge fold (before `scopes.rs:300`) — replacing the `outputs`-based emission predicate.
  - Seed escape roots two ways (mirror `prune_non_escaping_scopes.rs:918, 990`): (a) every `Term::Ret(Some(v))` operand; (b) every argument of a hook call (need an `is_hook(callee_name)` check — capitalized-component / `use*` naming, like React's `getHookKind`). Until we model hooks, returns dominate; default hook args to escaping when a callee name matches `use[A-Z]`.
  - Build the identifier-level dependency graph with a `MemoizationLevel` lattice (`Memoized` > `Conditional` > `Unmemoized` > `Never`; `prune_non_escaping_scopes.rs:69`, op-table `:391`). Map our `Op` kinds: `MakeObject`/`MakeArray`/`Call`/JSX-call → `Memoized`; `Member`/loads/conditional → `Conditional`; `Const`/`Bin`/`Un` → `Never`. Aliasing operands of allocations/calls become extra `Memoized` lvalues.
  - Run `compute_memoized_identifiers` DFS (`prune_non_escaping_scopes.rs:1043`, just read above): `memoized(id) = level==Memoized || (Conditional && (has_memoized_dep||force)) || (Unmemoized && force)`; on memoize, call `force_memoize_scope_dependencies` (`:1098`) — force=true through every dep of every scope the id belongs to.
  - Keep a scope iff it declares/reassigns a value in `memoized`, OR (declarations and reassignments both empty) OR (early_return_value set) — the keep-exception at `prune_non_escaping_scopes.rs:1155`. Otherwise prune (do not emit).
- Then set each surviving scope's effective `outputs` to *only the escaping values* (intersection with `memoized`), so the existing `!outputs.is_empty()` emission filter (`memoize_plan.rs:54`) and slot layout shrink correctly.
- Soundness invariant: a pruned scope's values are provably never observed across a render boundary (not transitively aliased into a return or a hook arg), so dropping its cache cannot change observable output. The `force_memoize_scope_dependencies` rule is **mandatory** — once a scope has any memoized output, all its transitive deps must be memoized or it over-invalidates. noAlias signatures: we have no DB, so assume aliasing (over-memoize slightly, never under-memoize).
- Gate: **`ours_only` drops sharply (target: the 86)**, agree non-decreasing, panic 0, no `theirs_only` regression.

**Step 2. Port PruneAlwaysInvalidatingScopes.**
- Change: add `scopes::prune_always_invalidating` after the merge fold. Track `always_invalidating` lvalues = `MakeArray`/`MakeObject`/JSX/`Call`-new results (`prune_always_invalidating_scopes.rs:50`), propagate through store/load aliasing; `unmemoized` = those produced outside any scope. If any scope *dependency* is in `unmemoized`, demote the scope (drop its cache), and propagate the marks downstream.
- Soundness invariant: a scope keyed on a value that is freshly allocated outside any scope every render would invalidate every render; demoting it is a no-op on output and removes a useless cache slot.
- Gate: `ours_only` non-increasing (likely small additional drop), agree non-decreasing, panic 0.

**Step 3. Generalize MergeReactiveScopesThatInvalidateTogether.**
- Change: extend `scopes.rs:318-359` from the single-dep case to React's three `can_merge_scopes` rules (`merge_…_that_invalidate_together.rs:440`): equal dep-sets; producer-declarations == consumer-deps (allow multiple); always-invalidating-typed output consumed by next. Add the reassignment gate (false if either scope reassigns) and `scope_is_eligible_for_merging` (`:554`).
- Note: React's merge is **positional/adjacent** (only consecutive scopes, runs broken by terminals/pruned scopes/non-const stores). Ours is global and can over-merge. Constrain to adjacency only once we have a block-ordered scope list (depends on Step 4). Until then, keep the conservative single-dep+JSX fold (don't widen the global fold, which risks over-merge). **This step is gated behind Step 4.**
- Soundness invariant: merging two scopes that provably invalidate on the same condition produces identical recompute behavior with fewer slots.
- Gate: agree non-decreasing, `ours_only` non-increasing, panic 0, no regression.

### Phase 2 — Property-path dependencies (closes dep-expression byte-exactness)

**Step 4. Port PropagateScopeDependencies (paths + validity).**
- Change: replace `ScopeInfo.deps: Vec<Value>` with `Vec<Dep>` where `Dep { root: Value, path: Vec<(PropKey, optional: bool)> }`. Build a temporaries sidemap (`propagate_scope_dependencies_hir.rs:250, 343`): a `Member` chain off an outer value records the whole chain as one dep instead of depending on the intermediate SSA temp. Add `check_valid_dependency` = def Point strictly before `scope.range.start` (`:1844`), and ref.current → bare ref collapse (`:1896`). Then derive minimal deps (drop `a.b.c` if `a.b` is also a dep). Sort by the name+path string key (`:3673`), not by SSA id (current `scopes.rs:387` sorts by Value id — **wrong once paths exist**).
- Codegen: extend `emit_op_value`/the dep RHS in `emit_scope` to fold a path into `member_expression` ops (mirror `codegen_reactive_function.rs:3132`); switch the whole chain to optional-member form if *any* entry is optional.
- Hoistable-non-null fixpoint + optional-chain collection (`propagate_scope_dependencies_hir.rs:1307`) is a larger sub-port; stage it as 4b. Until 4b, render only fully-unconditional paths and fall back to depending on the intermediate temp when non-null can't be proven (sound, slightly over-fragmented).
- Soundness invariant: `a.b.c` as one dep is semantically identical to depending on each intermediate, but matches React's emitted guard expression; the strict-before-scope.range.start validity check guarantees the dep is computed before the guard.
- Gate: agree increases (dep-expression byte-exactness on `props.a.b` components), panic 0, no regression. This step also produces the block-ordered scope list that **unblocks Step 3**.

### Phase 3 — Block-scope alignment (enables control-flow memoization without miscompile)

**Step 5. Port AlignReactiveScopesToBlockScopes.**
- Change: add `scopes::align_to_block_scopes(cfg, &joins, &mut intervals)` called inside `infer` *after* the interval merge (after `scopes.rs:49`) and *before* allocation assignment (`scopes.rs:53`). Implement the forward walk (`align_…_hir.rs:86`): maintain `active_scopes` + a stack of `active_block_fallthrough_ranges` keyed on `Cfg.joins`. On entering a block that is a recorded join: pop and pull every still-active scope's `start` back to the construct head (`:103`). On a terminal with a join (non-Branch): extend every active scope whose end crosses the construct out to the join's first point, and push the fallthrough range (`:180`). Handle goto-to-label widening for labeled break/continue (`:207`). Re-sync ranges to half-open.
- We can **drop the value-block machinery** (`:236`) initially — our ternary/logical are lowered to CondBr diamonds, not React value-blocks; only the if/loop `activeBlockFallthroughRanges` widen-start/widen-end is essential (minimum viable, per the FINDINGS).
- Then **relax `check_soundness`** (`memoize_plan.rs:543-549, 529-535`): the multi-block rejection becomes unnecessary because scopes are now block-aligned (a scope legitimately encloses the whole construct). Keep the "mutation crosses a guard" check as a *post-condition assertion* (hard-error, not silent) until Step 6 proves it can't fire.
- Soundness invariant: after alignment every scope range coincides with structured boundaries; a memo guard wrapping `[start, join)` provably encloses every mutation of every scope value (no mutation can occur on a control path that bypasses the guard).
- Gate: **agree increases on if/else fixtures** (previously hard-errored), `ours_only` non-increasing, panic 0, no regression on straight-line fixtures.

**Step 6. Port MergeOverlappingReactiveScopes (nesting-aware) + extend region recovery for nested scopes.**
- Change: replace the flat interval merge (`scopes.rs:43-49`) with the LIFO-stack crosser-union (`merge_overlapping_…_hir.rs:185`) operating on aligned half-open ranges, plus the "mutate an outer scope while inner active → union" rule (`:250`). Preserve insertion-order union roots (`:146` warns against sorting by id). Extend `memoize_plan` region emission so a scope can nest inside another (`emit_block` currently assumes flat per-block scopes, `memoize_plan.rs:158-210`).
- Soundness invariant: final scopes are disjoint-or-properly-nested; the crosser-union prevents two guards from interleaving (which would let a mutation land between two guards).
- Gate: agree non-decreasing (handles nested-scope fixtures), panic 0, no regression. Enables Step 3's adjacency-correct general merge.

### Phase 4 — Early returns (the inverted case) and loops

**Step 7. Port PropagateEarlyReturns (sentinel rewrite). Removes the hard-errors at `memoize_plan.rs:255, 323`.**
- Change:
  - Region recovery (`recover_regions`/`build_region`, `memoize_plan.rs:224-289`): stop hard-erroring on a branch arm ending in `Ret`; instead detect "return inside a surviving scope's range" and lower it to the labeled-break shape: redirect the `Ret` to jump to the scope's post-block, carrying the value via a dedicated SSA value / block-arg.
  - Add `ScopeInfo.early_return_value: Option<Value>`; the rewrite attaches to the **outermost** enclosing scope wrapping the return (`propagate_early_returns.rs:77-109`), bubbling up through inner scopes; idempotent (bail if already set).
  - The early-return value becomes an ordinary cached **output** (its own slot in the same running counter, `memoize_plan.rs:801`); initialize it at the scope head to `Symbol.for("react.early_return_sentinel")` (a *second* sentinel, distinct from `react.memo_cache_sentinel`).
  - `emit_scope` (`memoize_plan.rs:794`): after the memo if/else, emit `if (name !== Symbol.for("react.early_return_sentinel")) return name;`. **NOT** folded into the dep-keyed `$[i] !== dep` test.
  - Ensure the escape-prune (Step 1) and any unused-scope demotion KEEP a scope containing a return (`prune_unused_scopes.rs:68` `has_return_statement` rule) so the sentinel gate survives.
- Soundness invariant: the early-return decision is replayed from cache on a hit (the value is restored in the else branch, the post-memo sentinel compare re-runs), so memoization can't skip a return.
- Gate: **`ours_only` does not regress, agree increases on early-return fixtures**, panic 0. Critically: returns at top level / inside pruned scopes stay plain `Ret` (no sentinel) — the gate is "inside a surviving scope," not "is a return."

**Step 8. Loops (FlattenReactiveLoops) and hooks (FlattenScopesWithHooksOrUse).**
- Change: detect back-edges in `recover_regions` and flatten scopes wrapping loops (don't memoize across a loop body) rather than hard-erroring at `memoize_plan.rs:248`. Add `is_hook` detection and demote scopes containing a hook call (`flatten_scopes_with_hooks_or_use_hir.rs`).
- Soundness invariant: a scope spanning a loop or a hook call cannot be soundly cached (hook order / loop iteration), so flatten to plain code.
- Gate: agree increases on loop/hook fixtures, panic 0, no regression.

---

## (4) Per-step summary table

| Step | Concrete change (file/function) | Soundness invariant | Gate expectation |
|---|---|---|---|
| 0a | `mutability::analyze` — number terminals; add `term_point` | total order consistent with RPO | agree=, ours_only=, panic 0 |
| 0b | `lower.rs` — record `Cfg.joins` + `BlockKind` | join post-dominates head; hard-error if unrecordable | agree=, ours_only=, panic 0 |
| 1 | `scopes::prune_non_escaping` before `scopes.rs:300`; rewrite outputs as escaping∩memoized | pruned scope never observed across render | **ours_only ↓ (the 86)**, agree ≥, panic 0 |
| 2 | `scopes::prune_always_invalidating` after merge | always-fresh dep ⇒ cache is a no-op | ours_only ≤, agree ≥, panic 0 |
| 3 | generalize `scopes.rs:318-359` to 3 merge rules + adjacency | same-invalidation merge is behavior-identical | agree ≥, ours_only ≤, panic 0 (gated on Step 4) |
| 4 | `ScopeInfo.deps: Vec<Dep>` (root+path); path-aware codegen in `emit_scope`; sort by name-key | path dep ≡ intermediate deps, matches guard expr | **agree ↑** (path components), panic 0 |
| 5 | `scopes::align_to_block_scopes` after `scopes.rs:49`; relax `check_soundness:543` | aligned guard encloses every mutation | **agree ↑** (if/else), ours_only ≤, panic 0 |
| 6 | LIFO crosser-union replacing `scopes.rs:43-49`; nested-scope emit | scopes disjoint-or-nested | agree ≥ (nested), panic 0 |
| 7 | early-return sentinel: `recover_regions` lowering + `ScopeInfo.early_return_value` + post-memo guard in `emit_scope` | early return replayed from cache | **agree ↑** (early returns), ours_only =, panic 0 |
| 8 | loop flatten in `recover_regions`; `is_hook` demotion | can't cache across loop/hook | agree ↑ (loops/hooks), panic 0 |

Across all steps: **no `theirs_only` regression, panic count 0, reversible JSIR stays pure** (emission only through the existing `Op`-tree path; unhandled shapes hard-error per CLAUDE.md, never silently skip or return a stub).

---

## (5) Why this ordering fixes the inverted early-return problem

The early-return inversion is *two* bugs that look like one, and they must be fixed in the right order or each masks the other:

1. **We over-memoize (the 86), then hard-error when control flow appears.** Today `outputs` = "used by any later instruction OR any terminator" (`scopes.rs:377-386`, `term_operands` includes the CondBr cond and *all* branch args at `scopes.rs:395-407`). So a value that merely flows as a branch argument counts as an "output," we keep its scope, and then `recover_regions`/`check_soundness` trips over the branch and hard-errors (`memoize_plan.rs:255, 323, 543`). The early-return diamond is exactly this shape: a value reaching a `Ret` inside a branch arm. Because our escape predicate is wrong, we are simultaneously memoizing scopes React prunes *and* refusing the control flow React memoizes.

2. **Fixing escape FIRST (Step 1) collapses the noise.** Once `outputs` = "transitively aliased into a return or a hook arg" (React's actual escape set), the spurious branch-argument "outputs" disappear. Many would-be scopes around control flow are pruned to nothing — which both drops `ours_only` *and* removes scopes that would otherwise force `check_soundness` to confront a branch it can't yet handle. The surviving scopes are precisely the ones whose values genuinely escape, which is the set early-return cares about.

3. **Alignment (Step 5) then makes the remaining control-flow scopes legal.** With escape-correct scopes block-aligned to whole constructs, `check_soundness`'s multi-block rejection becomes unnecessary, so the diamond stops being a hard-error and becomes a structured `If` region.

4. **Only THEN does the early-return sentinel (Step 7) have a correct foundation.** React's sentinel fires *only for a return inside a surviving reactive scope*, and replays the decision on the scope's *existing* memo guard. If we tried Step 7 before Step 1, we'd attach sentinels to scopes that shouldn't exist (the over-memoized ones), and the "surviving scope" gate — the whole correctness condition — would be meaningless. By the time Step 7 runs, "surviving scope" is well-defined (escape-pruned, aligned, dep-keyed), so the sentinel guard lands on exactly the scopes React puts it on, and a top-level or pruned-scope return correctly stays a plain `Ret`.

In short: the inversion is downstream of the wrong escape predicate. Fix escape → the spurious scopes vanish and the genuine control-flow scopes surface; align → they become structurally legal; sentinel → they replay early returns from cache. Doing it in any other order means the early-return pass attaches semantics to scopes that are themselves wrong.

---

### Cross-repo anchor index
- React escape DFS (just read): `prune_non_escaping_scopes.rs:1043` (`compute_memoized_identifiers`), `:1098` (`force_memoize_scope_dependencies`), keep-exception `:1155`.
- React pipeline order: `entrypoint/pipeline.rs:449,539,550,566,582,593,613,624,668,694,707,733`.
- React early return: `propagate_early_returns.rs:26,77,112,185`; codegen `codegen_reactive_function.rs:861-905`; two sentinels `:75-76`.
- React deps/paths: `propagate_scope_dependencies_hir.rs:34,250,343,1307,1844,1896`; codegen `codegen_reactive_function.rs:3132,3673`.
- React align/merge: `align_reactive_scopes_to_block_scopes_hir.rs:86,103,180,207,292`; `merge_overlapping_reactive_scopes_hir.rs:146,185,250,332`.
- Ours: escape predicate bug `scopes.rs:377-386,395-407`; emission filter `memoize_plan.rs:54`; early-return hard-errors `memoize_plan.rs:255,323`; multi-block / mutation-cross gates `memoize_plan.rs:529-535,543-549`; deps `scopes.rs:307-317,361-387`; merge fold `scopes.rs:300-359`; range convention `mutability.rs:26-27`; Point `mutability.rs:22`.