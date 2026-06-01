# Alignment implementation spec (from the hand-driven tractability probe)

Target fixture: `ssa-leave-case.js` (simplest if/else React memoizes; cache=3, ONE
dep-keyed memo block, no nested sentinels). React's output:

```js
const $ = _c(3);
let t0;
if ($[0] !== props.p0 || $[1] !== props.p1) {
  const x = []; let y;
  if (props.p0) { x.push(props.p1); y = x; }
  t0 = <Stringify>{x}{y}</Stringify>;
  $[0] = props.p0; $[1] = props.p1; $[2] = t0;
} else { t0 = $[2]; }
return t0;
```

We currently HARD-ERROR. Exact blockers traced (committed code):
- `memoize_plan.rs:559-567` — `x.push(props.p1)` inside the consequent mutates `x`,
  but the Call isn't an "owned instruction" of x's scope (owned = define / StoreMember,
  single-block) → "mutation crosses a branch boundary".
- `memoize_plan.rs:575-581` — "each scope must live in exactly one block."

Foundation already committed and usable: `Cfg.joins: HashMap<head,join>`,
`Cfg.block_kinds`, `Ranges.term_point` (steps 0a/0b), the if/else region walk
(`Node::If` in `emit_node`, memoize_plan.rs:134), and PruneNonEscapingScopes.

## Why incremental gated steps can't do this (and agents thrash)
The three changes below must land TOGETHER to flip the gate — none moves `agree`
alone, and a partial combination NET-REGRESSES (newly-memoized if/else fixtures come
out as `mismatch`, and adjacent fixtures break). That is why the per-pass workflow
stalled and a single broad agent thrashed for hours. This needs ONE coherent change.

## The coordinated 3-piece change (well-scoped)

### Piece 1 — Align scope ranges to constructs (`scopes.rs::infer`, after the merge at :49)
For each scope whose `[start,end]` partially overlaps a construct `head..join`
(found via `cfg.joins` + `ranges.term_point`): snap the range outward so it encloses
the whole construct (start ≤ head's point, end ≥ join's first point). Re-derive owned
region as the contiguous block set `[head .. join)`. Record per aligned scope its
`construct_head`/`join` so emission can wrap it. Sound because the guard then encloses
every mutation on every control path (no path bypasses it).

### Piece 2 — Region-aware soundness (`memoize_plan.rs::check_soundness`)
- Replace "scope owns instructions in exactly ONE block" (:575) with "scope owns a
  contiguous region `[head..join)`"; the single-block case stays a degenerate region.
- The mutation-cross check (:559) passes if the mutating instruction's block lies
  WITHIN the scope's aligned region (it's inside the guard). Keep the hard-error for
  mutations genuinely outside the aligned region.

### Piece 3 — Guard wraps a region node (`memoize_plan.rs` emission, EmitCtx)
Today a scope's guard is emitted at its last-owned instruction inside one block
(emit_block), and `Node::If` is emitted separately. Change: a scope aligned to a
construct emits as `if (deps changed) { <the If region node> ; outputs ; cache } else
{ restore }` — i.e. the recovered `Node::If` (already built by the region walk) is
emitted INSIDE the guard consequent, then output assignments + slot writes, with the
else branch restoring outputs from cache. Bind scope→region (from Piece 1) so the
emitter knows which scope wraps which `Node::If`.

## Verification (per the gate)
- `corpus --show ssa-leave-case.js` → ours (cache,blocks) == react (3,1); diff byte-near.
- Run emitted JS under Node with the fixture's `sequentialRenders` → value-equal +
  reference-stable.
- Full corpus: agree must RISE (ssa-leave-case + the if/else cluster), ours_only not up,
  panic 0, NO regression on the 31 currently-agreeing fixtures.
- Anything not soundly+structurally matchable still HARD-ERRORS (stays react_only).

## Scope estimate
This is React's AlignReactiveScopesToBlockScopes + the emission binding — a real,
multi-hour, single-session coordinated change across scopes.rs + memoize_plan.rs.
Tractable and well-defined; NOT a quick edit. Repeat per control-flow family
(early-return adds the sentinel, loops add flatten).
