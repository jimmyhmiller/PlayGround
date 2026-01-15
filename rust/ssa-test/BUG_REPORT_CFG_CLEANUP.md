# Bug Report: CfgCleanup corrupts phi operands when run multiple times

## Summary

The `CfgCleanup` optimization pass corrupts phi nodes when run multiple times in a fixed-point iteration loop. The `rebuild_predecessors_from_terminators` function can ADD new predecessors to blocks without adding corresponding phi operands, causing a mismatch between predecessor count and phi operand count.

## Reproduction

Run any optimization pipeline that includes `CfgCleanup` with `run_until_fixed_point()` for more than 1 iteration on code with phi nodes and conditional branches.

### Minimal Example

```rust
// A function like:
fn test(x, args) {
    let result = if x {
        null
    } else {
        read_field(args, 0)
    }
    return result
}
```

When compiled to SSA with a phi node merging the if/else branches:
- **1 iteration**: Correct output
- **2+ iterations**: Incorrect output (phi operands become misaligned)

## Root Cause

In `src/optim/passes/cfg_cleanup.rs`, the function `rebuild_predecessors_from_terminators()`:

1. Rebuilds predecessor lists from scratch by scanning all jump targets (lines 212-227)
2. Compares old vs new predecessor lists
3. Only handles **removed** predecessors by removing phi operands (lines 261-276)
4. Does NOT handle **added** predecessors - no phi operands are added

### The Problem

When a block's predecessors change from `[A]` to `[A, B, C]`:
- The phi node still has only 1 operand (for predecessor A)
- But now there are 3 predecessors
- Phi operand index 0 is accessed for all 3 predecessors, giving wrong values for B and C

### Code Location

```rust
// src/optim/passes/cfg_cleanup.rs, lines 281-285
// Third pass: update the predecessor lists
for block in &mut translator.blocks {
    let new_preds = new_predecessors.remove(&block.id).unwrap_or_default();
    block.predecessors = new_preds;  // <-- predecessors updated without updating phis
}
```

## Debug Output

When running with `DEBUG_CFG_CLEANUP=1`, you can see blocks with phis getting more predecessors:

```
[CFG_CLEANUP] Block BlockId(5) preds changed: [BlockId(3)] -> [BlockId(2), BlockId(3), BlockId(4)] (has_phi=true)
[CFG_CLEANUP] Block BlockId(4) preds changed: [BlockId(2)] -> [BlockId(2), BlockId(3)] (has_phi=true)
```

## Suggested Fixes

### Option 1: Don't add new predecessors
Modify `rebuild_predecessors_from_terminators` to only REMOVE stale predecessors, never add new ones. New predecessors should only be added by the pass that creates the edge.

### Option 2: Add placeholder phi operands
When new predecessors are detected, add `undef` or placeholder operands to all phis in that block.

### Option 3: Validate and abort
Before updating predecessors, check if any block with phis would have predecessors added. If so, either skip the update for that block or report an error.

### Option 4: Track phi-predecessor correspondence
Store which phi operand corresponds to which predecessor explicitly (rather than relying on index correspondence), making the order irrelevant.

## Workaround

Exclude `CfgCleanup` from optimization pipelines that run multiple iterations:

```rust
// Don't include CfgCleanup in the pipeline
pipeline.add_pass(CopyPropagation::new());
pipeline.add_pass(ConstantPropagation::new());
// pipeline.add_pass(CfgCleanup::new());  // DISABLED - corrupts phis
pipeline.add_pass(DeadCodeElimination::new());

let _ = pipeline.run_until_fixed_point(&mut translator, 10);
```

## Impact

- **Severity**: High - causes silent incorrect code generation
- **Affected**: Any use of `CfgCleanup` in a multi-iteration optimization pipeline
- **Symptoms**: Wrong values returned from functions with phi nodes after if/else or switch statements
