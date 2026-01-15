# Bug Report: JumpThreading causes crashes/incorrect results on 2+ iterations

## Summary

The `JumpThreading` optimization pass causes SIGTRAP crashes and incorrect boolean return values when run in a multi-iteration fixed-point optimization pipeline (2+ iterations). The issue manifests as:
1. Trace trap (SIGTRAP) crashes on complex programs
2. Incorrect boolean return values from recursive functions

## Reproduction

### Test Case 1: Incorrect Boolean Return (map_check.bg)

```beagle
namespace map_check
import "beagle.collections" as rmap

fn build-map(n, m) {
    if n <= 0 {
        m
    } else {
        build-map(n - 1, rmap/map-assoc(m, n, n * 10))
    }
}

fn check(n, m) {
    if n <= 0 {
        true
    } else {
        let v = rmap/map-get(m, n)
        let expected = n * 10
        if v == expected {
            check(n - 1, m)
        } else {
            false
        }
    }
}

fn main() {
    let m = build-map(50, rmap/map())
    let result = check(50, m)
    if result {
        println("got true")   // Expected
    } else {
        println("got false")  // Actual with JumpThreading + 2 iterations
    }
}
```

**Results:**
- 1 iteration: `got true` ✓
- 2 iterations: `got false` ✗

### Test Case 2: SIGTRAP Crash (binary_trees)

Any complex program with deep recursion and conditionals crashes with SIGTRAP (exit code 133) when JumpThreading is enabled with 2+ iterations.

## Isolation

The bug was isolated through systematic pass elimination:

| Passes Enabled | 1 Iteration | 2+ Iterations |
|----------------|-------------|---------------|
| Copy + Const + Fold + DCE | ✓ Works | ✓ Works |
| + ControlFlowSimplification | ✓ Works | ✓ Works |
| + JumpThreading | ✓ Works | ✗ Crashes/Wrong |
| + CfgCleanup | ✓ Works | ✗ Crashes/Wrong |
| + CSE | ✓ Works | ✗ Crashes/Wrong |

**Conclusion:** JumpThreading alone triggers the bug on 2nd iteration.

## Root Cause Hypothesis

JumpThreading modifies the CFG by:
1. Identifying trampoline blocks (blocks with only an unconditional jump)
2. Rewiring predecessors to jump directly to the target
3. Updating predecessor lists

On the second iteration, something becomes corrupted:
- Possible phi operand/predecessor mismatch (similar to CfgCleanup bug)
- Possible stale block references after block elimination
- Possible incorrect jump target rewriting

## Workaround

Limit optimization pipeline to 1 iteration when JumpThreading is included:

```rust
// Safe: single iteration
let _ = pipeline.run_until_fixed_point(&mut translator, 1);

// Unsafe: multiple iterations with JumpThreading
// let _ = pipeline.run_until_fixed_point(&mut translator, 10);
```

Or exclude JumpThreading from multi-iteration pipelines:

```rust
pipeline.add_pass(CopyPropagation::new());
pipeline.add_pass(ConstantPropagation::new());
pipeline.add_pass(ConstantFolding::new());
pipeline.add_pass(ControlFlowSimplificationPass::new());
// pipeline.add_pass(JumpThreading::new());  // DISABLED - corrupts on iteration 2+
pipeline.add_pass(CfgCleanup::new());
pipeline.add_pass(CommonSubexpressionElimination::new());
pipeline.add_pass(DeadCodeElimination::new());
```

## Impact

- **Severity**: High - causes silent incorrect code generation and crashes
- **Affected**: Any use of `JumpThreading` in a multi-iteration optimization pipeline
- **Symptoms**:
  - SIGTRAP crashes (exit code 133)
  - Wrong boolean return values from recursive functions
  - Functions returning `false` instead of `true`

## Environment

- Platform: macOS (ARM64)
- Rust: stable
- ssa-test library with Beagle IR integration

## Related Issues

Similar pattern to the CfgCleanup bug (BUG_REPORT_CFG_CLEANUP.md) where phi operand indices become misaligned with predecessor lists during multi-iteration runs.
