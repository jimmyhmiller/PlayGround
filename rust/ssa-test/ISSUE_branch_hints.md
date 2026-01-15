# Feature Request: Branch Probability Hints for Block Layout Optimization

## Problem

The `ExtTspBlockLayout` algorithm uses static heuristics to weight edges during block ordering. Currently, it assumes:
- The first target of a conditional branch (ConditionalThen) is more likely (60%)
- The second target (ConditionalElse) is less likely (40%)

This heuristic is **wrong for guards and GC checks**, where:
- The first target is the **failure/error path** (rarely taken, should be cold)
- The second target is the **success/fall-through path** (usually taken, should be hot)

### Real-World Impact

In Beagle's SSA backend, this incorrect heuristic causes:
- **47% slower execution** compared to sequential block order
- GC check trampolines placed on the hot path
- Guard failure paths weighted as likely

Benchmark results for fib(40):
- Linear scan: 1.11s
- SSA with sequential layout: 1.22s (9% slower)
- SSA with ExtTSP layout: 1.65s (49% slower)

## Current API

The `OptimizableInstruction` trait provides:
```rust
fn jump_targets(&self) -> Vec<BlockId> {
    vec![]
}
```

But there's no way for instructions to indicate:
1. The semantic meaning of each target (guard failure, GC check, normal conditional)
2. Branch probability hints (likely/unlikely)

## Proposed Solution

### Option 1: Add `edge_hints()` method to `OptimizableInstruction`

```rust
/// Hint about branch probability/semantics for block layout optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BranchHint {
    /// No hint - use default heuristics
    None,
    /// This edge is likely to be taken (hot path)
    Likely,
    /// This edge is unlikely to be taken (cold path, error handling)
    Unlikely,
    /// This is a loop back-edge
    LoopBack,
}

trait OptimizableInstruction {
    // ... existing methods ...

    /// Returns branch hints for each jump target.
    ///
    /// The returned vec should have the same length as `jump_targets()`.
    /// Default implementation returns `None` for all targets (use default heuristics).
    fn branch_hints(&self) -> Vec<BranchHint> {
        vec![BranchHint::None; self.jump_targets().len()]
    }
}
```

### Option 2: Return richer edge information from `jump_targets()`

```rust
pub struct EdgeTarget {
    pub block: BlockId,
    pub hint: BranchHint,
}

trait OptimizableInstruction {
    /// Returns jump targets with optional hints.
    fn jump_targets_with_hints(&self) -> Vec<EdgeTarget> {
        self.jump_targets().into_iter()
            .map(|b| EdgeTarget { block: b, hint: BranchHint::None })
            .collect()
    }
}
```

### Option 3: Extend `EdgeKind` with semantic information

```rust
pub enum EdgeKind {
    Unconditional,
    ConditionalThen,
    ConditionalElse,
    Switch { index: usize },
    FallThrough,
    // NEW: Semantic edge kinds
    GuardSuccess,     // Guard passed, continue normally (hot)
    GuardFailure,     // Guard failed, handle error (cold)
    GCTrigger,        // GC was triggered (cold)
    GCSkipped,        // No GC needed (hot)
}
```

## Implementation in Beagle

Once the API is available, Beagle would implement:

```rust
impl OptimizableInstruction for SsaIrInstruction {
    fn branch_hints(&self) -> Vec<BranchHint> {
        match self {
            SsaIrInstruction::Op { op, .. } => match op {
                SsaOperation::GuardInt { .. } | SsaOperation::GuardFloat { .. } => {
                    // First target is fail_target (cold), second is fall_through (hot)
                    vec![BranchHint::Unlikely, BranchHint::Likely]
                }
                SsaOperation::JumpIf { is_gc_check: true, .. } => {
                    // GC checks: first target is gc_triggered (cold)
                    vec![BranchHint::Unlikely, BranchHint::Likely]
                }
                SsaOperation::JumpIf { .. } => {
                    // Normal user conditionals: no hint, use default heuristics
                    vec![BranchHint::None, BranchHint::None]
                }
                _ => vec![],
            },
            _ => vec![],
        }
    }
}
```

## ExtTspBlockLayout Changes

The `build_edges` method would use hints when available:

```rust
fn build_edges<V, I, F>(&self, translator: &SSATranslator<V, I, F>, ...) -> Vec<EdgeInfo> {
    // ...
    if let Some(last) = block.instructions.last() {
        let targets = last.jump_targets();
        let hints = last.branch_hints();  // NEW

        for (i, &to) in targets.iter().enumerate() {
            let hint = hints.get(i).copied().unwrap_or(BranchHint::None);

            let mut weight = 1.0;

            // Apply hint-based weighting
            match hint {
                BranchHint::Likely => weight *= 0.9,
                BranchHint::Unlikely => weight *= 0.1,
                BranchHint::LoopBack => weight *= 0.8, // Prefer loop exit as fall-through
                BranchHint::None => {
                    // Fall back to position-based heuristics
                    if i == 0 { weight *= self.then_branch_weight; }
                    else { weight *= 1.0 - self.then_branch_weight; }
                }
            }

            // ... rest of edge building
        }
    }
}
```

## Alternatives Considered

1. **Profile-guided optimization (PGO)**: More accurate but requires instrumentation runs. Branch hints are a simpler static approach.

2. **Pattern matching in ExtTspBlockLayout**: Could try to detect guard patterns from CFG structure, but this is fragile and instruction-set specific.

3. **Disable block layout entirely**: Current workaround in Beagle, but loses potential optimization benefits for user code.

## Priority

High - The current heuristics actively hurt performance for any language with GC or type guards.
