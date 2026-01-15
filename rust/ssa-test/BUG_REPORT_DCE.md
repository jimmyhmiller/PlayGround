# Bug Report: DCE causes incorrect program behavior despite not removing side-effecting instructions

## Summary

The `DeadCodeElimination` pass causes incorrect program output even though it correctly preserves all side-effecting instructions (Call operations). The bug is subtle and manifests only when DCE is enabled, despite debug output confirming all calls are kept.

## Reproduction

### Minimal Test Case

```beagle
// /tmp/simple_dce_test.bg
namespace simple_dce_test

fn incr(counter) {
    swap!(counter, fn(x) { x + 1 })
}

fn main() {
    let counter = atom(0)
    incr(counter)
    incr(counter)
    incr(counter)
    println(deref(counter))
}
```

**Expected output**: `3`
**Actual output with DCE**: `0`

### Commands to Reproduce

```bash
# With DCE enabled (broken)
cargo run -- --ssa-backend-optimized /tmp/simple_dce_test.bg
# Output: 0

# With DCE disabled (correct)
DISABLE_DCE=1 cargo run -- --ssa-backend-optimized /tmp/simple_dce_test.bg
# Output: 3
```

## Investigation Findings

### 1. DCE Correctly Preserves Side-Effecting Instructions

Debug output shows ALL Call instructions are kept:

```
DEBUG_DCE_VERBOSE=1 cargo run -- --ssa-backend-optimized /tmp/simple_dce_test.bg 2>&1 | grep "Call"
```

Output:
```
[DCE] Keeping side-effecting: Op { op: Call { builtin: false }, dest: Some(SsaVariable("v23")), sources: [Var(SsaVariable("v9"))] }
[DCE] Keeping side-effecting: Op { op: Call { builtin: true }, dest: Some(SsaVariable("v8")), sources: [...] }
... (all calls preserved)
```

### 2. DCE Removes Dead Code Correctly

DCE removes ~7700 instructions, mostly:
- `Assign` operations with unused destinations (7712)
- `LoadConstant` with unused results (8)
- Dead phi nodes with `Const(Null)` operands (5)

All of these appear to be legitimately dead code.

### 3. The Bug is NOT in Instruction Removal

Since all calls are preserved and only dead code is removed, the bug must be in some interaction between:
1. DCE's phi node removal and subsequent phi elimination
2. DCE's effect on liveness analysis
3. DCE's effect on register allocation
4. Some state corruption during DCE iteration

### 4. Bug Occurs with DCE Alone

The bug reproduces with ONLY DCE enabled (no other optimization passes):

```rust
pipeline.add_pass(DeadCodeElimination::new());  // Only this pass
```

This rules out interaction with other passes.

## Key Questions

1. **Why does removing dead code affect live code behavior?**
   - The calls to `incr` are kept
   - The calls to `swap!` within `incr` are kept
   - Yet the counter stays at 0

2. **Is there a bug in phi node removal?**
   - DCE removes phi nodes whose destinations aren't in `live_vars`
   - Could this affect phi elimination later?

3. **Is there something wrong with the liveness computation?**
   - `compute_live_variables` builds liveness from side-effecting/terminator uses
   - Could it miss some important variables?

## Hypothesis

The bug might be related to how DCE interacts with phi nodes in functions with multiple exit points or complex control flow. The `incr` function wraps `swap!`, which has internal control flow for the compare-and-swap loop.

## Environment

- Platform: macOS ARM64
- Rust: stable
- ssa-test library with Beagle IR integration

## Workaround

Disable DCE in the optimization pipeline:

```rust
// pipeline.add_pass(DeadCodeElimination::new());  // DISABLED
```

Or use environment variable:
```bash
DISABLE_DCE=1 cargo run -- --ssa-backend-optimized program.bg
```

## Files

- Test case: `/tmp/simple_dce_test.bg`
- DCE pass: `src/optim/passes/dce.rs`
- Beagle integration: `beagle/src/backend/ssa/mod.rs`
