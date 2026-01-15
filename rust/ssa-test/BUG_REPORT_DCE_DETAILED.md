# Bug Report: DCE causes incorrect program behavior - Detailed Investigation

## Summary

The `DeadCodeElimination` pass causes incorrect program output even though it correctly preserves all side-effecting instructions. This is a **subtle bug** that requires deep investigation.

## Minimal Reproduction

```beagle
// /tmp/cas_test.bg
namespace cas_test

fn main() {
    let counter = atom(0)
    compare-and-swap!(counter, 0, 1)
    println(deref(counter))
}
```

**Expected output**: `1`
**Actual output with DCE**: `0`

## Key Findings

### 1. DCE Correctly Keeps Side Effects

Debug output confirms ALL side-effecting operations are preserved:
- All `Call` instructions are kept
- All `AtomicStore`, `HeapStore`, etc. are kept
- All terminators (`Return`, `Jump`, `JumpIf`) are kept

### 2. DCE Only Removes Truly Dead Code

The only instructions removed are genuinely dead:
- Unused constant assignments like `v19 = TaggedInt(0)`
- Unused copy assignments like `v11 = v10`

### 3. BEFORE and AFTER IR Look Identical for Critical Paths

When comparing IR dumps:
- Block structure is unchanged
- Phi nodes are unchanged
- Control flow is unchanged

### 4. Simpler Operations Work Fine

- `reset!(counter, 1)` works correctly with DCE
- Only `compare-and-swap!` and `swap!` fail

## Hypothesis

The bug is NOT in what DCE removes. It's in some **side effect** of running DCE that affects subsequent compilation stages:

1. **PassResult::changed()** triggers re-running passes - could corrupt state
2. **Phi elimination** after DCE might behave differently
3. **Register allocation** might be affected by instruction numbering changes
4. **Block instruction offsets** might become inconsistent

## Files

- Test case: `/tmp/cas_test.bg`
- DCE pass: `src/optim/passes/dce.rs`
- Beagle integration: `beagle/src/backend/ssa/mod.rs`

## Environment

- Platform: macOS ARM64 (Apple Silicon)
- Rust: stable
- ssa-test library with Beagle IR integration

## Workaround

Disable DCE via environment variable:
```bash
DISABLE_DCE=1 cargo run -- --ssa-backend-optimized program.bg
```

## Debug Commands

```bash
# See what DCE removes
DEBUG_DCE_VERBOSE=1 cargo run -- --ssa-backend-optimized /tmp/cas_test.bg 2>&1 | grep "\[DCE\]"

# See IR before/after DCE
DEBUG_DCE_DUMP=1 cargo run -- --ssa-backend-optimized /tmp/cas_test.bg 2>&1

# See phi elimination
DEBUG_PHI=1 cargo run -- --ssa-backend-optimized /tmp/cas_test.bg 2>&1
```
