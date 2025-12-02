# func13 Test Failure Investigation Report

## Status
- **Tests Passing**: 14/15 (was 13/15 before fixes)
- **Fixed**: func12 ✅
- **Remaining**: func13 ❌

## Problem Summary

The func13 test fails because the Rust implementation generates SVG output that differs from the TypeScript reference:
- **TypeScript**: 3133 lines, 316 paths
- **Rust**: 3085 lines, 300 paths
- **Difference**: 48 fewer lines, 16 fewer paths

## Root Cause

The missing 16 paths are all **backedge dummy nodes** positioned at `x=1229.5` (which comes from node position `x=1213` + `PORT_START=16`).

### Key Finding: Nodes Are Created But Wrong Position

**Critical Discovery**: The backedge dummies ARE being created in Rust, but:
1. Rust creates 4 nodes per layer in layers 15-19
2. TypeScript creates 5 nodes per layer in layers 15-19
3. The missing 5th node is an additional backedge dummy

## Detailed Analysis

### The Missing Backedge Chain

TypeScript has a chain of backedge dummies at `x=1213` that spans:
- Layers 1-14: ✅ Present in both TS and Rust
- **Layers 15-19**: ❌ Missing in Rust (this is the bug)
- Layers 20+: ✅ Present in both TS and Rust

**TypeScript backedges at x=1229.5**: 56 paths
**Rust backedges at x=1229.5**: 40 paths
**Missing**: Exactly 16 paths (matching the total difference)

### Layer 15 Detailed Comparison

**TypeScript Layer 15** (5 nodes):
```
Node 47: x=80,   is_dummy=true,  flags=1 (LEFTMOST_DUMMY), dst=52
Node 48: x=200,  is_dummy=false, flags=0 (Block 17), dst=54,53
Node 49: x=640,  is_dummy=false, flags=0 (Block 22), dst=48
Node 50: x=1031, is_dummy=true,  flags=6 (IMMINENT_BACKEDGE_DUMMY), dst=49
Node 51: x=1213, is_dummy=true,  flags=2 (BACKEDGE_DUMMY_START), dst=46  ← MISSING IN RUST
```

**Rust Layer 15** (4 nodes after backedge dummy creation):
```
Node 47: x=80,   is_dummy=true,  flags=1, dst=[51]
Node 48: x=200,  is_dummy=false, flags=0 (Block 17), dst=[53,52]
Node 49: x=640,  is_dummy=false, flags=0 (Block 22), dst=[48]
Node 50: x=1031, is_dummy=true,  flags=6, dst=[49]
```

**Key Observation**: TypeScript has TWO different backedge dummy types in layer 15:
- Node 50: `flags=6` = IMMINENT_BACKEDGE_DUMMY (4) + BACKEDGE_DUMMY_START (2)
- Node 51: `flags=2` = BACKEDGE_DUMMY_START (chain continuation)

Rust only has one (Node 50 with flags=6).

### Initial Node Positions

After `make_layout_nodes()` completes, ALL nodes in layer 15 start at `x=20`:
```
[0] node_id=47, is_dummy=true, x=20
[1] node_id=48, is_dummy=false, x=20
[2] node_id=49, is_dummy=false, x=20
[3] node_id=50, is_dummy=true, x=20
```

The positioning happens during `straighten_edges()` and subsequent layout passes.

### Backedge Dummy Creation Logic

Location: `src/graph.rs:766-838`

The code creates backedge dummies for blocks that are "rightmost in any loop":

```rust
for pending_dummy in &pending_loop_dummies {
    if pending_dummy.block_number == block.number {
        // Find the backedge block for this loop
        if let Some(loop_header) = self.blocks_by_id.get(&pending_dummy.loop_id) {
            let backedge_block = self.blocks.iter()
                .find(|b| b.attributes.contains(&"backedge".to_string())
                    && b.successors.contains(&loop_header.number));

            if let Some(backedge) = backedge_block {
                let mut backedge_dummy = LayoutNode::new_dummy_node(
                    node_id,
                    backedge.clone(),
                    layer
                );

                // Chain to previous dummy OR mark as IMMINENT
                if let Some(&prev_dummy_id) = latest_dummies_for_backedges.get(&backedge.number) {
                    // Has previous: chain to it
                    backedge_dummy.dst_nodes[0] = Some(prev_dummy_id);
                } else {
                    // First dummy: mark as IMMINENT
                    backedge_dummy.flags |= IMMINENT_BACKEDGE_DUMMY;
                }

                layout_nodes_by_layer[layer].push(backedge_dummy);
                latest_dummies_for_backedges.insert(backedge.number, node_id);
                node_id += 1;
            }
        }
    }
}
```

**Debug output confirms**:
- Backedge dummies ARE being created for layers 15-19
- Each layer gets 1 backedge dummy created
- `backedge_block found: true` for all layers
- Dummies are pushed to `layout_nodes_by_layer`

But TypeScript creates 2 dummies per layer, not 1!

## Blocks and Loops in Layers 15-19

All blocks in layers 15-19 belong to **loop 17** (`loop_id=BlockID(17)`):

```
Layer 15: Block 17 (x=200), Block 22 (x=640) - loop_id=BlockID(17)
Layer 16: Block 18 (x=200) - loop_id=BlockID(17)
Layer 17: Block 19 (x=200) - loop_id=BlockID(17)
Layer 18: Block 20 (x=200) - loop_id=BlockID(17)
Layer 19: Block 21 (x=200) - loop_id=BlockID(17)
```

The `pending_loop_dummies` correctly identifies the rightmost blocks:
```
Layer 15: block_number=BlockNumber(22) for loop BlockID(17)
Layer 16: block_number=BlockNumber(18) for loop BlockID(17)
Layer 17: block_number=BlockNumber(19) for loop BlockID(17)
Layer 18: block_number=BlockNumber(20) for loop BlockID(17)
Layer 19: block_number=BlockNumber(21) for loop BlockID(17)
```

## What's Different Between TypeScript and Rust?

### Hypothesis: Dual Backedge Dummy System

TypeScript appears to create TWO types of backedge dummies in some cases:

1. **IMMINENT_BACKEDGE_DUMMY** (flags=6): Connects directly to the backedge block
2. **Chain Continuation Dummy** (flags=2): Connects to the previous dummy in the chain

Rust's current implementation only creates ONE dummy per layer, which gets:
- Either IMMINENT flag (if first in chain)
- Or chains to previous (if continuing chain)

But TypeScript might be creating BOTH types simultaneously.

### Key Question

**Why does TypeScript create 5 nodes but Rust creates 4?**

The answer likely lies in how TypeScript handles the transition between:
- Layers 1-14: Backedge chain exists at x=1213
- Layer 15: First layer where blocks in loop 17 appear
- Need to connect the existing chain to the new loop structure

TypeScript might create:
1. A continuation of the old chain (Node 51, flags=2, x=1213)
2. A new IMMINENT dummy for the new block (Node 50, flags=6, x=1031)

## Code Locations

### Rust Implementation

**Backedge dummy creation**: `src/graph.rs:766-838`
- Creates one dummy per rightmost block
- Chains to previous OR marks as IMMINENT
- Uses `latest_dummies_for_backedges` to track chain

**Backedge chain tracking**: `src/graph.rs:550`
```rust
let mut latest_dummies_for_backedges: HashMap<BlockNumber, LayoutNodeID> = HashMap::new();
```

**Node positioning**: `src/graph.rs:500`
- `straighten_edges()`: Initial positioning
- `finangle_joints()`: Joint calculations
- `verticalize()`: Vertical positioning

### TypeScript Reference

**Location**: `/Users/jimmyhmiller/Documents/Code/open-source/iongraph2/src/Graph.ts`

To debug, added logging at line 391-399:
```typescript
for (let layer = 15; layer <= 19 && layer < layoutNodesByLayer.length; layer++) {
  const nodes = layoutNodesByLayer[layer];
  console.error(`TS Layer ${layer}: ${nodes.length} nodes`);
  for (const node of nodes) {
    console.error(`  Node ${node.id}: x=${node.pos.x}, is_dummy=${!node.block}, flags=${node.flags}, dst=${node.dstNodes.map(n => n.id)}`);
  }
}
```

## Debugging Commands

### Run Rust with debug output:
```bash
DEBUG_FUNC13=1 cargo test test_mega_complex_func13 -- --nocapture 2>&1
```

### Run TypeScript to generate reference:
```bash
cd /Users/jimmyhmiller/Documents/Code/open-source/iongraph2
npm run build-package
node generate-svg-function.mjs examples/mega-complex.json 13 0 output.svg
```

### Compare specific layers:
```bash
# Count nodes per layer
DEBUG_FUNC13=1 cargo test test_mega_complex_func13 -- --nocapture 2>&1 | grep "Layer 1[5-9]: .* total nodes"

# See pending loop dummies
DEBUG_FUNC13=1 cargo test test_mega_complex_func13 -- --nocapture 2>&1 | grep -A 3 "pending_loop_dummies"

# See backedge dummy creation
DEBUG_FUNC13=1 cargo test test_mega_complex_func13 -- --nocapture 2>&1 | grep -A 5 "Creating backedge dummy"
```

## Next Steps for Investigation

1. **Compare TypeScript's backedge dummy creation logic**
   - Look for where TypeScript creates multiple dummies per layer
   - Check if there's logic for creating both IMMINENT and continuation dummies simultaneously
   - File: `Graph.ts`, search for backedge dummy creation

2. **Trace through `latest_dummies_for_backedges` in TypeScript**
   - See when/how TypeScript updates this map
   - Compare with Rust's implementation
   - Might reveal why TypeScript creates extra dummies

3. **Check for edge cases in loop transitions**
   - Layer 14 is the last layer before the gap
   - Layer 15 is the first layer with blocks in loop 17
   - This transition might require special handling

4. **Investigate forward edge dummies**
   - Maybe the "missing" dummy isn't a backedge dummy at all
   - Could be a forward edge dummy that spans these layers
   - Check `active_edges` handling around layers 15-19

5. **Compare the actual dummy node creation calls**
   - Add logging to see EVERY `new_dummy_node` call
   - Compare counts between TS and Rust
   - Find which specific dummy creation is missing

## Test Data

**Input**: `tests/fixtures/mega-complex.json`
- Function 13: "-e:25"
- Pass 0: "BuildSSA"
- 68 MIR blocks
- Complex nested loop structure

**Expected output**: `tests/fixtures/ts-mega-complex-func13-pass0.svg`
- Generated by TypeScript
- 3133 lines, 316 paths
- 1273x16108 dimensions

## Related Fixes Applied

### func12 Fixes (Completed)

1. **PORT_START bug** (`src/graph.rs:1707-1710`):
   - Dummy nodes weren't getting PORT_START added when rendering edges
   - Fixed by adding PORT_START for both dummy and non-dummy nodes

2. **IMMINENT_BACKEDGE_DUMMY rendering** (`src/graph.rs:1651-1693`):
   - Was checking node flags BEFORE destination loop, skipping all destinations
   - Fixed by moving check INSIDE the loop to match TypeScript behavior

These fixes brought tests from 13/15 to 14/15 passing.

## Conclusion

The func13 issue is a **node count mismatch** in layers 15-19 where TypeScript creates 5 nodes per layer but Rust creates only 4. The missing node is a backedge dummy at x=1213 that should be part of a backedge chain.

The root cause appears to be a subtle difference in how TypeScript and Rust handle backedge dummy creation when a backedge chain continues through layers that have new blocks in the same loop. TypeScript seems to create both a continuation dummy AND a new IMMINENT dummy, while Rust only creates one.

Further investigation should focus on comparing the exact TypeScript logic for backedge dummy creation to understand when and why it creates multiple dummies per layer.
