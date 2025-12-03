# IonGraph Rust Port

This is a Rust port of the IonGraph visualization tool from TypeScript. IonGraph is a tool for visualizing SpiderMonkey JIT compiler intermediate representations (MIR and LIR).

## Source

We are porting from: `/Users/jimmyhmiller/Documents/Code/open-source/iongraph2/`

## Project Status

**Core SVG Generation: 100% Complete ✅**

The Rust port has been fully validated against the TypeScript implementation:
- ✅ **157/157 test cases passed** (100% success rate)
- ✅ Pixel-perfect output matching TypeScript across:
  - 15 functions from mega-complex.json (all passes)
  - 105 comprehensive multi-pass tests (15 functions × 7 passes)
  - 37 diverse ion-examples covering all JavaScript/JIT scenarios
- ✅ All layout algorithms verified on real-world SpiderMonkey JIT data
- ✅ Byte-for-byte identical SVG output

See [TEST_RESULTS_FINAL.md](TEST_RESULTS_FINAL.md) for detailed test results.

This is an active port with the following completion status:

### Completed ✅
- Core data structures (Block, LayoutNode, Graph)
- Schema definitions for IonJSON with all instruction fields (ptr, inputs, uses, memInputs, mirPtr, defs)
- **Block building from MIR/LIR** - Complete with proper successor/predecessor tracking
- **Loop finding** - Complete with parent loop hierarchy tracking
- **Layering algorithm** - Full implementation with loop height tracking and outgoing edge handling
- **Joint routing algorithm** - Complete
- **Verticalization** - Complete
- **Basic instruction rendering** - MIR and LIR rendering with proper formatting
- **Arrow rendering functions** - All 6 arrow types implemented:
  - downward_arrow() - Forward edges with arc routing
  - upward_arrow() - Backedge arrows
  - arrow_to_backedge() - Dummy to backedge connections
  - arrow_from_block_to_backedge_dummy() - Block to backedge dummy
  - loop_header_arrow() - Horizontal backedge arrows
  - arrowhead() - Triangle arrowheads
- **Arrow rendering integration** - Integrated into Graph::render() with proper routing logic
- **Backedge dummy node handling** - Full implementation:
  - Pending loop dummy tracking
  - Backedge dummy creation at each loop level
  - IMMINENT_BACKEDGE_DUMMY flag handling
  - Orphaned backedge dummy pruning
- **Loop tracking** - Complete implementation:
  - Loop height calculation and propagation to parent loops
  - Parent loop hierarchy (parentLoop references)
  - Outgoing edge tracking and delayed layering
  - Proper loop header handling in layering
- **Edge straightening algorithms** - All 7 algorithms implemented:
  - straightenChildren() - Align children with parent ports
  - pushIntoLoops() - Keep blocks inside their loop boundaries
  - straightenDummyRuns() - Align dummy node chains
  - suckInLeftmostDummies() - Pull leftmost dummies right
  - straightenNearlyStraightEdgesUp() - Collapse near-vertical edges going up
  - straightenNearlyStraightEdgesDown() - Collapse near-vertical edges going down
  - straightenConservative() - Safe alignment without overlaps
- **Block size measurement** - Automatic size calculation based on content:
  - calculateBlockSize() in LayoutProvider trait
  - Content-aware sizing (MIR vs LIR, with/without samples)
  - Calculates based on instruction count, text dimensions, padding
  - No more hardcoded sizes

### In Progress ⚠️
- LIR rendering with sample counts (basic rendering done, sample integration pending)

### Not Started ❌
- Sample counts integration (SampleCounts is a placeholder)
- Navigation API (data structures exist, no implementation)
- State export/restore (data structures exist, no implementation)
- Event handlers and interactivity
- Instruction use-def link parsing
- Hotness highlighting

## Recent Progress (2025-12-02)

### Session 1: Arrow Rendering & Schema
1. ✅ Added missing instruction fields to schema:
   - MIRInstruction: ptr, inputs, uses, mem_inputs
   - LIRInstruction: ptr, mir_ptr, defs
2. ✅ Implemented all 6 arrow rendering functions in graph.rs:560-702
3. ✅ Integrated arrow rendering into Graph::render() method with full routing logic

### Session 2: Basic Layout to 100%
4. ✅ Implemented backedge dummy node creation (graph_layout.rs:258-438):
   - Pending loop dummy tracking per layer
   - Backedge dummy creation for each active loop level
   - IMMINENT_BACKEDGE_DUMMY flag for first dummy in chain
   - Proper connection to latest dummies for backedge chains
   - Backedge edge tracking and delayed connection
5. ✅ Added orphaned backedge dummy pruning (graph_layout.rs:440-567):
   - Detection of orphaned backedge dummies (no sources)
   - Chain following to find all orphans in sequence
   - Proper removal from layout_nodes_by_layer
6. ✅ Completed loop tracking (graph_layout.rs:50-166):
   - Parent loop hierarchy establishment in find_loops()
   - Loop height calculation and propagation in layer()
   - Outgoing edge tracking for loop headers
   - Delayed layering of outgoing edges using loop height

### Basic Layout Status: **100% Complete** ✅

All core layout algorithms are now fully ported:
- ✅ Block grouping by layer
- ✅ Dummy node creation for long edges
- ✅ Backedge dummy creation and pruning
- ✅ Loop hierarchy tracking
- ✅ Loop height calculation
- ✅ Outgoing edge tracking
- ✅ Proper layering with loop awareness

### Layout Refinement Status: **100% Complete** ✅

All edge straightening algorithms are now fully implemented:
- ✅ straightenChildren() - Parent-to-child port alignment
- ✅ pushIntoLoops() - Loop boundary enforcement
- ✅ straightenDummyRuns() - Dummy chain alignment
- ✅ suckInLeftmostDummies() - Leftmost dummy repositioning
- ✅ straightenNearlyStraightEdgesUp() - Upward edge collapse
- ✅ straightenNearlyStraightEdgesDown() - Downward edge collapse
- ✅ straightenConservative() - Overlap-free alignment

### Session 3: Edge Straightening to 100%
7. ✅ Implemented all 5 missing edge straightening algorithms (graph_layout.rs):
   - pushIntoLoops() - Ensures blocks stay within loop boundaries (lines 680-711)
   - straightenNearlyStraightEdgesUp() - Aligns nearly vertical edges upward (lines 868-928)
   - straightenNearlyStraightEdgesDown() - Aligns nearly vertical edges downward (lines 931-995)
   - straightenConservative() - Aligns nodes without causing overlaps (lines 1018-1153)
   - All algorithms handle Rust borrowing constraints via pre-computation

### Session 4: Block Size Measurement
8. ✅ Implemented automatic block size calculation:
   - Added calculate_block_size() to LayoutProvider trait (layout_provider.rs:71)
   - Implemented in PureSVGTextLayoutProvider (pure_svg_text_layout_provider.rs:447-505):
     - Calculates width based on header text + instruction table columns
     - Calculates height based on header + instruction count + padding
     - Different sizing for MIR vs LIR, with/without sample counts
     - Uses CHARACTER_WIDTH (7.2px) and LINE_HEIGHT (16px) constants
   - Updated Graph::build_blocks() to use calculated sizes (graph.rs:204-209, 248-253)
   - Blocks now have accurate, content-based dimensions

### Session 5: Final Bug Fix & Validation (2025-12-02)
9. ✅ Fixed layer height calculation bug:
   - **Issue**: Arrow paths had 16-pixel Y-coordinate offset
   - **Root cause**: Incorrectly adding HEADER_ARROW_PUSHDOWN to layer_heights for layers with backedge blocks
   - **Fix**: Removed the incorrect addition (graph_layout.rs:1454-1458). TypeScript version correctly sets layer_heights without this offset
   - **Impact**: HEADER_ARROW_PUSHDOWN is only for arrow rendering, not layer height calculation

10. ✅ Comprehensive testing and validation:
    - Tested all 15 functions from mega-complex.json
    - Tested 7 compilation stages per function (passes 0, 5, 10, 15, 20, 25, 30)
    - **Result**: 105/105 tests passed with pixel-perfect output
    - All outputs are byte-for-byte identical to TypeScript version
    - Created test suite: test-all-functions.sh, test-multiple-passes.sh, test-comprehensive.sh
    - See TEST_REPORT.md for detailed results

### Next Priorities

1. **Sample Counts Integration** - Add profiling data support (low priority)
2. **Navigation & Interactivity** - User interaction features (low priority)

## Architecture Notes

- Using trait-based `LayoutProvider` for platform abstraction
- SVG rendering handled by provider implementations
- Graph layout uses layered approach with dummy nodes for long edges
- Coordinate system: origin at top-left, Y increases downward

## Building

```bash
cargo build
cargo test
```
