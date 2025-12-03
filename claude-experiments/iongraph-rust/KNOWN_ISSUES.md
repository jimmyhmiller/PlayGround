# Known Issues

## Test Results: 1671/2742 Passing (61% Success Rate)

### Recent Fixes (This Session)

Two critical bugs were fixed that improved test pass rate:

#### Bug 1: Block Order Not Preserved
- **Issue**: Using `BTreeMap` sorted blocks by number, destroying original order from `pass.mir.blocks`
- **Impact**: Blocks within the same layer were positioned incorrectly
- **Fix**: Use `Vec` with `HashMap` for index lookup to preserve original array order

#### Bug 2: `find_loops` Skipping Root Blocks  
- **Issue**: Root blocks had `loop_id` pre-initialized to their own ID before `find_loops` was called
- **Impact**: `find_loops` saw the matching `loop_id` and returned early, skipping entire subgraphs
- **Result**: Blocks like Block 7 (loop exit) never got correct layer assignment
- **Fix**: Don't pre-set `loop_id` for roots - let `find_loops` handle it during traversal

**Result**: +91 tests passing (1580 → 1671)

### Remaining Failures (1071 tests)

The remaining failures likely stem from similar issues in the layout pipeline:

1. **Layer assignment edge cases**: Some complex loop structures may still have incorrect layer assignments
2. **Edge straightening differences**: The `straighten_edges` algorithm may not match TypeScript exactly
3. **Joint track calculation**: Subtle differences in how overlapping edges are assigned to tracks
4. **Dummy node positioning**: X-coordinate assignment for dummy nodes during edge straightening

### Technical Analysis

The Rust implementation now correctly handles:
- ✅ Original block order preservation (critical for layout)
- ✅ Loop hierarchy detection (`find_loops`)
- ✅ Outgoing edge tracking for loop headers
- ✅ Layer assignment with loop height offsets
- ✅ Backedge handling

Areas that may still need investigation:
- Edge straightening iterations and convergence
- Nearly-straight edge detection threshold
- Block width calculations for dummy nodes
- Joint offset calculations for overlapping edges

### Suggested Next Steps

1. **Pick another failing test** and diff the SVG output line-by-line
2. **Compare block positions** - if positions differ, trace back to layout algorithm
3. **Check edge paths** - if paths differ, investigate `straighten_edges` or `finangle_joints`
4. **Add debug logging** selectively to trace specific differences

### How to Debug

```bash
# Run a specific failing test with output
cargo test test_while_loop_func0_pass17 -- --nocapture 2>&1 | head -50

# Generate Rust SVG for manual comparison
cargo run -- ion-examples/while-loop.json 0 17 /tmp/rust.svg

# Generate TypeScript SVG
cd /Users/jimmyhmiller/Documents/Code/open-source/iongraph2
node generate-svg-function.mjs "/path/to/ion-examples/while-loop.json" 0 17 /tmp/ts.svg

# Compare
diff /tmp/ts.svg /tmp/rust.svg | head -50
```

### Test Infrastructure

- **2742 total tests** covering all ion-examples JSON files
- Tests compare byte-for-byte against TypeScript-generated fixtures
- Fixtures in `tests/fixtures/ion-examples/`
- Test generator: `node generate-test-suite.mjs`
