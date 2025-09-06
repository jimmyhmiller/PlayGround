# 📊 IonGraph Snapshot Comparison Report
============================================================

## 📈 Summary Statistics
- **Total Tests**: 16
- **✅ Matched**: 5
- **❌ Missing in Rust**: 4
- **❓ Missing in TypeScript**: 7
- **🔄 Different**: 0

**🎯 Port Coverage: 31.2%**

## ✅ Matched

### Block Intermediate Representation Snapshots > should create expected block IR for complex control flow
**TypeScript**: `Graph.snapshots.test.ts.snap`
**Rust**: `iongraph_rust__graph__tests__block_ir_complex.snap`
**Details**: Content structure matches

### Block Intermediate Representation Snapshots > should create expected block IR for loop pass
**TypeScript**: `Graph.snapshots.test.ts.snap`
**Rust**: `iongraph_rust__graph__tests__block_ir_loop.snap`
**Details**: Content structure matches

### Block Intermediate Representation Snapshots > should create expected block IR for simple pass
**TypeScript**: `Graph.snapshots.test.ts.snap`
**Rust**: `iongraph_rust__graph__tests__block_ir_simple.snap`
**Details**: Content structure matches

### Layout Node Structure Snapshots > should create expected layout structure for complex control flow
**TypeScript**: `Graph.snapshots.test.ts.snap`
**Rust**: `iongraph_rust__graph__tests__layout_complex_pass.snap`
**Details**: Content structure matches

### Layout Node Structure Snapshots > should create expected layout structure for simple pass
**TypeScript**: `Graph.snapshots.test.ts.snap`
**Rust**: `iongraph_rust__graph__tests__layout_simple_pass.snap`
**Details**: Content structure matches

## ❌ Missing In Rust

### Instruction Highlighting Snapshots > should create expected highlighting state
**TypeScript**: `Graph.snapshots.test.ts.snap`
**Rust**: *Not implemented*
**Details**: No corresponding Rust test found

### Layout Node Structure Snapshots > should create expected layout structure for loop pass
**TypeScript**: `Graph.snapshots.test.ts.snap`
**Rust**: *Not implemented*
**Details**: No corresponding Rust test found

### Navigation State Snapshots > should create expected navigation state after navigation sequence
**TypeScript**: `Graph.snapshots.test.ts.snap`
**Rust**: *Not implemented*
**Details**: No corresponding Rust test found

### Navigation State Snapshots > should create expected navigation state after selection
**TypeScript**: `Graph.snapshots.test.ts.snap`
**Rust**: *Not implemented*
**Details**: No corresponding Rust test found

## ❓ Missing In Ts

### svg_arrow_structure_complex
**Rust**: `iongraph_rust__graph__tests__svg_arrow_structure_complex.snap`
**TypeScript**: *Not found*
**Details**: Rust-specific test (not in original TypeScript)

### layout_metrics_snapshot_loop
**Rust**: `iongraph_rust__graph__tests__layout_metrics_snapshot_loop.snap`
**TypeScript**: *Not found*
**Details**: Rust-specific test (not in original TypeScript)

### svg_arrow_structure_loop
**Rust**: `iongraph_rust__graph__tests__svg_arrow_structure_loop.snap`
**TypeScript**: *Not found*
**Details**: Rust-specific test (not in original TypeScript)

### layout_metrics_snapshot_complex
**Rust**: `iongraph_rust__graph__tests__layout_metrics_snapshot_complex.snap`
**TypeScript**: *Not found*
**Details**: Rust-specific test (not in original TypeScript)

### layout_metrics_snapshot_simple
**Rust**: `iongraph_rust__graph__tests__layout_metrics_snapshot_simple.snap`
**TypeScript**: *Not found*
**Details**: Rust-specific test (not in original TypeScript)

### layout_switch_like_pass
**Rust**: `iongraph_rust__graph__tests__layout_switch_like_pass.snap`
**TypeScript**: *Not found*
**Details**: Rust-specific test (not in original TypeScript)

### svg_arrow_structure_simple
**Rust**: `iongraph_rust__graph__tests__svg_arrow_structure_simple.snap`
**TypeScript**: *Not found*
**Details**: Rust-specific test (not in original TypeScript)
