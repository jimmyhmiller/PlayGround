# Benchmarking Suite Summary

## Overview

A comprehensive benchmarking infrastructure has been set up for the IonGraph Rust port, enabling:
1. Statistical performance analysis with Criterion.rs
2. Direct comparison with TypeScript implementation
3. Profiling with tools like samply, perf, and Instruments
4. Automated testing and comparison scripts

## Files Added

### Benchmark Suites
- **`benches/layout_algorithms.rs`** - Benchmarks the layout pipeline
- **`benches/rendering.rs`** - Benchmarks rendering and end-to-end performance
- **`benches/README.md`** - Documentation for benchmark suites

### Profiling Binary
- **`src/bin/profile_render.rs`** - Standalone binary optimized for profiling tools
  - Supports iteration control
  - Function-specific profiling
  - Compatible with samply, perf, Instruments

### Comparison Tools
- **`compare-performance.sh`** - Automated Rust vs TypeScript comparison
  - Runs both implementations
  - Generates comparison table with speedup factors
  - JSON output for further analysis

- **`/Users/jimmyhmiller/Documents/Code/open-source/iongraph2/bench-iongraph.mjs`** - TypeScript benchmark runner
  - Statistical analysis (mean, median, p95, p99)
  - JSON output for comparison
  - Warmup phase

### Quick Access Scripts
- **`quick-bench.sh`** - Fast performance check without full statistics
- **`BENCHMARK_QUICK_START.md`** - Quick reference guide
- **`BENCHMARKS.md`** - Comprehensive benchmarking documentation
- **`BENCHMARKING_SUMMARY.md`** - This file

### Code Changes
- **`Cargo.toml`** - Added Criterion dependency and benchmark targets
- **`src/lib.rs`** - Re-exported commonly used types for easier benchmark imports
- **`src/graph.rs`** - Added `Clone` derive to `GraphOptions` and `SampleCounts`

## Usage Examples

### 1. Statistical Benchmarks (Criterion)

```bash
# Run all benchmarks
cargo bench

# Run specific suite
cargo bench --bench rendering

# View HTML reports
open target/criterion/report/index.html
```

Output includes:
- Mean, median, min, max execution times
- Statistical significance of changes
- Violin plots and trend graphs
- Outlier detection

### 2. Compare with TypeScript

```bash
./compare-performance.sh ion-examples/mega-complex.json 100
```

Output example:
```
Function        Rust (ms)       TypeScript (ms) Speedup
--------        --------        -------------- -------
Function 0      2.345           18.234         7.77x
Function 1      3.123           21.456         6.87x
Average         2.734           19.845         7.26x
```

### 3. Profile with samply

```bash
# Install samply (one-time)
cargo install samply

# Build and profile
cargo build --release --bin profile-render
samply record target/release/profile-render ion-examples/mega-complex.json 1000
```

Opens interactive flamegraph showing:
- Hot paths (wider = more time)
- Call stack relationships
- Searchable function names

### 4. Quick Performance Check

```bash
./quick-bench.sh 50
```

Runs 50 iterations and displays average time per render.

## Benchmark Structure

### Layout Pipeline Benchmarks
Tests the layout algorithm (no rendering):
- Loop finding
- Layering
- Dummy node creation
- Edge straightening
- Verticalization
- Joint routing

### Rendering Benchmarks
Tests SVG generation and end-to-end performance:
- Complete layout pipeline
- SVG rendering only (pre-computed layout)
- Full end-to-end (layout + rendering)
- Complexity comparisons

### Performance Metrics

Each benchmark reports:
- **Throughput**: Iterations per second
- **Latency**: Time per iteration
- **Statistical confidence**: 95% confidence intervals
- **Comparison**: Change vs previous runs (if available)

## Expected Performance

Based on mega-complex.json test data:

| Operation | Rust | TypeScript | Speedup |
|-----------|------|------------|---------|
| End-to-end | 1-5ms | 10-50ms | 5-10x |
| Layout only | 0.5-3ms | 8-40ms | 8-15x |
| SVG render | 0.5-2ms | 2-10ms | 4-5x |

Memory usage:
- Rust: Predictable, no GC pauses
- TypeScript: Higher due to GC overhead

## Profiling Tools Supported

### samply (macOS/Linux)
- Interactive flamegraphs
- Web-based visualization
- Easy installation: `cargo install samply`

### perf (Linux)
- System-level profiling
- Call graph analysis
- Hardware counter support

### Instruments (macOS)
- Xcode profiling tools
- Time Profiler
- Memory profiling

### Valgrind (Linux)
- Memory profiling with Massif
- Cache simulation
- Heap analysis

## Integration with Development

### Before Optimization
1. Run `cargo bench` to establish baseline
2. Save baseline: `cargo bench -- --save-baseline before`

### During Optimization
1. Make changes
2. Profile with samply to identify hotspots
3. Run specific benchmarks to verify improvements

### After Optimization
1. Run `cargo bench -- --baseline before`
2. Check for regressions in other areas
3. Update baseline if improvement is confirmed

## Continuous Integration

For CI environments:
```bash
# Run benchmarks once (no statistics) to verify they compile and run
cargo bench --no-fail-fast -- --test
```

## Future Enhancements

Potential additions:
1. Memory profiling benchmarks
2. Comparison with other graph visualization libraries
3. Benchmarks for different input sizes
4. Regression detection in CI
5. Performance dashboard

## Documentation

See these files for more details:
- **BENCHMARK_QUICK_START.md** - Quick reference for common tasks
- **BENCHMARKS.md** - Complete documentation with examples
- **benches/README.md** - Benchmark suite details
