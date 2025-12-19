# IonGraph Rust Benchmarks

This document describes the benchmarking suite for the IonGraph Rust port.

## Quick Start

```bash
# Run all benchmarks
cargo bench

# Compare Rust vs TypeScript performance
./compare-performance.sh ion-examples/mega-complex.json 100

# Profile with samply
cargo install samply
cargo build --release --bin profile-render
samply record target/release/profile-render ion-examples/mega-complex.json 1000

# View benchmark reports
open target/criterion/report/index.html
```

## Overview

The benchmark suite uses [Criterion.rs](https://github.com/bheisler/criterion.rs), a statistics-driven benchmarking library for Rust. It provides:

- Statistical analysis of performance
- Detection of performance regressions
- HTML reports with graphs
- Comparison between benchmark runs

## Running Benchmarks

### Run All Benchmarks

```bash
cargo bench
```

This will run all benchmark suites and generate HTML reports in `target/criterion/`.

### Run Specific Benchmark Suite

```bash
# Run only layout algorithm benchmarks
cargo bench --bench layout_algorithms

# Run only rendering benchmarks
cargo bench --bench rendering
```

### Run Specific Benchmark

```bash
# Run only the find_loops benchmark
cargo bench --bench layout_algorithms find_loops

# Run only end-to-end benchmarks
cargo bench --bench rendering end_to_end
```

### Save Baseline

To compare performance between changes:

```bash
# Save current performance as baseline
cargo bench -- --save-baseline before-optimization

# Make your changes...

# Compare against baseline
cargo bench -- --baseline before-optimization
```

## Benchmark Suites

### 1. Layout Algorithms (`layout_algorithms.rs`)

Benchmarks individual layout algorithm stages:

- **find_loops**: Loop detection algorithm performance
- **layer_algorithm**: Block layering (longest path algorithm)
- **create_dummy_nodes**: Dummy node creation for long edges
- **edge_straightening**: All edge straightening algorithms combined
- **verticalization**: Vertical coordinate assignment
- **joint_routing**: Edge routing with joint points

Each benchmark tests the first 5 functions from `mega-complex.json`.

### 2. Rendering (`rendering.rs`)

Benchmarks rendering and end-to-end performance:

- **complete_layout**: Full layout pipeline (no rendering)
- **svg_rendering**: SVG generation only (layout pre-computed)
- **end_to_end**: Complete pipeline from function data to SVG
- **complexity_comparison**: Compare performance across different input complexities
- **block_building**: Initial block construction from function data

These benchmarks test the first 10 functions from `mega-complex.json`.

## Interpreting Results

Criterion provides detailed statistics for each benchmark:

```
find_loops/function/0   time:   [45.123 µs 45.456 µs 45.789 µs]
                        change: [-2.1% -1.5% -0.9%] (p = 0.00 < 0.05)
                        Performance has improved.
```

- **time**: Lower bound, estimate, upper bound (95% confidence interval)
- **change**: Performance change vs. previous run (if available)
- **p-value**: Statistical significance (< 0.05 indicates significant change)

## HTML Reports

After running benchmarks, open the HTML reports:

```bash
# macOS
open target/criterion/report/index.html

# Linux
xdg-open target/criterion/report/index.html

# Windows
start target/criterion/report/index.html
```

The reports include:

- Violin plots showing performance distribution
- Line charts comparing multiple runs
- Statistical analysis and outlier detection
- Detailed timing breakdowns

## Performance Targets

Based on the TypeScript implementation comparison:

| Operation | Target | Notes |
|-----------|--------|-------|
| End-to-end rendering | < 50ms | Per complex function |
| Layout pipeline | < 30ms | Without SVG generation |
| SVG rendering | < 20ms | Pre-computed layout |
| Find loops | < 5ms | Per function |
| Layering | < 10ms | Per function |

These are guideline targets for typical complex functions.

## Profiling

For more detailed profiling, use these tools:

### Flamegraph (macOS/Linux)

```bash
cargo install flamegraph

# Profile a specific benchmark
cargo flamegraph --bench rendering -- end_to_end --bench
```

### perf (Linux)

```bash
cargo bench --bench rendering -- --profile-time=10 end_to_end
```

### Instruments (macOS)

```bash
# Build with release profile
cargo build --release --benches

# Run with Instruments
instruments -t "Time Profiler" target/release/deps/rendering-*
```

## Adding New Benchmarks

To add a new benchmark:

1. Create a new function in the appropriate benchmark file
2. Use the existing patterns for consistency
3. Add to the `criterion_group!` macro at the bottom
4. Document expected performance characteristics

Example:

```rust
fn bench_my_algorithm(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_algorithm");

    let ion_json = load_test_data("ion-examples/mega-complex.json");

    for (idx, func) in ion_json.functions.iter().enumerate().take(5) {
        group.bench_with_input(
            BenchmarkId::new("function", idx),
            func,
            |b, func| {
                b.iter(|| {
                    // Your benchmark code here
                    black_box(my_algorithm(black_box(func)));
                });
            },
        );
    }

    group.finish();
}
```

## Test Data

Benchmarks use the following test data:

- **mega-complex.json**: 15 complex functions with various JIT optimization passes
  - Tests realistic, production-level complexity
  - Includes loops, branches, and complex control flow
  - Multiple compilation passes showing optimization progression

Additional test data can be added to `ion-examples/` directory.

## CI Integration

To run benchmarks in CI without saving results:

```bash
cargo bench --no-fail-fast -- --test
```

This runs benchmarks once each (no statistical analysis) to verify they don't crash.

## Optimization Tips

When optimizing based on benchmarks:

1. **Profile first**: Use flamegraphs to identify hotspots
2. **Measure incrementally**: Run benchmarks after each change
3. **Check for regressions**: Compare all benchmarks, not just the one you're optimizing
4. **Consider memory**: Use `heaptrack` or `valgrind` to check allocations
5. **Validate correctness**: Run tests after optimizations to ensure correctness

## Known Performance Characteristics

- **Layout algorithms**: O(n²) in worst case, typically O(n log n)
- **Edge straightening**: Linear passes, O(n) per pass
- **SVG generation**: Linear in number of blocks and edges
- **Memory usage**: Proportional to number of blocks and edges

## Comparing with TypeScript

The project includes tools for direct performance comparison with the TypeScript implementation.

### Quick Comparison

Run the comparison script to benchmark both implementations side-by-side:

```bash
./compare-performance.sh ion-examples/mega-complex.json 100
```

This will:
1. Build the Rust binary in release mode
2. Run Rust benchmarks
3. Run TypeScript benchmarks
4. Display a comparison table with speedup factors

Example output:

```
Function        Rust (ms)       TypeScript (ms) Speedup
--------        --------        -------------- -------
Function 0      2.345           18.234         7.77x
Function 1      3.123           21.456         6.87x
...
Average         2.734           19.845         7.26x
```

### Manual TypeScript Benchmarking

To run TypeScript benchmarks independently:

```bash
cd /Users/jimmyhmiller/Documents/Code/open-source/iongraph2
node bench-iongraph.mjs path/to/input.json 100
```

### Expected Performance Differences

- **Rust should be 5-10x faster** for CPU-bound layout operations
- **Memory usage**: Rust typically uses less memory (no GC overhead)
- **Startup time**: TypeScript may be slower due to JIT compilation
- **Peak performance**: Rust maintains consistent performance, TypeScript may vary during warmup

### Profiling with samply

[samply](https://github.com/mstange/samply) is a command-line profiling tool for macOS and Linux.

#### Install samply

```bash
cargo install samply
```

#### Profile the Rust Implementation

```bash
# Build release binary
cargo build --release --bin profile-render

# Profile with samply
samply record target/release/profile-render ion-examples/mega-complex.json 1000

# Profile specific function
samply record target/release/profile-render ion-examples/mega-complex.json 1000 0
```

This will:
1. Run the workload 1000 times (for better profiling data)
2. Collect sampling data
3. Open an interactive flamegraph in your browser

#### Profiling Tips

- **Use high iteration counts** (1000+) for clearer profiles
- **Profile specific functions** if you know which one is slow
- **Look for hot paths** in the flamegraph (wider sections)
- **Compare before/after** optimizations by saving profiles

#### Understanding the Flamegraph

- **Width**: Time spent in function (wider = more time)
- **Height**: Call stack depth
- **Color**: Different colors for different modules/types
- **Click**: Zoom into specific call paths
- **Search**: Find specific function names

Example profiling workflow:

```bash
# Profile baseline
samply record target/release/profile-render ion-examples/mega-complex.json 1000
# Save the flamegraph URL

# Make optimizations...
cargo build --release

# Profile optimized version
samply record target/release/profile-render ion-examples/mega-complex.json 1000
# Compare the flamegraphs
```

### Other Profiling Tools

#### perf (Linux only)

```bash
perf record --call-graph dwarf target/release/profile-render ion-examples/mega-complex.json 1000
perf report
```

#### Instruments (macOS)

```bash
instruments -t "Time Profiler" target/release/profile-render ion-examples/mega-complex.json 1000
```

#### Valgrind (Linux)

For memory profiling:

```bash
valgrind --tool=massif target/release/profile-render ion-examples/mega-complex.json 100
massif-visualizer massif.out.*
```
