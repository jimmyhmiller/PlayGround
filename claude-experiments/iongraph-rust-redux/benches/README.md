# Benchmark Suite

This directory contains Criterion.rs benchmarks for the IonGraph Rust port.

## Files

- **layout_algorithms.rs** - Benchmarks for individual layout algorithm stages:
  - Loop finding
  - Layering
  - Dummy node creation
  - Edge straightening
  - Verticalization
  - Joint routing

- **rendering.rs** - End-to-end benchmarks:
  - Complete layout pipeline
  - SVG rendering
  - Full end-to-end rendering
  - Complexity comparisons
  - Block building

## Running

```bash
# Run all benchmarks
cargo bench

# Run specific suite
cargo bench --bench layout_algorithms
cargo bench --bench rendering

# Run specific benchmark
cargo bench --bench layout_algorithms find_loops
```

## Results

Benchmark results are saved in `../target/criterion/` with:
- Statistical analysis
- HTML reports with graphs
- Historical comparison data

View the reports:
```bash
open ../target/criterion/report/index.html
```

## See Also

See [BENCHMARKS.md](../BENCHMARKS.md) for complete documentation on:
- Running benchmarks
- Interpreting results
- Profiling with samply
- Comparing with TypeScript
- Performance optimization tips
