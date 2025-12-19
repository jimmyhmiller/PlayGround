# Benchmarking Quick Start Guide

## 1. Run Criterion Benchmarks

Run comprehensive statistical benchmarks:

```bash
# Run all benchmarks (takes 5-10 minutes)
cargo bench

# Run specific benchmark suites
cargo bench --bench layout_algorithms
cargo bench --bench rendering

# View HTML reports with graphs
open target/criterion/report/index.html
```

## 2. Compare Rust vs TypeScript Performance

Run side-by-side comparison:

```bash
./compare-performance.sh ion-examples/mega-complex.json 100
```

Example output:
```
Function        Rust (ms)       TypeScript (ms) Speedup
--------        --------        -------------- -------
Function 0      2.345           18.234         7.77x
Function 1      3.123           21.456         6.87x
Average         2.734           19.845         7.26x
```

## 3. Profile with samply

Install samply (one-time):
```bash
cargo install samply
```

Profile the Rust implementation:
```bash
# Build release binary
cargo build --release --bin profile-render

# Profile all functions (1000 iterations for good data)
samply record target/release/profile-render ion-examples/mega-complex.json 1000

# Profile specific function only
samply record target/release/profile-render ion-examples/mega-complex.json 1000 0
```

This opens an interactive flamegraph in your browser showing:
- Which functions consume the most time (wider = slower)
- Call stack relationships (height = depth)
- Click to zoom into specific areas
- Search for function names

## 4. Quick Manual Benchmark

For a quick performance check without full statistics:

```bash
# Build release
cargo build --release --bin profile-render

# Run 100 iterations
target/release/profile-render ion-examples/mega-complex.json 100

# Run specific function only (faster)
target/release/profile-render ion-examples/mega-complex.json 100 0
```

## Typical Performance

Based on mega-complex.json:
- **End-to-end rendering**: 1-5ms per function (Rust) vs 10-50ms (TypeScript)
- **Speedup**: 5-10x faster than TypeScript
- **Memory**: Lower and more predictable than TypeScript (no GC)

## Next Steps

See [BENCHMARKS.md](BENCHMARKS.md) for:
- Detailed benchmark suite documentation
- How to interpret Criterion results
- Advanced profiling techniques
- Adding new benchmarks
- Performance optimization tips
