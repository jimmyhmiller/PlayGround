# Go Parser Benchmarks (esbuild)

This benchmark tests esbuild's JavaScript parser against real-world libraries.

## Setup

The benchmark uses esbuild's internal `js_parser` package directly (parse-only, no code generation).
This requires cloning esbuild and building the benchmark from within its module:

```bash
# Clone esbuild (if not already present)
git clone --depth 1 https://github.com/evanw/esbuild.git esbuild

# Build the benchmark
cd esbuild && go build -o ../benchmark-go ./cmd/benchmark
```

## Running

```bash
# From benchmarks/go directory
./benchmark-go [warmup_iterations] [measurement_iterations]

# Examples
./benchmark-go           # 5 warmup, 10 measurement (default)
./benchmark-go 10 50     # 10 warmup, 50 measurement iterations
```

## Why Internal Parser?

esbuild's public API (`api.Transform`) includes both parsing AND code generation (printing).
To fairly compare with other parsers that only parse, we access the internal `js_parser.Parse()`
function directly, which returns an AST without generating output code.

## Test Libraries

- React (10.5 KB)
- Vue 3 (130 KB)
- React DOM (128.8 KB)
- Lodash (531.3 KB)
- Three.js (1.28 MB)
- TypeScript Compiler (8.8 MB)

## Note

If Go is not installed, this benchmark will be skipped in the cross-language comparison.
