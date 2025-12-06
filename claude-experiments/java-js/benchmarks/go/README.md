# Go JavaScript Parser Benchmarks

## Prerequisites

You need Go installed to run these benchmarks.

Install Go from: https://go.dev/dl/

## Running

```bash
# Download dependencies
go mod download

# Build
go build

# Run benchmarks
./javascript-parser-benchmarks
```

## What's Benchmarked

- **esbuild** - Extremely fast bundler and minifier written in Go
  - Uses esbuild's parser via the Go API
  - Note: esbuild includes parsing + transformation, not just parsing

## Note

If Go is not installed, this benchmark will be skipped in the cross-language comparison.
