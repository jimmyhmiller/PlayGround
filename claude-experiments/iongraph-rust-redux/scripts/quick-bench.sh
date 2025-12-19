#!/bin/bash

# Quick benchmark script - runs a fast performance check
# Usage: ./quick-bench.sh [iterations]

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

ITERATIONS="${1:-50}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}IonGraph Quick Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Iterations: ${GREEN}$ITERATIONS${NC}"
echo ""

# Build release binary
echo -e "${GREEN}Building release binary...${NC}"
cargo build --release --bin profile-render 2>&1 | grep -v "Compiling\|Finished" || true

echo ""
echo -e "${GREEN}Running benchmark...${NC}"
echo ""

# Run benchmark (suppress debug output)
target/release/profile-render ion-examples/mega-complex.json "$ITERATIONS" 0 2>/dev/null

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo ""
echo "For detailed statistical benchmarks:"
echo "  cargo bench"
echo ""
echo "For profiling with flamegraphs:"
echo "  samply record target/release/profile-render ion-examples/mega-complex.json 1000"
echo ""
echo "For comparison with TypeScript:"
echo "  ./compare-performance.sh ion-examples/mega-complex.json 100"
echo -e "${BLUE}========================================${NC}"
