#!/bin/bash

# Performance comparison script between Rust and TypeScript implementations
# Usage: ./compare-performance.sh <input.json>

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.json> [iterations]"
    echo ""
    echo "Example: $0 ion-examples/mega-complex.json 100"
    exit 1
fi

INPUT_FILE="$1"
ITERATIONS="${2:-100}"
TYPESCRIPT_DIR="/Users/jimmyhmiller/Documents/Code/open-source/iongraph2"

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}IonGraph Performance Comparison${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Input file: ${YELLOW}$INPUT_FILE${NC}"
echo -e "Iterations: ${YELLOW}$ITERATIONS${NC}"
echo ""

# Build Rust release binary
echo -e "${GREEN}Building Rust binary (release mode)...${NC}"
cargo build --release --bin profile-render

echo ""
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}Running Rust Implementation${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
echo ""

RUST_OUTPUT=$(target/release/profile-render "$INPUT_FILE" "$ITERATIONS" 2>&1)
echo "$RUST_OUTPUT"

# Extract timing data from Rust output
RUST_TIMES=$(echo "$RUST_OUTPUT" | grep "Average:" | awk '{print $2}')

echo ""
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}Running TypeScript Implementation${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
echo ""

if [ ! -f "$TYPESCRIPT_DIR/bench-iongraph.mjs" ]; then
    echo -e "${YELLOW}Warning: TypeScript benchmark script not found${NC}"
    echo -e "Creating benchmark script at $TYPESCRIPT_DIR/bench-iongraph.mjs"
fi

# Run TypeScript benchmarks
cd "$TYPESCRIPT_DIR"
TS_OUTPUT=$(node bench-iongraph.mjs "../claude-experiments/iongraph-rust-redux/$INPUT_FILE" "$ITERATIONS" 2>&1)
cd - > /dev/null

echo "$TS_OUTPUT"

# Extract timing data from TypeScript output
TS_TIMES=$(echo "$TS_OUTPUT" | grep "Mean:" | awk '{print $2}')

# Generate comparison table
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Performance Comparison Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

printf "%-15s %-15s %-15s %-15s\n" "Function" "Rust (ms)" "TypeScript (ms)" "Speedup"
printf "%-15s %-15s %-15s %-15s\n" "--------" "--------" "--------------" "-------"

RUST_ARRAY=($RUST_TIMES)
TS_ARRAY=($TS_TIMES)

TOTAL_RUST=0
TOTAL_TS=0
COUNT=0

for i in "${!RUST_ARRAY[@]}"; do
    RUST_TIME=${RUST_ARRAY[$i]}
    TS_TIME=${TS_ARRAY[$i]}

    if [ -n "$RUST_TIME" ] && [ -n "$TS_TIME" ]; then
        SPEEDUP=$(echo "scale=2; $TS_TIME / $RUST_TIME" | bc)
        printf "%-15s %-15s %-15s ${GREEN}%-15s${NC}\n" "Function $i" "$RUST_TIME" "$TS_TIME" "${SPEEDUP}x"

        TOTAL_RUST=$(echo "$TOTAL_RUST + $RUST_TIME" | bc)
        TOTAL_TS=$(echo "$TOTAL_TS + $TS_TIME" | bc)
        COUNT=$((COUNT + 1))
    fi
done

if [ $COUNT -gt 0 ]; then
    echo ""
    printf "%-15s %-15s %-15s %-15s\n" "--------" "--------" "--------------" "-------"

    AVG_RUST=$(echo "scale=3; $TOTAL_RUST / $COUNT" | bc)
    AVG_TS=$(echo "scale=3; $TOTAL_TS / $COUNT" | bc)
    AVG_SPEEDUP=$(echo "scale=2; $AVG_TS / $AVG_RUST" | bc)

    printf "%-15s ${GREEN}%-15s${NC} %-15s ${GREEN}%-15s${NC}\n" "Average" "$AVG_RUST" "$AVG_TS" "${AVG_SPEEDUP}x"
    printf "%-15s ${GREEN}%-15s${NC} %-15s\n" "Total" "$TOTAL_RUST" "$TOTAL_TS"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Comparison complete!${NC}"
echo -e "${BLUE}========================================${NC}"
