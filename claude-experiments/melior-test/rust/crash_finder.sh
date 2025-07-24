#!/bin/bash

# Script to systematically find the minimal test set that causes crashes
# Usage: ./crash_finder.sh

set -euo pipefail

RUST_DIR="/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/melior-test/rust"
TEST_FILE="crash_reproduction_tests"
ITERATIONS=100
CRASH_COUNT=0
SUCCESS_COUNT=0

cd "$RUST_DIR"

echo "=== Crash Reproduction Analysis ==="
echo "Testing file: $TEST_FILE"
echo "Iterations: $ITERATIONS"
echo "Date: $(date)"
echo

log_result() {
    local iteration=$1
    local result=$2
    local exit_code=$3
    echo "[$iteration/$ITERATIONS] $result (exit code: $exit_code)"
}

run_test_iterations() {
    echo "Running $ITERATIONS iterations of $TEST_FILE..."
    
    for i in $(seq 1 $ITERATIONS); do
        if cargo test --test "$TEST_FILE" >/dev/null 2>&1; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            log_result "$i" "PASS" "0"
        else
            CRASH_COUNT=$((CRASH_COUNT + 1))
            log_result "$i" "CRASH" "$?"
        fi
    done
    
    echo
    echo "=== Results ==="
    echo "Total iterations: $ITERATIONS"
    echo "Successful runs: $SUCCESS_COUNT"
    echo "Crashes: $CRASH_COUNT"
    echo "Crash rate: $(echo "scale=2; $CRASH_COUNT * 100 / $ITERATIONS" | bc -l)%"
    echo
    
    return $CRASH_COUNT
}

# Initial test run
echo "Phase 1: Testing full copy with all tests..."
if run_test_iterations; then
    echo "No crashes detected in $ITERATIONS runs. The issue may not be reproducible with this approach."
    exit 0
else
    echo "Crashes detected! Proceeding to binary search..."
fi

echo
echo "Phase 2: Starting binary search to find minimal crash case..."

# If we get here, we detected crashes and need to start removing tests
# We'll create progressively smaller test files

# Extract test modules for binary search
echo "Extracting test modules..."
grep -n "^mod " "tests/${TEST_FILE}.rs" | head -10

echo
echo "Manual binary search required. The test file is ready for systematic reduction."
echo "Next steps:"
echo "1. Edit tests/${TEST_FILE}.rs to comment out half the test modules"
echo "2. Run this script again"
echo "3. If crashes stop, uncomment and try removing different modules"
echo "4. If crashes continue, remove more modules"
echo "5. Repeat until you find the minimal set"