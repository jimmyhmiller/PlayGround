#!/bin/bash
# Test runner for lispier examples
# Runs each .lisp file in examples/ and checks against expected output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build first
echo "Building..."
zig build
echo ""

PASSED=0
FAILED=0
TOTAL=0

for file in examples/*.lisp; do
    TOTAL=$((TOTAL + 1))

    # Extract expected output from comment: "; Expected output: <value>"
    EXPECTED=$(grep -i "Expected output:" "$file" | sed 's/.*Expected output:[[:space:]]*//')

    if [ -z "$EXPECTED" ]; then
        echo "SKIP: $file (no expected output specified)"
        continue
    fi

    # Run the file and capture output
    OUTPUT=$(zig build run -- "$file" 2>&1)

    # Extract the result line
    RESULT=$(echo "$OUTPUT" | grep "^Result:" | sed 's/Result:[[:space:]]*//')

    if [ -z "$RESULT" ]; then
        echo "FAIL: $file"
        echo "  Expected: $EXPECTED"
        echo "  Got: (no result)"
        echo "  Full output:"
        echo "$OUTPUT" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        continue
    fi

    # Compare (trim whitespace)
    EXPECTED_TRIMMED=$(echo "$EXPECTED" | xargs)
    RESULT_TRIMMED=$(echo "$RESULT" | xargs)

    if [ "$EXPECTED_TRIMMED" = "$RESULT_TRIMMED" ]; then
        echo "PASS: $file -> $RESULT_TRIMMED"
        PASSED=$((PASSED + 1))
    else
        echo "FAIL: $file"
        echo "  Expected: $EXPECTED_TRIMMED"
        echo "  Got: $RESULT_TRIMMED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "================================"
echo "Results: $PASSED passed, $FAILED failed, $TOTAL total"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
