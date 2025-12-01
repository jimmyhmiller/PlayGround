#!/bin/bash

# Script to generate test cases for ALL functions in mega-complex.json

TS_DIR="/Users/jimmyhmiller/Documents/Code/open-source/iongraph2"
EXAMPLES_DIR="$TS_DIR/examples"
FIXTURES_DIR="./tests/fixtures"
SCRIPT="$TS_DIR/generate-svg-function.mjs"
MEGA_COMPLEX="$EXAMPLES_DIR/mega-complex.json"

# Create fixtures directory if it doesn't exist
mkdir -p "$FIXTURES_DIR"

# Get the number of functions in mega-complex.json
FUNCTION_COUNT=$(jq '.functions | length' "$MEGA_COMPLEX")

echo "Generating TypeScript SVG test fixtures for ALL mega-complex functions..."
echo "Total functions: $FUNCTION_COUNT"
echo

generated_count=0
failed_count=0

for func_idx in $(seq 0 $((FUNCTION_COUNT - 1))); do
  # Always use pass 0 for consistency
  pass_idx=0
  desc="mega-complex-func${func_idx}-pass${pass_idx}"
  output_svg="$FIXTURES_DIR/ts-$desc.svg"

  echo "Generating: Function $func_idx / $((FUNCTION_COUNT - 1))"

  if node "$SCRIPT" "$MEGA_COMPLEX" "$func_idx" "$pass_idx" "$output_svg" 2>&1 | grep -q "✓ SVG generated"; then
    echo "  ✓ Generated: $output_svg"
    ((generated_count++))
  else
    echo "  ✗ Failed to generate func $func_idx"
    ((failed_count++))
  fi
done

echo
echo "========================================="
echo "Generated: $generated_count test fixtures"
echo "Failed: $failed_count"
echo "Location: $FIXTURES_DIR"
echo "========================================="

# Copy the mega-complex.json once (it's large, so we only need one copy)
if [ $generated_count -gt 0 ]; then
  echo "Copying mega-complex.json to fixtures..."
  cp "$MEGA_COMPLEX" "$FIXTURES_DIR/mega-complex.json"
  echo "✓ Done"
fi
