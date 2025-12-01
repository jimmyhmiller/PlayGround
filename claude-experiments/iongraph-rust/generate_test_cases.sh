#!/bin/bash

# Script to generate TypeScript SVG test cases for comparison testing

TS_DIR="/Users/jimmyhmiller/Documents/Code/open-source/iongraph2"
EXAMPLES_DIR="$TS_DIR/examples"
FIXTURES_DIR="./tests/fixtures"
SCRIPT="$TS_DIR/generate-svg-function.mjs"

# Create fixtures directory if it doesn't exist
mkdir -p "$FIXTURES_DIR"

echo "Generating TypeScript SVG test fixtures..."
echo

# Define test cases: filename, function_index, pass_index, description
declare -a test_cases=(
  "complex-sample.json:0:0:fibonacci"
  "medium-graph.json:0:0:medium-graph"
  "mega-complex.json:5:0:mega-complex-func5-pass0"
  "test-30-fixed.json:0:0:test-30"
  "test-50-final.json:0:0:test-50"
)

generated_count=0

for test_case in "${test_cases[@]}"; do
  IFS=':' read -r json_file func_idx pass_idx desc <<< "$test_case"

  input_path="$EXAMPLES_DIR/$json_file"
  output_svg="$FIXTURES_DIR/ts-$desc.svg"

  if [ ! -f "$input_path" ]; then
    echo "⚠️  Skipping $json_file (file not found)"
    continue
  fi

  echo "Generating: $desc"
  echo "  Input: $json_file (function $func_idx, pass $pass_idx)"

  if node "$SCRIPT" "$input_path" "$func_idx" "$pass_idx" "$output_svg" 2>&1 | grep -q "✓ SVG generated"; then
    echo "  ✓ Generated: $output_svg"

    # Also copy the JSON file for Rust tests
    cp "$input_path" "$FIXTURES_DIR/$desc.json"
    echo "  ✓ Copied JSON: $FIXTURES_DIR/$desc.json"

    ((generated_count++))
  else
    echo "  ✗ Failed to generate $desc"
  fi

  echo
done

echo "========================================="
echo "Generated $generated_count test fixtures"
echo "Location: $FIXTURES_DIR"
echo "========================================="
