#!/bin/bash
# Extract generated MLIR from mlir_lisp output
# Usage: ./mlisp_to_mlir.sh input.mlisp output.mlir

set -e

INPUT="$1"
OUTPUT="$2"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 input.mlisp output.mlir"
    exit 1
fi

# Run mlir_lisp and extract MLIR between the markers
./zig-out/bin/mlir_lisp "$INPUT" 2>&1 | \
  awk '/^Generated MLIR:$/,/^----------------------------------------$/ {
    if ($0 != "Generated MLIR:" && $0 != "----------------------------------------") print
  }' > "$OUTPUT"

# Check if output file has content
if [ ! -s "$OUTPUT" ]; then
    echo "Error: Failed to extract MLIR from output"
    exit 1
fi

echo "Successfully converted $INPUT to $OUTPUT"
