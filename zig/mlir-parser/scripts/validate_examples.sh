#!/bin/bash

# Script to validate all MLIR examples with mlir-opt

examples_dir="/Users/jimmyhmiller/Documents/Code/PlayGround/zig/mlir-parser/test_data/examples"

echo "Validating MLIR examples..."
echo "======================================"

valid_count=0
invalid_count=0

for file in "$examples_dir"/*.mlir; do
    filename=$(basename "$file")
    if mlir-opt --allow-unregistered-dialect --mlir-print-op-generic "$file" > /dev/null 2>&1; then
        echo "✓ $filename"
        ((valid_count++))
    else
        echo "✗ $filename"
        ((invalid_count++))
        mlir-opt --allow-unregistered-dialect --mlir-print-op-generic "$file" 2>&1 | grep -A2 "error:" | head -3
        echo ""
    fi
done

echo "======================================"
echo "Valid: $valid_count"
echo "Invalid: $invalid_count"
