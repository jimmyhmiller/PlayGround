#!/usr/bin/env bash
# Convert all MLIR example files to generic format

set -e

EXAMPLES_DIR="test_data/examples"

echo "Converting all .mlir files in $EXAMPLES_DIR to generic format..."

for file in "$EXAMPLES_DIR"/*.mlir; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Converting: $filename"

        # Use mlir-opt to convert to generic format
        # --mlir-print-op-generic outputs operations in generic form
        if mlir-opt --mlir-print-op-generic "$file" -o "$file.tmp" 2>/dev/null; then
            mv "$file.tmp" "$file"
            echo "  ✓ Converted successfully"
        else
            echo "  ✗ Failed to convert (may have errors or unsupported features)"
            rm -f "$file.tmp"
        fi
    fi
done

echo ""
echo "Conversion complete!"
