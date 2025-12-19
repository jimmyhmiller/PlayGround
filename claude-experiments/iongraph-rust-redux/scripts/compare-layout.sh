#!/bin/bash
# Compare layout between Ion JSON and DOT conversion
#
# Usage:
#   ./compare-layout.sh <ion-json> <function-index> [pass-index]
#
# Example:
#   ./compare-layout.sh ion-examples/mega-complex.json 5 0

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ion-json> <function-index> [pass-index]"
    echo ""
    echo "Example:"
    echo "  $0 ion-examples/mega-complex.json 5 0"
    exit 1
fi

JSON_FILE="$1"
FUNC_IDX="$2"
PASS_IDX="${3:-0}"

# Output directory
OUT_DIR="compare-output"
mkdir -p "$OUT_DIR"

# Build if needed
echo "Building..."
cargo build --release --bin iongraph --bin ion-to-dot --bin dotgraph 2>/dev/null

echo ""
echo "=== Comparing layout for function $FUNC_IDX, pass $PASS_IDX ==="
echo ""

# Step 1: Generate SVG directly from Ion JSON
echo "1. Rendering Ion JSON directly..."
./target/release/iongraph --ion "$JSON_FILE" "$FUNC_IDX" "$PASS_IDX" "$OUT_DIR/ion-direct.svg"

# Step 2: Convert Ion JSON to DOT
echo ""
echo "2. Converting Ion JSON to DOT..."
./target/release/ion-to-dot "$JSON_FILE" "$FUNC_IDX" "$PASS_IDX" "$OUT_DIR/converted.dot"

# Step 3: Render DOT file
echo ""
echo "3. Rendering DOT file..."
./target/release/dotgraph "$OUT_DIR/converted.dot" "$OUT_DIR/dot-rendered.svg"

# Step 4: Compare file sizes and dimensions
echo ""
echo "=== Comparison Results ==="
echo ""

ION_SIZE=$(wc -c < "$OUT_DIR/ion-direct.svg" | tr -d ' ')
DOT_SIZE=$(wc -c < "$OUT_DIR/dot-rendered.svg" | tr -d ' ')

ION_DIMS=$(head -1 "$OUT_DIR/ion-direct.svg" | grep -o 'width="[0-9]*" height="[0-9]*"' || echo "unknown")
DOT_DIMS=$(head -1 "$OUT_DIR/dot-rendered.svg" | grep -o 'width="[0-9]*" height="[0-9]*"' || echo "unknown")

echo "Ion direct:    $ION_SIZE bytes, $ION_DIMS"
echo "DOT rendered:  $DOT_SIZE bytes, $DOT_DIMS"

# Step 5: Count blocks and edges in both
ION_BLOCKS=$(grep -c 'class="ig-block"' "$OUT_DIR/ion-direct.svg" 2>/dev/null || grep -c '<g transform' "$OUT_DIR/ion-direct.svg" | head -1)
DOT_BLOCKS=$(grep -c '<g transform' "$OUT_DIR/dot-rendered.svg" | head -1)

echo ""
echo "Output files:"
echo "  $OUT_DIR/ion-direct.svg    (from iongraph)"
echo "  $OUT_DIR/converted.dot     (DOT source)"
echo "  $OUT_DIR/dot-rendered.svg  (from dotgraph)"

echo ""
echo "Open both SVGs to visually compare the layout structure."
