#!/bin/bash

# Download real-world JavaScript libraries for benchmarking
# These are commonly used benchmark targets for JavaScript parsers

set -e

LIBS_DIR="benchmarks/real-world-libs"
mkdir -p "$LIBS_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  Downloading Real-World JavaScript Libraries"
echo "════════════════════════════════════════════════════════════"
echo ""

# TypeScript Compiler (very large, ~3.8 MB)
echo "[1/6] Downloading TypeScript compiler..."
curl -s "https://cdn.jsdelivr.net/npm/typescript@5.3.3/lib/typescript.js" \
    -o "$LIBS_DIR/typescript.js"
echo "✓ typescript.js ($(wc -c < "$LIBS_DIR/typescript.js" | xargs) bytes)"

# React Production Bundle
echo "[2/6] Downloading React production bundle..."
curl -s "https://unpkg.com/react@18.2.0/umd/react.production.min.js" \
    -o "$LIBS_DIR/react.production.min.js"
echo "✓ react.production.min.js ($(wc -c < "$LIBS_DIR/react.production.min.js" | xargs) bytes)"

# React DOM Production Bundle
echo "[3/6] Downloading React DOM production bundle..."
curl -s "https://unpkg.com/react-dom@18.2.0/umd/react-dom.production.min.js" \
    -o "$LIBS_DIR/react-dom.production.min.js"
echo "✓ react-dom.production.min.js ($(wc -c < "$LIBS_DIR/react-dom.production.min.js" | xargs) bytes)"

# Vue 3 Production Bundle
echo "[4/6] Downloading Vue 3 production bundle..."
curl -s "https://unpkg.com/vue@3.3.13/dist/vue.global.prod.js" \
    -o "$LIBS_DIR/vue.global.prod.js"
echo "✓ vue.global.prod.js ($(wc -c < "$LIBS_DIR/vue.global.prod.js" | xargs) bytes)"

# Lodash
echo "[5/6] Downloading Lodash..."
curl -s "https://cdn.jsdelivr.net/npm/lodash@4.17.21/lodash.js" \
    -o "$LIBS_DIR/lodash.js"
echo "✓ lodash.js ($(wc -c < "$LIBS_DIR/lodash.js" | xargs) bytes)"

# Three.js (3D library, very large)
echo "[6/6] Downloading Three.js..."
curl -s "https://cdn.jsdelivr.net/npm/three@0.159.0/build/three.js" \
    -o "$LIBS_DIR/three.js"
echo "✓ three.js ($(wc -c < "$LIBS_DIR/three.js" | xargs) bytes)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Downloaded 6 real-world libraries:"
echo ""
ls -lh "$LIBS_DIR" | awk 'NR>1 {printf "  %-30s %8s\n", $9, $5}'
echo ""
echo "Total size: $(du -sh "$LIBS_DIR" | cut -f1)"
echo ""
echo "These files will be used for real-world parsing benchmarks."
echo "════════════════════════════════════════════════════════════"
