#!/bin/bash
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  Cross-Language JavaScript Parser Benchmarks"
echo "  Our Parser vs Rust (OXC, SWC) vs JavaScript (Babel, Acorn)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Create results directory
RESULTS_DIR="benchmark-results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "═══════════════════════════════════════════════════════════════"
echo "  Running Cross-Language Real-World Benchmarks (Rust)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "🏃 Running Rust benchmarks (OXC, SWC)..."
cd benchmarks/rust
cargo build --release 2>&1 | grep -v "Compiling\|Finished" || true
cargo run --release --bin benchmark-real-world 2>&1 | tee "../../$RESULTS_DIR/rust_realworld_${TIMESTAMP}.txt"
cd ../..
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  Running Cross-Language Real-World Benchmarks (JavaScript)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "🏃 Running JavaScript benchmarks (Babel, Acorn)..."
cd benchmarks/javascript
node benchmark-real-world.js 2>&1 | tee "../../$RESULTS_DIR/js_realworld_${TIMESTAMP}.txt"
cd ../..
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  Benchmark Results Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to:"
echo "  📄 $RESULTS_DIR/rust_realworld_${TIMESTAMP}.txt"
echo "  📄 $RESULTS_DIR/js_realworld_${TIMESTAMP}.txt"
echo ""

echo "Rust Real-World Benchmark Summary:"
echo "───────────────────────────────────────────────────────────────"
tail -30 "$RESULTS_DIR/rust_realworld_${TIMESTAMP}.txt" || echo "Results not found"
echo ""

echo "JavaScript Real-World Benchmark Summary:"
echo "───────────────────────────────────────────────────────────────"
tail -30 "$RESULTS_DIR/js_realworld_${TIMESTAMP}.txt" || echo "Results not found"
echo ""

echo "✅ Cross-language benchmarks complete!"
echo ""
echo "Full results available in: $RESULTS_DIR/"
