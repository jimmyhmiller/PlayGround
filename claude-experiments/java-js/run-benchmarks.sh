#!/bin/bash
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  JavaScript Parser Benchmarks - Full Suite"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Create results directory
RESULTS_DIR="benchmark-results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "═══════════════════════════════════════════════════════════════"
echo "  1. Building Java Project"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📦 Building project..."
mvn clean package -q -DskipTests
echo "✅ Build complete"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  2. Running Java Benchmarks"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Run small comparative benchmarks
echo "🏃 Running ComparativeParserBenchmark (small synthetic tests)..."
java --enable-preview -jar target/benchmarks.jar ComparativeParserBenchmark \
    -f 3 -wi 3 -i 5 \
    2>&1 | tee "$RESULTS_DIR/comparative_${TIMESTAMP}.txt"
echo ""

# Run real-world ES5 benchmarks with Rhino
echo "🏃 Running RealWorldEs5JavaBenchmark (Lodash, React, React-DOM with Rhino)..."
java --enable-preview -jar target/benchmarks.jar RealWorldEs5JavaBenchmark \
    -f 3 -wi 3 -i 5 \
    2>&1 | tee "$RESULTS_DIR/realworld_es5_${TIMESTAMP}.txt"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  3. Running Cross-Language Real-World Benchmarks (Rust)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "🏃 Running Rust benchmarks (OXC, SWC)..."
cd benchmarks/rust
cargo build --release 2>&1 | grep -v "Compiling\|Finished" || true
cargo run --release --bin benchmark-real-world 2>&1 | tee "../../$RESULTS_DIR/rust_realworld_${TIMESTAMP}.txt"
cd ../..
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  4. Running Cross-Language Real-World Benchmarks (JavaScript)"
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
echo "  📄 $RESULTS_DIR/comparative_${TIMESTAMP}.txt"
echo "  📄 $RESULTS_DIR/realworld_es5_${TIMESTAMP}.txt"
echo "  📄 $RESULTS_DIR/rust_realworld_${TIMESTAMP}.txt"
echo "  📄 $RESULTS_DIR/js_realworld_${TIMESTAMP}.txt"
echo ""

echo "Java Comparative Benchmark Results:"
echo "───────────────────────────────────────────────────────────────"
grep -A 20 "^Benchmark" "$RESULTS_DIR/comparative_${TIMESTAMP}.txt" | tail -20 || echo "Results not found"
echo ""

echo "Java Real-World ES5 Benchmark Results:"
echo "───────────────────────────────────────────────────────────────"
grep -A 20 "^Benchmark" "$RESULTS_DIR/realworld_es5_${TIMESTAMP}.txt" | tail -20 || echo "Results not found"
echo ""

echo "Rust Real-World Benchmark Summary:"
echo "───────────────────────────────────────────────────────────────"
tail -30 "$RESULTS_DIR/rust_realworld_${TIMESTAMP}.txt" || echo "Results not found"
echo ""

echo "JavaScript Real-World Benchmark Summary:"
echo "───────────────────────────────────────────────────────────────"
tail -30 "$RESULTS_DIR/js_realworld_${TIMESTAMP}.txt" || echo "Results not found"
echo ""

echo "✅ All benchmarks complete!"
echo ""
echo "Full results available in: $RESULTS_DIR/"
