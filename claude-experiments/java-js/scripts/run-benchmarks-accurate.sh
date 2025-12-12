#!/bin/bash
# Accurate benchmarks for publishing (uses -f 3, slower but accurate)
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  Accurate Benchmarks (Production Mode)"
echo "  Using forking for accurate results (will take longer)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

RESULTS_DIR="benchmark-results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📦 Building..."
mvn clean package -q -DskipTests

echo "🏃 Running Java benchmarks (accurate mode with 3 forks)..."
echo "This will take ~10-15 minutes..."
java --enable-preview -jar target/benchmarks.jar ComparativeParserBenchmark -f 3 -wi 3 -i 5 \
    2>&1 | tee "$RESULTS_DIR/comparative_accurate_${TIMESTAMP}.txt"
    
java --enable-preview -jar target/benchmarks.jar RealWorldEs5JavaBenchmark -f 3 -wi 3 -i 5 \
    2>&1 | tee "$RESULTS_DIR/realworld_es5_accurate_${TIMESTAMP}.txt"

echo ""
echo "🏃 Running cross-language benchmarks..."
cd benchmarks/rust
cargo build --release 2>&1 | grep -v "Compiling\|Finished" || true
cargo run --release --bin benchmark-real-world 2>&1 | tee "../../$RESULTS_DIR/rust_realworld_${TIMESTAMP}.txt"
cd ../..

cd benchmarks/javascript
node benchmark-real-world.js 2>&1 | tee "../../$RESULTS_DIR/js_realworld_${TIMESTAMP}.txt"
cd ../..

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Results Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Java Comparative:"
grep -A 20 "^Benchmark" "$RESULTS_DIR/comparative_accurate_${TIMESTAMP}.txt" | tail -20 || echo "Not found"
echo ""
echo "Java Real-World ES5:"
grep -A 20 "^Benchmark" "$RESULTS_DIR/realworld_es5_accurate_${TIMESTAMP}.txt" | tail -20 || echo "Not found"
echo ""
echo "✅ All accurate benchmarks complete! Results in: $RESULTS_DIR/"
