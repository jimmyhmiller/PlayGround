#!/bin/bash
# Quick benchmarks for development (uses -f 0, faster but less accurate)
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  Quick Benchmarks (Development Mode)"
echo "  Warning: Results may be less accurate due to no forking"
echo "═══════════════════════════════════════════════════════════════"
echo ""

RESULTS_DIR="benchmark-results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📦 Building..."
mvn clean package -q -DskipTests

echo "🏃 Running Java benchmarks (quick mode)..."
java --enable-preview -jar target/benchmarks.jar ComparativeParserBenchmark -f 0 -wi 2 -i 3 \
    2>&1 | tee "$RESULTS_DIR/comparative_quick_${TIMESTAMP}.txt"
    
java --enable-preview -jar target/benchmarks.jar RealWorldEs5JavaBenchmark -f 0 -wi 2 -i 3 \
    2>&1 | tee "$RESULTS_DIR/realworld_es5_quick_${TIMESTAMP}.txt"

echo ""
echo "✅ Quick benchmarks complete! Results in: $RESULTS_DIR/"
