#!/bin/bash

# Quick Cross-Language Benchmark (fast version for testing)
# For full benchmarks, use ./run-all-benchmarks.sh

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Quick Cross-Language JavaScript Parser Benchmarks"
echo "════════════════════════════════════════════════════════════"
echo ""

mkdir -p benchmark-results

# Quick Java benchmarks
echo "[1/3] Running Java benchmarks (quick mode)..."
if [ ! -f "target/benchmarks.jar" ]; then
    mvn clean package -DskipTests -q
fi
java --enable-preview -jar target/benchmarks.jar ComparativeParserBenchmark \
    -f 0 -wi 1 -i 1 -rf json -rff benchmark-results/java-results.json 2>&1 | \
    grep -E "(Benchmark|Result)" | head -20 || true

# JavaScript benchmarks
echo ""
echo "[2/3] Running JavaScript benchmarks..."
if [ ! -d "benchmarks/javascript/node_modules" ]; then
    (cd benchmarks/javascript && npm install --silent)
fi
(cd benchmarks/javascript && node benchmark.js) > benchmark-results/javascript-results.txt 2>&1

# Rust benchmarks
echo ""
echo "[3/3] Running Rust benchmarks..."
if [ ! -f "benchmarks/rust/target/release/benchmark" ]; then
    (cd benchmarks/rust && cargo build --release --quiet)
fi
(cd benchmarks/rust && cargo run --release --quiet) > benchmark-results/rust-results.txt 2>&1

# Analyze results
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Generating comparison report..."
echo "════════════════════════════════════════════════════════════"
python3 analyze-benchmarks.py

echo ""
echo "Done! For more rigorous benchmarks, run: ./run-all-benchmarks.sh"
