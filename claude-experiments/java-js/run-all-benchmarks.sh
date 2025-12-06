#!/bin/bash

# Comprehensive Cross-Language JavaScript Parser Benchmarks
# Runs benchmarks for parsers in Java, JavaScript, and Rust

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Cross-Language JavaScript Parser Benchmarks"
echo "════════════════════════════════════════════════════════════"
echo ""

# Create results directory
mkdir -p benchmark-results

# =============================================================================
# 1. Run Java/JVM Benchmarks
# =============================================================================
echo -e "${BLUE}[1/3] Running Java/JVM benchmarks...${NC}"
echo "Building Java benchmark JAR..."
mvn clean package -DskipTests -q

echo "Running Java benchmarks (this may take a few minutes)..."
java --enable-preview -jar target/benchmarks.jar ComparativeParserBenchmark \
    -f 1 -wi 3 -i 5 -rf json -rff benchmark-results/java-results.json 2>&1 | \
    grep -E "(Benchmark|Score|ourParser|rhino|nashorn|graalJS)" || true

echo -e "${GREEN}✓ Java benchmarks complete${NC}"
echo ""

# =============================================================================
# 2. Run JavaScript Benchmarks
# =============================================================================
echo -e "${BLUE}[2/3] Running JavaScript benchmarks...${NC}"

if [ ! -d "benchmarks/javascript/node_modules" ]; then
    echo "Installing Node.js dependencies..."
    cd benchmarks/javascript && npm install --silent && cd ../..
fi

echo "Running JavaScript benchmarks..."
cd benchmarks/javascript
node benchmark.js > ../../benchmark-results/javascript-results.txt 2>&1
cd ../..

echo -e "${GREEN}✓ JavaScript benchmarks complete${NC}"
echo ""

# =============================================================================
# 3. Run Rust Benchmarks
# =============================================================================
echo -e "${BLUE}[3/3] Running Rust benchmarks...${NC}"

if [ ! -f "benchmarks/rust/target/release/benchmark" ]; then
    echo "Building Rust benchmarks (first run may take a while)..."
    cd benchmarks/rust && cargo build --release --quiet && cd ../..
fi

echo "Running Rust benchmarks..."
cd benchmarks/rust
cargo run --release --quiet > ../../benchmark-results/rust-results.txt 2>&1
cd ../..

echo -e "${GREEN}✓ Rust benchmarks complete${NC}"
echo ""

# =============================================================================
# Generate Comparison Report
# =============================================================================
echo "════════════════════════════════════════════════════════════"
echo "  Benchmark Results Summary"
echo "════════════════════════════════════════════════════════════"
echo ""

# Extract key results and display
echo "Results saved to benchmark-results/"
echo ""
echo -e "${YELLOW}Quick Summary:${NC}"
echo ""

# Java results (extract from JMH JSON output)
echo "JVM Parsers (from JMH):"
if [ -f "benchmark-results/java-results.json" ]; then
    # This will need a JSON parser for proper extraction
    echo "  (See benchmark-results/java-results.json for detailed results)"
fi
echo ""

# JavaScript results
echo "JavaScript Parsers:"
grep "Small Function:" benchmark-results/javascript-results.txt -A 1 | tail -1 || echo "  See benchmark-results/javascript-results.txt"
echo ""

# Rust results
echo "Rust Parsers:"
grep -A 10 "Small Function" benchmark-results/rust-results.txt | grep "µs" || echo "  See benchmark-results/rust-results.txt"
echo ""

echo "════════════════════════════════════════════════════════════"
echo -e "${GREEN}All benchmarks complete!${NC}"
echo ""
echo "Detailed results:"
echo "  - Java/JVM:    benchmark-results/java-results.json"
echo "  - JavaScript:  benchmark-results/javascript-results.txt"
echo "  - Rust:        benchmark-results/rust-results.txt"
echo "════════════════════════════════════════════════════════════"
