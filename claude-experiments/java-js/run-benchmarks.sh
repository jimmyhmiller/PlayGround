#!/bin/bash

# JMH Benchmark Runner Script
# This script builds and runs JMH benchmarks for the JavaScript parser

set -e

echo "Building benchmark JAR..."
mvn clean package -DskipTests -q

echo ""
echo "Running benchmarks..."
echo "Usage: ./run-benchmarks.sh [benchmark-pattern] [jmh-options]"
echo "Example: ./run-benchmarks.sh ComparativeParserBenchmark -f 1 -wi 3 -i 5"
echo ""

# Default to running all benchmarks with minimal iterations for quick testing
BENCHMARK_PATTERN="${1:-.*}"
JMH_ARGS="${@:2}"

# If no JMH args provided, use quick test defaults
if [ -z "$JMH_ARGS" ]; then
    JMH_ARGS="-f 0 -wi 1 -i 1"
fi

java --enable-preview -jar target/benchmarks.jar "$BENCHMARK_PATTERN" $JMH_ARGS
