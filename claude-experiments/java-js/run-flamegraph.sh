#!/bin/bash
# Run JMH benchmarks with async-profiler and generate flamegraphs
# Requires async-profiler to be installed

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  JMH Flamegraph Profiling"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Configuration
RESULTS_DIR="benchmark-results/flamegraphs"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Detect async-profiler location
ASYNC_PROFILER=""
if [ -d "/opt/homebrew/Cellar/async-profiler" ]; then
    # Homebrew on Apple Silicon
    ASYNC_PROFILER=$(find /opt/homebrew/Cellar/async-profiler -name "libasyncProfiler.so" -o -name "libasyncProfiler.dylib" | head -1)
elif [ -d "/usr/local/Cellar/async-profiler" ]; then
    # Homebrew on Intel Mac
    ASYNC_PROFILER=$(find /usr/local/Cellar/async-profiler -name "libasyncProfiler.so" -o -name "libasyncProfiler.dylib" | head -1)
elif [ -d "$HOME/.sdkman/candidates/async-profiler" ]; then
    # SDKMAN installation
    ASYNC_PROFILER=$(find "$HOME/.sdkman/candidates/async-profiler" -name "libasyncProfiler.so" -o -name "libasyncProfiler.dylib" | head -1)
elif [ -d "./async-profiler" ]; then
    # Local installation
    ASYNC_PROFILER=$(find ./async-profiler -name "libasyncProfiler.so" -o -name "libasyncProfiler.dylib" | head -1)
fi

if [ -z "$ASYNC_PROFILER" ]; then
    echo "❌ async-profiler not found!"
    echo ""
    echo "Please install async-profiler:"
    echo ""
    echo "Option 1 - Homebrew (macOS):"
    echo "  brew install async-profiler"
    echo ""
    echo "Option 2 - Manual download:"
    echo "  wget https://github.com/async-profiler/async-profiler/releases/download/v3.0/async-profiler-3.0-macos.zip"
    echo "  unzip async-profiler-3.0-macos.zip"
    echo ""
    exit 1
fi

echo "✓ Found async-profiler: $ASYNC_PROFILER"
echo ""

# Always rebuild to ensure we're profiling the latest code
echo "📦 Building project..."
mvn clean package -q -DskipTests
echo ""

# Benchmark selection
echo "Select benchmark to profile:"
echo "  1) RealWorldBenchmark.parseReact (small, ~0.8ms)"
echo "  2) RealWorldBenchmark.parseVue (medium, ~5ms)"
echo "  3) RealWorldBenchmark.parseLodash (medium, ~8ms)"
echo "  4) RealWorldBenchmark.parseReactDom (large, ~18ms)"
echo "  5) RealWorldBenchmark.parseThreeJs (large, ~32ms)"
echo "  6) RealWorldBenchmark.parseTypeScript (very large, ~212ms)"
echo "  7) ComparativeParserBenchmark (all small benchmarks)"
echo ""
read -p "Enter choice [1-7] (default: 3): " choice
choice=${choice:-3}

case $choice in
    1)
        BENCHMARK="RealWorldBenchmark.parseReact"
        NAME="react"
        ;;
    2)
        BENCHMARK="RealWorldBenchmark.parseVue"
        NAME="vue"
        ;;
    3)
        BENCHMARK="RealWorldBenchmark.parseLodash"
        NAME="lodash"
        ;;
    4)
        BENCHMARK="RealWorldBenchmark.parseReactDom"
        NAME="react-dom"
        ;;
    5)
        BENCHMARK="RealWorldBenchmark.parseThreeJs"
        NAME="threejs"
        ;;
    6)
        BENCHMARK="RealWorldBenchmark.parseTypeScript"
        NAME="typescript"
        ;;
    7)
        BENCHMARK="ComparativeParserBenchmark"
        NAME="comparative"
        ;;
    *)
        echo "Invalid choice, using Lodash"
        BENCHMARK="RealWorldBenchmark.parseLodash"
        NAME="lodash"
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Running: $BENCHMARK"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Run with profiler
OUTPUT_FILE="$RESULTS_DIR/${NAME}_${TIMESTAMP}"

# Convert to absolute path if relative
if [[ "$ASYNC_PROFILER" != /* ]]; then
    ASYNC_PROFILER="$(cd "$(dirname "$ASYNC_PROFILER")" && pwd)/$(basename "$ASYNC_PROFILER")"
fi

echo "Using profiler: $ASYNC_PROFILER"
echo ""
echo "Running JMH with async-profiler..."
echo "This will take a few minutes..."
echo ""

java --enable-native-access=ALL-UNNAMED \
    -jar target/benchmarks.jar "$BENCHMARK" \
    -f 1 \
    -wi 3 \
    -i 10 \
    -prof async:libPath="$ASYNC_PROFILER"\;output=flamegraph\;dir="$RESULTS_DIR" \
    2>&1 | grep -E "(Benchmark|Score|Iteration|completed|Profiler)" || true

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Flamegraph Generated!"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Find the generated flamegraph (JMH might create it with different naming)
FLAMEGRAPH=$(find "$RESULTS_DIR" -name "*.html" -type f -mmin -5 | head -1)

if [ -n "$FLAMEGRAPH" ]; then
    echo "✅ Flamegraph saved to:"
    echo "   $FLAMEGRAPH"
    echo ""
    echo "Opening in browser..."

    # Open in default browser
    if command -v open &> /dev/null; then
        # macOS
        open "$FLAMEGRAPH"
    elif command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open "$FLAMEGRAPH"
    else
        echo "Please open the file manually in your browser"
    fi
else
    echo "⚠️  Flamegraph file not found in $RESULTS_DIR"
    echo "   Check for files matching: ${NAME}_${TIMESTAMP}*.html"
    echo ""
    echo "Available files:"
    ls -lh "$RESULTS_DIR" | grep ".html" || echo "  No HTML files found"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Tip: You can also generate CPU, allocation, or lock profiles:"
echo ""
echo "CPU profile:"
echo "  java -jar target/benchmarks.jar $BENCHMARK \\"
echo "    -prof async:event=cpu,output=flamegraph"
echo ""
echo "Allocation profile:"
echo "  java -jar target/benchmarks.jar $BENCHMARK \\"
echo "    -prof async:event=alloc,output=flamegraph"
echo ""
echo "Lock profile:"
echo "  java -jar target/benchmarks.jar $BENCHMARK \\"
echo "    -prof async:event=lock,output=flamegraph"
echo ""
