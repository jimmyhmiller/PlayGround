#!/bin/bash
# Unified Cross-Language Parser Benchmark
# Runs our Java parser alongside Rust and JavaScript parsers
# Produces a single comparison table for each library
#
# METHODOLOGY:
# Each implementation:
#   1. Starts a single process
#   2. Loads all test files into memory
#   3. For each file: performs internal warmup, then measurement
#   4. Reports only parsing time (excludes process startup/file I/O)
#
# This ensures we measure parsing performance consistently across all
# implementations without including JIT compilation or process startup costs.
#
# Usage:
#   ./run-unified-comparison.sh [warmup] [measurement]
#   ./run-unified-comparison.sh           # 5 warmup, 10 measurement (default)
#   ./run-unified-comparison.sh 10 50     # 10 warmup, 50 measurement

set -e

# Parse command line arguments
WARMUP_ITERATIONS=${1:-5}
MEASUREMENT_ITERATIONS=${2:-10}

RESULTS_DIR="benchmark-results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "════════════════════════════════════════════════════════════════"
echo "  Unified Cross-Language Parser Benchmark"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Methodology: Each process performs internal warmup then measurement"
echo "  • Warmup: $WARMUP_ITERATIONS iterations (excludes from timing)"
echo "  • Measurement: $MEASUREMENT_ITERATIONS iterations (averaged)"
echo "  • Only parsing time measured (no startup/I/O)"
echo ""
echo "Parsers:"
echo "  • Our Java Parser (JMH)"
echo "  • Rust: OXC, SWC"
echo "  • Go: esbuild"
echo "  • JavaScript: Babel, Acorn, Meriyah"
echo ""
echo "Libraries:"
echo "  • React (10.5 KB)"
echo "  • Vue 3 (130 KB)"
echo "  • React DOM (128.8 KB)"
echo "  • Lodash (531.3 KB)"
echo "  • Three.js (1.28 MB)"
echo "  • TypeScript (8.8 MB)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# 1. Build and run our Java parser
echo "[1/4] Running Our Java Parser..."
echo "  Building..."
mvn compile -q -DskipTests
echo "  Running benchmark..."
mvn exec:java -q -Dexec.mainClass="com.jsparser.benchmarks.SimpleBenchmark" \
    -Dexec.args="$WARMUP_ITERATIONS $MEASUREMENT_ITERATIONS" \
    2>&1 | tee "$RESULTS_DIR/java_our_${TIMESTAMP}.txt"
echo "  ✓ Java benchmarks complete"
echo ""

# 2. Run Rust benchmarks
echo "[2/4] Running Rust Parsers (OXC, SWC)..."
cd benchmarks/rust
cargo build --release 2>&1 | grep -v "Compiling\|Finished\|warning" || true
cargo run --release --bin benchmark-real-world --quiet -- "$WARMUP_ITERATIONS" "$MEASUREMENT_ITERATIONS" \
    2>&1 | tee "../../$RESULTS_DIR/rust_${TIMESTAMP}.txt" | grep -E "(Library:|Parser|OXC|SWC)" || true
cd ../..
echo "  ✓ Rust benchmarks complete"
echo ""

# 3. Run Go benchmarks (using esbuild's internal parser directly)
echo "[3/4] Running Go Parser (esbuild)..."
cd benchmarks/go
if [ ! -d "esbuild" ]; then
    echo "  Cloning esbuild..."
    git clone --depth 1 https://github.com/evanw/esbuild.git esbuild 2>&1 | grep -v "^Cloning\|^remote:" || true
fi
cd esbuild
go build -o ../benchmark-go ./cmd/benchmark 2>&1 | grep -v "^go:" || true
cd ..
./benchmark-go "$WARMUP_ITERATIONS" "$MEASUREMENT_ITERATIONS" \
    2>&1 | tee "../../$RESULTS_DIR/go_${TIMESTAMP}.txt" | grep -E "(Library:|Parser|esbuild)" || true
cd ../..
echo "  ✓ Go benchmarks complete"
echo ""

# 4. Run JavaScript benchmarks
echo "[4/4] Running JavaScript Parsers..."
cd benchmarks/javascript
node benchmark-real-world.js "$WARMUP_ITERATIONS" "$MEASUREMENT_ITERATIONS" \
    2>&1 | tee "../../$RESULTS_DIR/js_${TIMESTAMP}.txt" | grep -E "(Library:|Parser|Babel|Acorn|Esprima|Meriyah)" || true
cd ../..
echo "  ✓ JavaScript benchmarks complete"
echo ""

# 4. Generate unified comparison
echo "════════════════════════════════════════════════════════════════"
echo "  Generating Unified Comparison Table"
echo "════════════════════════════════════════════════════════════════"
echo ""

python3 - <<'PYTHON_SCRIPT'
import json
import re

# Parse Java text output
def parse_java_txt(txt_file):
    results = {}
    try:
        with open(txt_file) as f:
            content = f.read()
            current_lib = None

            for line in content.split('\n'):
                if line.startswith('Library:'):
                    lib_match = re.search(r'Library: (.*?)$', line)
                    if lib_match:
                        lib_name = lib_match.group(1).strip().lower()
                        if 'react dom' in lib_name:
                            current_lib = 'reactdom'
                        elif 'react' in lib_name and 'dom' not in lib_name:
                            current_lib = 'react'
                        elif 'vue' in lib_name:
                            current_lib = 'vue'
                        elif 'lodash' in lib_name:
                            current_lib = 'lodash'
                        elif 'three' in lib_name:
                            current_lib = 'threejs'
                        elif 'typescript' in lib_name:
                            current_lib = 'typescript'

                elif current_lib and 'Our Java Parser' in line and '|' in line:
                    # Parse result line: "🥇 Our Java Parser    |         123.456 |              100.0"
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 2:
                        try:
                            time = float(parts[1].strip())
                            results[current_lib] = time
                        except:
                            pass
    except Exception as e:
        print(f"Warning: Could not parse Java results: {e}")
    return results

# Parse Rust text output
def parse_rust_txt(txt_file):
    results = {}
    try:
        with open(txt_file) as f:
            content = f.read()
            # Match library sections
            sections = re.findall(
                r'Library: (.*?)\n.*?'
                r'OXC.*?\|\s*([\d.]+).*?\n'
                r'.*?SWC.*?\|\s*([\d.]+)',
                content,
                re.DOTALL
            )
            for lib_name, oxc_time, swc_time in sections:
                lib_key = lib_name.strip().lower().replace(' ', '').replace('.js', '')
                if lib_key == 'reactdom':
                    lib_key = 'reactdom'
                elif lib_key == 'vue3':
                    lib_key = 'vue'
                elif lib_key == 'three':
                    lib_key = 'threejs'
                elif lib_key == 'typescriptcompiler':
                    lib_key = 'typescript'

                results[lib_key] = {
                    'oxc': float(oxc_time),
                    'swc': float(swc_time)
                }
    except Exception as e:
        print(f"Warning: Could not parse Rust results: {e}")
    return results

# Parse JavaScript text output
def parse_js_txt(txt_file):
    results = {}
    try:
        with open(txt_file) as f:
            content = f.read()
            current_lib = None

            for line in content.split('\n'):
                if line.startswith('Library:'):
                    # Extract library name
                    lib_match = re.search(r'Library: (.*?)(?:\.js)?$', line)
                    if lib_match:
                        lib_name = lib_match.group(1).strip().lower()
                        lib_name = lib_name.replace('.production.min', '').replace('-', '').replace('.', '')
                        if 'reactdom' in lib_name:
                            current_lib = 'reactdom'
                        elif 'react' in lib_name and 'dom' not in lib_name:
                            current_lib = 'react'
                        elif 'vue' in lib_name:
                            current_lib = 'vue'
                        elif 'lodash' in lib_name:
                            current_lib = 'lodash'
                        elif 'three' in lib_name:
                            current_lib = 'threejs'
                        elif 'typescript' in lib_name:
                            current_lib = 'typescript'

                        if current_lib:
                            results[current_lib] = {}

                elif current_lib and '|' in line:
                    # Parse result line
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 2:
                        parser_name = parts[0].replace('🥇', '').replace('🥈', '').replace('🥉', '').strip()
                        try:
                            time = float(parts[1].strip())
                            if '@babel/parser' in parser_name.lower():
                                results[current_lib]['babel'] = time
                            elif 'acorn' in parser_name.lower():
                                results[current_lib]['acorn'] = time
                            elif 'meriyah' in parser_name.lower():
                                results[current_lib]['meriyah'] = time
                        except:
                            pass
    except Exception as e:
        print(f"Warning: Could not parse JS results: {e}")
    return results

# Parse Go text output
def parse_go_txt(txt_file):
    results = {}
    try:
        with open(txt_file) as f:
            content = f.read()
            current_lib = None

            for line in content.split('\n'):
                if line.startswith('Library:'):
                    lib_match = re.search(r'Library: (.*?)$', line)
                    if lib_match:
                        lib_name = lib_match.group(1).strip().lower()
                        if 'react dom' in lib_name:
                            current_lib = 'reactdom'
                        elif 'react' in lib_name and 'dom' not in lib_name:
                            current_lib = 'react'
                        elif 'vue' in lib_name:
                            current_lib = 'vue'
                        elif 'lodash' in lib_name:
                            current_lib = 'lodash'
                        elif 'three' in lib_name:
                            current_lib = 'threejs'
                        elif 'typescript' in lib_name:
                            current_lib = 'typescript'

                elif current_lib and 'esbuild' in line and '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 2:
                        try:
                            time = float(parts[1].strip())
                            results[current_lib] = time
                        except:
                            pass
    except Exception as e:
        print(f"Warning: Could not parse Go results: {e}")
    return results

# Find latest result files
import glob
import os

RESULTS_DIR = "benchmark-results"

latest_java = max(glob.glob(f'{RESULTS_DIR}/java_our_*.txt'), default=None, key=os.path.getctime)
latest_rust = max(glob.glob(f'{RESULTS_DIR}/rust_*.txt'), default=None, key=os.path.getctime)
latest_go = max(glob.glob(f'{RESULTS_DIR}/go_*.txt'), default=None, key=os.path.getctime)
latest_js = max(glob.glob(f'{RESULTS_DIR}/js_*.txt'), default=None, key=os.path.getctime)

java_results = parse_java_txt(latest_java) if latest_java else {}
rust_results = parse_rust_txt(latest_rust) if latest_rust else {}
go_results = parse_go_txt(latest_go) if latest_go else {}
js_results = parse_js_txt(latest_js) if latest_js else {}

# Library definitions
libraries = [
    ('react', 'React', 10.5),
    ('vue', 'Vue 3', 130.0),
    ('reactdom', 'React DOM', 128.8),
    ('lodash', 'Lodash', 531.3),
    ('threejs', 'Three.js', 1282.3),
    ('typescript', 'TypeScript', 8808.7),
]

print("\n" + "=" * 110)
print(" " * 25 + "UNIFIED CROSS-LANGUAGE PARSER BENCHMARK")
print("=" * 110)

# Track aggregate stats per parser
parser_totals = {}  # parser -> {'total_time': x, 'total_size': y, 'libs': n}

for lib_key, lib_name, size_kb in libraries:
    print(f"\n{lib_name} ({size_kb:.1f} KB)")
    print("-" * 110)
    print(f"{'Parser':<30} | {'Time (ms)':>12} | {'vs Fastest':>12} | {'Throughput (KB/ms)':>20}")
    print("-" * 110)

    # Collect all times
    times = []

    # Our Java parser
    if lib_key in java_results:
        times.append(('Our Java Parser', java_results[lib_key]))

    # Rust parsers
    if lib_key in rust_results:
        if 'oxc' in rust_results[lib_key]:
            times.append(('OXC (Rust)', rust_results[lib_key]['oxc']))
        if 'swc' in rust_results[lib_key]:
            times.append(('SWC (Rust)', rust_results[lib_key]['swc']))

    # Go parsers
    if lib_key in go_results:
        times.append(('esbuild (Go)', go_results[lib_key]))

    # JavaScript parsers
    if lib_key in js_results:
        if 'babel' in js_results[lib_key]:
            times.append(('Babel (JavaScript)', js_results[lib_key]['babel']))
        if 'acorn' in js_results[lib_key]:
            times.append(('Acorn (JavaScript)', js_results[lib_key]['acorn']))
        if 'meriyah' in js_results[lib_key]:
            times.append(('Meriyah (JavaScript)', js_results[lib_key]['meriyah']))

    if not times:
        print(f"  No results available for {lib_name}")
        continue

    # Sort by time (fastest first)
    times.sort(key=lambda x: x[1])
    fastest_time = times[0][1]

    # Print sorted results and accumulate totals
    for i, (parser, time) in enumerate(times):
        vs_fastest = time / fastest_time
        throughput = size_kb / time if time > 0 else 0

        medal = ''
        if i == 0:
            medal = '🥇 '
        elif i == 1:
            medal = '🥈 '
        elif i == 2:
            medal = '🥉 '

        print(f"{medal}{parser:<30} | {time:>12.3f} | {vs_fastest:>11.2f}x | {throughput:>20.1f}")

        # Accumulate totals
        if parser not in parser_totals:
            parser_totals[parser] = {'total_time': 0, 'total_size': 0, 'libs': 0}
        parser_totals[parser]['total_time'] += time
        parser_totals[parser]['total_size'] += size_kb
        parser_totals[parser]['libs'] += 1

# Print overall ranking
print("\n" + "=" * 110)
print(" " * 35 + "OVERALL RANKING")
print("=" * 110)
print(f"\n{'Parser':<30} | {'Total Time (ms)':>15} | {'vs Fastest':>12} | {'Avg Throughput (KB/ms)':>22} | {'Libs':>5}")
print("-" * 110)

# Sort by total time
ranking = [(p, d['total_time'], d['total_size'], d['libs'])
           for p, d in parser_totals.items()]
ranking.sort(key=lambda x: x[1])

if ranking:
    fastest_total = ranking[0][1]
    for i, (parser, total_time, total_size, libs) in enumerate(ranking):
        vs_fastest = total_time / fastest_total if fastest_total > 0 else 0
        avg_throughput = total_size / total_time if total_time > 0 else 0

        medal = ''
        if i == 0:
            medal = '🥇 '
        elif i == 1:
            medal = '🥈 '
        elif i == 2:
            medal = '🥉 '

        print(f"{medal}{parser:<30} | {total_time:>15.3f} | {vs_fastest:>11.2f}x | {avg_throughput:>22.1f} | {libs:>5}")

print("\n" + "=" * 110)
print()
PYTHON_SCRIPT

echo "✅ Unified benchmark complete!"
echo ""
echo "Raw results saved to:"
echo "  • Java:       $RESULTS_DIR/java_our_${TIMESTAMP}.txt"
echo "  • Rust:       $RESULTS_DIR/rust_${TIMESTAMP}.txt"
echo "  • Go:         $RESULTS_DIR/go_${TIMESTAMP}.txt"
echo "  • JavaScript: $RESULTS_DIR/js_${TIMESTAMP}.txt"
echo ""
