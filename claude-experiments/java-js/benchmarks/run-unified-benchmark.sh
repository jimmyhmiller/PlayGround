#!/bin/bash
# Unified cross-language benchmark runner
# Runs our Java parser, Rust parsers, and JS parsers on the same files
# and produces a unified comparison table

set -e

# Create results directory
RESULTS_DIR="../benchmark-results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "════════════════════════════════════════════════════════════════"
echo "  Unified Cross-Language Parser Benchmark"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "This will benchmark:"
echo "  - Our Java Parser"
echo "  - Rust parsers (OXC, SWC)"
echo "  - JavaScript parsers (Babel, Acorn, Esprima, Meriyah)"
echo ""
echo "On real-world libraries:"
echo "  - React (10.5 KB)"
echo "  - Vue 3 (130 KB)"
echo "  - React DOM (128.8 KB)"
echo "  - Lodash (531.3 KB)"
echo "  - Three.js (1.28 MB)"
echo "  - TypeScript (8.8 MB)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Build our Java parser if needed
if [ ! -f "../target/benchmarks.jar" ]; then
    echo "Building Java parser..."
    cd ..
    mvn clean package -q -DskipTests
    cd benchmarks
fi

# 1. Run our Java parser benchmarks
echo "[1/3] Benchmarking Our Java Parser..."
cd ..
java -jar target/benchmarks.jar ".*ourParser.*react.*" -f 1 -wi 2 -i 5 -rf json -rff "$RESULTS_DIR/java_our_parser_${TIMESTAMP}.json" 2>&1 | grep -E "(Benchmark|Score)" || true
cd benchmarks
echo ""

# 2. Run Rust benchmarks
echo "[2/3] Benchmarking Rust Parsers (OXC, SWC)..."
cd rust
cargo run --release --bin benchmark-real-world --quiet 2>&1 > "../../$RESULTS_DIR/rust_${TIMESTAMP}.txt"
cd ..
echo ""

# 3. Run JavaScript benchmarks
echo "[3/3] Benchmarking JavaScript Parsers..."
cd javascript
node benchmark-real-world.js 2>&1 > "../../$RESULTS_DIR/js_${TIMESTAMP}.txt"
cd ..
echo ""

# 4. Generate unified comparison table
echo "════════════════════════════════════════════════════════════════"
echo "  Generating Unified Results Table"
echo "════════════════════════════════════════════════════════════════"
echo ""

python3 - <<'EOF'
import json
import re
import sys

# Parse Java results from JSON
def parse_java_results(json_file):
    try:
        with open(json_file) as f:
            data = json.load(f)
            results = {}
            for benchmark in data:
                name = benchmark['benchmark']
                # Extract library name from benchmark name
                match = re.search(r'ourParser_(\w+)', name)
                if match:
                    lib = match.group(1)
                    # Convert microseconds to milliseconds
                    time_ms = benchmark['primaryMetric']['score']
                    results[lib] = time_ms
            return results
    except:
        return {}

# Parse Rust results
def parse_rust_results(txt_file):
    results = {}
    try:
        with open(txt_file) as f:
            content = f.read()
            # Find each library section
            libraries = re.findall(r'Library: (.*?)\n.*?OXC.*?\|\s+([\d.]+)', content, re.DOTALL)
            for lib, time in libraries:
                lib_key = lib.strip().lower().replace(' ', '_')
                results[lib_key] = {'oxc': float(time)}

            # Get SWC times
            swc_times = re.findall(r'SWC.*?\|\s+([\d.]+)', content)
            lib_names = re.findall(r'Library: (.*?)\n', content)
            for i, (lib, swc_time) in enumerate(zip(lib_names, swc_times)):
                lib_key = lib.strip().lower().replace(' ', '_')
                if lib_key in results:
                    results[lib_key]['swc'] = float(swc_time)
    except:
        pass
    return results

# Parse JavaScript results
def parse_js_results(txt_file):
    results = {}
    try:
        with open(txt_file) as f:
            content = f.read()
            current_lib = None
            for line in content.split('\n'):
                if line.startswith('Library:'):
                    current_lib = line.split(':')[1].strip().lower().replace('.js', '').replace('.', '_')
                    results[current_lib] = {}
                elif '|' in line and current_lib:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 2 and parts[0]:
                        parser = parts[0].replace('🥇', '').replace('🥈', '').replace('🥉', '').strip().lower()
                        time_str = parts[1].strip()
                        try:
                            time = float(time_str)
                            results[current_lib][parser] = time
                        except:
                            pass
    except:
        pass
    return results

# Load all results
import glob
import os

results_dir = '../benchmark-results'
latest_java = max(glob.glob(f'{results_dir}/java_our_parser_*.json'), default=None, key=os.path.getctime)
latest_rust = max(glob.glob(f'{results_dir}/rust_*.txt'), default=None, key=os.path.getctime)
latest_js = max(glob.glob(f'{results_dir}/js_*.txt'), default=None, key=os.path.getctime)

java_results = parse_java_results(latest_java) if latest_java else {}
rust_results = parse_rust_results(latest_rust) if latest_rust else {}
js_results = parse_js_results(latest_js) if latest_js else {}

# Library mappings
libs = [
    ('react', 'React', '10.5 KB'),
    ('vue', 'Vue 3', '130 KB'),
    ('react_dom', 'React DOM', '128.8 KB'),
    ('lodash', 'Lodash', '531.3 KB'),
    ('three', 'Three.js', '1.28 MB'),
    ('typescript', 'TypeScript', '8.8 MB'),
]

print("\n" + "="*120)
print("UNIFIED CROSS-LANGUAGE PARSER BENCHMARK RESULTS")
print("="*120)

for lib_key, lib_name, size in libs:
    print(f"\n{lib_name} ({size})")
    print("-" * 120)
    print(f"{'Parser':<25} | {'Time (ms)':>12} | {'vs Our Parser':>15} | {'Throughput':>20}")
    print("-" * 120)

    # Collect all times for this library
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

    # JS parsers
    js_lib_key = lib_key.replace('_', '-')
    for js_key in [lib_key, js_lib_key, f'{lib_key}_production_min', f'{lib_key}.production.min']:
        if js_key in js_results:
            for parser, time in js_results[js_key].items():
                if 'babel' in parser:
                    times.append(('Babel (JS)', time))
                elif 'acorn' in parser:
                    times.append(('Acorn (JS)', time))
                elif 'esprima' in parser:
                    times.append(('Esprima (JS)', time))
                elif 'meriyah' in parser:
                    times.append(('Meriyah (JS)', time))

    # Sort by time and display
    times.sort(key=lambda x: x[1])

    our_time = java_results.get(lib_key, times[0][1] if times else 1)

    for i, (parser, time) in enumerate(times):
        vs_ours = time / our_time if our_time > 0 else 0
        vs_str = f"{vs_ours:.2f}x"

        # Calculate throughput (rough estimate)
        size_kb = float(size.split()[0])
        if 'MB' in size:
            size_kb *= 1024
        throughput = size_kb / time if time > 0 else 0

        medal = ''
        if i == 0:
            medal = '🥇 '
        elif i == 1:
            medal = '🥈 '
        elif i == 2:
            medal = '🥉 '

        print(f"{medal}{parser:<25} | {time:>12.3f} | {vs_str:>15} | {throughput:>17.1f} KB/ms")

    if not times:
        print("  No results available")

print("\n" + "="*120)
print()
EOF

echo "✅ Unified benchmark complete!"
echo ""
echo "Results saved to: $RESULTS_DIR/"
