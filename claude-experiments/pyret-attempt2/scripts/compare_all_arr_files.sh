#!/bin/bash
# Compare ALL .arr files in the Pyret repository
# This shows how many real Pyret programs our parser can handle
# Automatically updates failure annotations after each run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if PYRET_REPO environment variable is set
if [ -z "$PYRET_REPO" ]; then
    echo "ERROR: PYRET_REPO environment variable is not set."
    echo ""
    echo "Usage: PYRET_REPO=/path/to/pyret-lang $0"
    echo ""
    echo "Or set it permanently in your shell profile:"
    echo "  export PYRET_REPO=/path/to/pyret-lang"
    echo ""
    exit 1
fi

# Validate that PYRET_REPO points to a valid pyret-lang repository
if [ ! -f "$PYRET_REPO/ast-to-json.jarr" ]; then
    echo "ERROR: $PYRET_REPO does not contain ast-to-json.jarr"
    echo "Please ensure PYRET_REPO points to a valid pyret-lang repository."
    exit 1
fi

echo "Using Pyret repository: $PYRET_REPO"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0

# Cache directory for Pyret parser output
CACHE_DIR="$PYRET_REPO/cache/ast-json"
mkdir -p "$CACHE_DIR"

# Temporary files
TEMP_PYRET="/tmp/pyret_bulk_test.json"
TEMP_RUST="/tmp/rust_bulk_test.json"

echo "=================================================="
echo "Pyret Parser - Full Repository Comparison"
echo "=================================================="
echo ""
echo "Cache directory: $CACHE_DIR"
echo ""
echo "Building parser in release mode..."
cd "$PROJECT_ROOT"
cargo build --release --bin to_pyret_json 2>&1 | grep -v "warning:"
PARSER_BIN="$PROJECT_ROOT/target/release/to_pyret_json"
echo "Parser built at: $PARSER_BIN"
echo ""
echo "Finding all .arr files in $PYRET_REPO ..."
echo ""

# Find all .arr files (excluding node_modules and build directories)
arr_files=$(find "$PYRET_REPO" -name "*.arr" \
  -not -path "*/node_modules/*" \
  -not -path "*/build/*" \
  -not -path "*/.git/*" \
  -not -path "*/pitometer/*" \
  2>/dev/null)

total_count=$(echo "$arr_files" | wc -l | tr -d ' ')
echo "Found $total_count .arr files to test"
echo "=================================================="
echo ""

# Create results directory
RESULTS_DIR="$PROJECT_ROOT/bulk_test_results"
mkdir -p "$RESULTS_DIR"
PASSING_LOG="$RESULTS_DIR/passing_files.txt"
FAILING_LOG="$RESULTS_DIR/failing_files.txt"

# Clear logs
> "$PASSING_LOG"
> "$FAILING_LOG"

# Test each file
current=0
for arr_file in $arr_files; do
    current=$((current + 1))
    TOTAL=$((TOTAL + 1))

    # Get relative path for display
    rel_path="${arr_file#$PYRET_REPO/}"

    # Progress indicator (every 10th file)
    if [ $((current % 10)) -eq 0 ]; then
        echo "[$current/$total_count] Testing files..."
    fi

    # Create cache file path based on source file path
    # Replace / with _ to create a flat cache structure
    cache_key=$(echo "$rel_path" | tr '/' '_' | tr '.' '_')
    cached_pyret="$CACHE_DIR/${cache_key}.json"

    # Check if we have a cached Pyret AST
    if [ -f "$cached_pyret" ]; then
        # Use cached version
        cp "$cached_pyret" "$TEMP_PYRET"
    else
        # Parse with official Pyret parser and cache the result
        if ! node "$PYRET_REPO/ast-to-json.jarr" "$arr_file" "$TEMP_PYRET" 2>/dev/null 1>&2; then
            # Pyret parser failed - skip this file
            echo -e "${YELLOW}SKIP${NC} $rel_path (Pyret parser failed)"
            continue
        fi
        # Cache the successful parse
        cp "$TEMP_PYRET" "$cached_pyret"
    fi

    # Try to parse with our Rust parser (using pre-built binary)
    if ! "$PARSER_BIN" "$arr_file" 2>/dev/null > "$TEMP_RUST"; then
        # Rust parser failed
        FAILED=$((FAILED + 1))
        echo -e "${RED}FAIL${NC} $rel_path (Rust parser failed)"
        echo "$rel_path" >> "$FAILING_LOG"
        continue
    fi

    # Compare the two JSON outputs (normalize for field order)
    if python3 << 'PYTHON_SCRIPT'
import json
import sys
import re

def normalize_srcloc(text):
    if isinstance(text, str):
        return re.sub(r'srcloc\("([^"]+)"', 'srcloc("file.arr"', text)
    return text

def normalize_json(obj):
    if isinstance(obj, dict):
        return {k: normalize_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj]
    elif isinstance(obj, str):
        return normalize_srcloc(obj)
    else:
        return obj

try:
    with open('/tmp/pyret_bulk_test.json') as f:
        pyret = json.load(f)
    with open('/tmp/rust_bulk_test.json') as f:
        rust = json.load(f)

    pyret_norm = normalize_json(pyret)
    rust_norm = normalize_json(rust)

    sys.exit(0 if pyret_norm == rust_norm else 1)
except Exception as e:
    sys.exit(1)
PYTHON_SCRIPT
    then
        # Success!
        PASSED=$((PASSED + 1))
        echo -e "${GREEN}PASS${NC} $rel_path"
        echo "$rel_path" >> "$PASSING_LOG"
    else
        # JSON mismatch
        FAILED=$((FAILED + 1))
        echo -e "${RED}FAIL${NC} $rel_path (AST mismatch)"
        echo "$rel_path" >> "$FAILING_LOG"
    fi
done

echo ""
echo "=================================================="
echo "FINAL RESULTS"
echo "=================================================="
echo "Total files tested: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC} ($(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")%)"
echo -e "${RED}Failed: $FAILED${NC} ($(awk "BEGIN {printf \"%.1f\", ($FAILED/$TOTAL)*100}")%)"
echo ""
echo "Detailed results saved to:"
echo "  Passing: $PASSING_LOG"
echo "  Failing: $FAILING_LOG"
echo "=================================================="
echo ""

# Run re-annotation automatically
if [ -f "$FAILING_LOG" ] && [ $FAILED -gt 0 ]; then
    echo ""
    echo "🔄 Analyzing failures and updating annotations..."
    echo ""

    # Create failure analysis file
    OUTPUT_FILE="$RESULTS_DIR/failure_analysis.txt"
    > "$OUTPUT_FILE"

    echo "Analyzing failures..."
    file_count=0
    while IFS= read -r rel_path; do
        file_count=$((file_count + 1))
        if [ $((file_count % 50)) -eq 0 ]; then
            echo "  Analyzed $file_count files..."
        fi

        echo "=== $rel_path ===" >> "$OUTPUT_FILE"

        full_path="$PYRET_REPO/$rel_path"

        # Run parser and capture error
        error=$("$PARSER_BIN" "$full_path" 2>&1 | \
                grep -v "warning:" | \
                grep -E "(Error|Expected|Unexpected|failed)" | \
                head -1)

        if [ -z "$error" ]; then
            echo "SUCCESS: File parses correctly (AST mismatch)" >> "$OUTPUT_FILE"
        else
            echo "$error" >> "$OUTPUT_FILE"
        fi
        echo "" >> "$OUTPUT_FILE"
    done < "$FAILING_LOG"

    echo "  Analyzed $file_count files total."
    echo ""

    # Run categorization if scripts exist
    if [ -f "$SCRIPT_DIR/final_accurate_categorize.py" ]; then
        echo "Categorizing parse errors..."
        (cd "$SCRIPT_DIR" && python3 final_accurate_categorize.py 2>/dev/null)
        echo ""
    fi

    if [ -f "$SCRIPT_DIR/categorize_mismatches.py" ]; then
        echo "Checking AST differences..."
        (cd "$SCRIPT_DIR" && python3 categorize_mismatches.py 2>/dev/null)
        echo ""
    fi

    echo "✅ Annotations updated!"
    echo ""
fi

# Show summary of what features are missing
if [ $FAILED -gt 0 ]; then
    echo "Common failures likely due to missing features:"
    echo "  - Generic types (<T>)"
    echo "  - Object extension (obj.{ field: value })"
    echo "  - File imports (import file(\"...\"))"
    echo "  - Table expressions"
    echo ""
    echo "See NEXT_STEPS.md for implementation guide!"
fi

# Exit with success code
exit 0
