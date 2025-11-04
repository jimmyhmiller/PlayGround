#!/bin/bash
# Compare ALL .arr files in the Pyret repository
# This shows how many real Pyret programs our parser can handle

set -e

PYRET_REPO="/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0

# Temporary files
TEMP_PYRET="/tmp/pyret_bulk_test.json"
TEMP_RUST="/tmp/rust_bulk_test.json"

echo "=================================================="
echo "Pyret Parser - Full Repository Comparison"
echo "=================================================="
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
RESULTS_DIR="$SCRIPT_DIR/bulk_test_results"
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

    # Try to parse with official Pyret parser
    if ! node "$PYRET_REPO/ast-to-json.jarr" "$arr_file" "$TEMP_PYRET" 2>/dev/null 1>&2; then
        # Pyret parser failed - skip this file
        echo -e "${YELLOW}SKIP${NC} $rel_path (Pyret parser failed)"
        continue
    fi

    # Try to parse with our Rust parser
    if ! cargo run --quiet --bin to_pyret_json "$arr_file" 2>/dev/null > "$TEMP_RUST"; then
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
