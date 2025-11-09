#!/bin/bash
# Parse ALL .arr files in the Pyret repository with our Rust parser
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

echo "=================================================="
echo "Pyret Parser - Parse All .arr Files"
echo "=================================================="
echo ""
echo "Building parser in release mode..."
cargo build --release --bin parse_only 2>&1 | grep -v "warning:"
PARSER_BIN="$SCRIPT_DIR/target/release/parse_only"
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
echo "Found $total_count .arr files to parse"
echo "=================================================="
echo ""

# Create results directory
RESULTS_DIR="$SCRIPT_DIR/bulk_test_results"
mkdir -p "$RESULTS_DIR"
PASSING_LOG="$RESULTS_DIR/rust_parsing_success.txt"
FAILING_LOG="$RESULTS_DIR/rust_parsing_failed.txt"

# Clear logs
> "$PASSING_LOG"
> "$FAILING_LOG"

# Parse each file
current=0
for arr_file in $arr_files; do
    current=$((current + 1))
    TOTAL=$((TOTAL + 1))

    # Get relative path for display
    rel_path="${arr_file#$PYRET_REPO/}"

    # Show which file we're starting to parse
    echo -n "[$current/$total_count] Parsing $rel_path ... "

    # Try to parse with our Rust parser (using pre-built binary)
    if "$PARSER_BIN" "$arr_file" 2>/dev/null; then
        # Success!
        PASSED=$((PASSED + 1))
        echo -e "${GREEN}✓${NC}"
        echo "$rel_path" >> "$PASSING_LOG"
    else
        # Parser failed
        FAILED=$((FAILED + 1))
        echo -e "${RED}✗${NC}"
        echo "$rel_path" >> "$FAILING_LOG"
    fi
done

echo ""
echo "=================================================="
echo "FINAL RESULTS"
echo "=================================================="
echo "Total files tested: $TOTAL"
echo -e "${GREEN}Parsed successfully: $PASSED${NC} ($(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")%)"
echo -e "${RED}Failed to parse: $FAILED${NC} ($(awk "BEGIN {printf \"%.1f\", ($FAILED/$TOTAL)*100}")%)"
echo ""
echo "Detailed results saved to:"
echo "  Success: $PASSING_LOG"
echo "  Failed: $FAILING_LOG"
echo "=================================================="
echo ""

# Show summary
if [ $FAILED -gt 0 ]; then
    echo "To see error details for a failed file, run:"
    echo "  cargo run --bin to_pyret_json /path/to/file.arr"
    echo ""
fi

# Exit with success code
exit 0
