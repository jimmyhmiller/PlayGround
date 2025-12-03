#!/bin/bash
# Automated test suite to compare our implementation with Clojure
# Usage: ./tests/compare_with_clojure.sh [test_file.clj]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if clojure is available
if ! command -v clj &> /dev/null; then
    echo -e "${RED}Error: clj command not found. Please install Clojure.${NC}"
    exit 1
fi

# Build our implementation in release mode for accurate testing
echo "Building implementation..."
cargo build --release --quiet

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a single test expression
test_expression() {
    local expr="$1"
    local test_name="$2"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Run our implementation (suppress stderr which has DEBUG and assembly output)
    echo "$expr" > /tmp/test_expr.clj
    our_result=$(cargo run --release --quiet /tmp/test_expr.clj 2>/dev/null)

    # Run Clojure
    clj_result=$(clj -e "$expr" 2>/dev/null | tail -1)

    # Our implementation now matches Clojure's output format exactly
    # (nil prints nothing, booleans print as text, integers are untagged)
    our_display="$our_result"

    # For comparison, treat empty output as "nil" in both cases
    if [ -z "$our_result" ]; then
        our_display="nil"
    fi
    if [ -z "$clj_result" ]; then
        clj_result="nil"
    fi

    # Compare results
    if [ "$our_display" = "$clj_result" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        echo "  Expression: $expr"
        echo "  Result: $clj_result"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Expression: $expr"
        echo "  Expected (Clojure): $clj_result"
        echo "  Got (Ours): $our_display"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo
}

# Function to run tests from a file
run_test_file() {
    local file="$1"
    echo "========================================="
    echo "Running tests from: $file"
    echo "========================================="
    echo

    # Read test file and extract test expressions
    # Format: ;; TEST: description
    #         (expression)
    while IFS= read -r line; do
        if [[ "$line" =~ ^";; TEST: "(.+)$ ]]; then
            test_name="${BASH_REMATCH[1]}"
            # Read next non-empty, non-comment line as the expression
            while IFS= read -r expr_line; do
                # Skip empty lines
                if [[ "$expr_line" =~ ^[[:space:]]*$ ]]; then
                    continue
                fi
                # Skip comment lines
                if [[ "$expr_line" =~ ^";;" ]]; then
                    continue
                fi
                # Found the expression
                test_expression "$expr_line" "$test_name"
                break
            done
        fi
    done < "$file"
}

# If a test file is provided, run it
if [ $# -eq 1 ]; then
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: Test file not found: $1${NC}"
        exit 1
    fi
    run_test_file "$1"
else
    # Run all test suites
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║     Clojure Compatibility Test Suite                          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo

    # Basic equality and comparison tests
    echo "========================================="
    echo "Equality Tests"
    echo "========================================="
    echo

    test_expression "(= nil 0)" "nil not equal to 0"
    test_expression "(= nil false)" "nil not equal to false"
    test_expression "(= false 0)" "false not equal to 0"
    test_expression "(= true false)" "true not equal to false"
    test_expression "(= 5 5)" "equal integers"
    test_expression "(= 5 3)" "unequal integers"

    echo "========================================="
    echo "Comparison Tests"
    echo "========================================="
    echo

    test_expression "(< 1 2)" "less than true"
    test_expression "(< 2 1)" "less than false"
    test_expression "(> 2 1)" "greater than true"
    test_expression "(> 1 2)" "greater than false"

    echo "========================================="
    echo "Arithmetic Tests"
    echo "========================================="
    echo

    test_expression "(+ 1 2)" "simple addition"
    test_expression "(* 2 3)" "simple multiplication"
    test_expression "(- 5 3)" "simple subtraction"

    echo "========================================="
    echo "Let Expression Tests"
    echo "========================================="
    echo

    test_expression "(let [x 2])" "empty let body returns nil"
    test_expression "(let [x 5] x)" "let with single binding"
    test_expression "(let [x 2 y 3] (+ x y))" "let with multiple bindings"

    echo "========================================="
    echo "Boolean Tests"
    echo "========================================="
    echo

    test_expression "true" "literal true"
    test_expression "false" "literal false"
    test_expression "nil" "literal nil"
fi

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "Total:  $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed! 🎉${NC}"
    exit 0
fi
