#!/bin/bash

# Test runner script to check exit status of all tests individually
# This helps identify which specific tests are causing crashes

set +e  # Don't exit on error - we want to capture all exit codes

echo "🧪 MLIR Test Runner - Individual Test Status Check"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run a test and capture exit status
run_test() {
    local test_name="$1"
    local test_args="$2"
    
    echo -n "Testing ${test_name}... "
    
    if [ -n "$test_args" ]; then
        cargo test $test_args > /dev/null 2>&1
    elif [ "$test_name" = "ALL_TESTS" ]; then
        cargo test > /dev/null 2>&1
    else
        cargo test --test "$test_name" > /dev/null 2>&1
    fi
    
    local exit_code=$?
    
    case $exit_code in
        0)
            echo -e "${GREEN}✅ PASS${NC} (exit: $exit_code)"
            ;;
        5)
            echo -e "${RED}❌ SIGTRAP${NC} (exit: $exit_code)"
            ;;
        6)
            echo -e "${RED}❌ SIGABRT${NC} (exit: $exit_code)"
            ;;
        9)
            echo -e "${RED}❌ SIGKILL${NC} (exit: $exit_code)"
            ;;
        11)
            echo -e "${RED}❌ SIGSEGV${NC} (exit: $exit_code)"
            ;;
        *)
            echo -e "${YELLOW}⚠️  FAIL${NC} (exit: $exit_code)"
            ;;
    esac
    
    return $exit_code
}

# Function to run test with detailed output on failure
run_test_verbose() {
    local test_name="$1"
    local test_args="$2"
    
    echo "🔍 Running $test_name with verbose output:"
    echo "----------------------------------------"
    
    if [ -n "$test_args" ]; then
        cargo test $test_args
    elif [ "$test_name" = "ALL_TESTS" ]; then
        cargo test
    else
        cargo test --test "$test_name"
    fi
    
    local exit_code=$?
    echo "Exit code: $exit_code"
    echo ""
    
    return $exit_code
}

# Track statistics
total_tests=0
passed_tests=0
failed_tests=0
crashed_tests=0

echo -e "${BLUE}📋 Individual Test Files:${NC}"
echo ""

# Test individual test files
test_files=(
    "build_system_tests"
    "ffi_binding_simple_tests" 
    "regression_tests_safe"
    "tensor_dialect_tests"
    "tensor_ops_comprehensive_tests"
    "tensor_ops_simple_tests"
)

for test_file in "${test_files[@]}"; do
    run_test "$test_file"
    exit_code=$?
    total_tests=$((total_tests + 1))
    
    if [ $exit_code -eq 0 ]; then
        passed_tests=$((passed_tests + 1))
    elif [ $exit_code -eq 5 ] || [ $exit_code -eq 6 ] || [ $exit_code -eq 9 ] || [ $exit_code -eq 11 ]; then
        crashed_tests=$((crashed_tests + 1))
    else
        failed_tests=$((failed_tests + 1))
    fi
done

echo ""
echo -e "${BLUE}📋 Binary Unit Tests:${NC}"
echo ""

# Test binary unit tests
binaries=(
    "melior-test"
    "jit-test" 
    "minimal_jit"
    "minimal_working"
    "simple-jit"
    "simple-test"
    "simple_test"
)

for binary in "${binaries[@]}"; do
    run_test "$binary" "--bin $binary"
    exit_code=$?
    total_tests=$((total_tests + 1))
    
    if [ $exit_code -eq 0 ]; then
        passed_tests=$((passed_tests + 1))
    elif [ $exit_code -eq 5 ] || [ $exit_code -eq 6 ] || [ $exit_code -eq 9 ] || [ $exit_code -eq 11 ]; then
        crashed_tests=$((crashed_tests + 1))
    else
        failed_tests=$((failed_tests + 1))
    fi
done

echo ""
echo -e "${BLUE}📋 Full Test Suite:${NC}"
echo ""

run_test "ALL_TESTS" ""
all_tests_exit=$?
total_tests=$((total_tests + 1))

if [ $all_tests_exit -eq 0 ]; then
    passed_tests=$((passed_tests + 1))
elif [ $all_tests_exit -eq 5 ] || [ $all_tests_exit -eq 6 ] || [ $all_tests_exit -eq 9 ] || [ $all_tests_exit -eq 11 ]; then
    crashed_tests=$((crashed_tests + 1))
else
    failed_tests=$((failed_tests + 1))
fi

echo ""
echo "=================================================="
echo -e "${BLUE}📊 Summary:${NC}"
echo "Total tests: $total_tests"
echo -e "Passed: ${GREEN}$passed_tests${NC}"
echo -e "Failed: ${YELLOW}$failed_tests${NC}"
echo -e "Crashed: ${RED}$crashed_tests${NC}"
echo ""

# Calculate percentage
if [ $total_tests -gt 0 ]; then
    pass_percent=$((passed_tests * 100 / total_tests))
    echo -e "Success rate: ${GREEN}${pass_percent}%${NC}"
else
    echo "No tests run"
fi

echo ""

# Show detailed output for any crashed tests
if [ $crashed_tests -gt 0 ]; then
    echo -e "${RED}🚨 Crash Details:${NC}"
    echo "=================="
    echo ""
    
    # Re-run crashed tests with verbose output
    for test_file in "${test_files[@]}"; do
        cargo test --test "$test_file" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo -e "${RED}Crashed test details for: $test_file${NC}"
            run_test_verbose "$test_file"
        fi
    done
    
    # Check full test suite if it crashed
    if [ $all_tests_exit -ne 0 ]; then
        echo -e "${RED}Full test suite crash details:${NC}"
        echo "cargo test"
        cargo test 2>&1 | head -50  # Show first 50 lines of output
        echo ""
    fi
fi

echo "=================================================="
echo -e "${BLUE}💡 Usage Tips:${NC}"
echo "- Re-run this script: ./test_runner.sh"
echo "- Run specific test: cargo test --test <test_name>"
echo "- Run with output: cargo test --test <test_name> -- --nocapture"
echo "- Check binaries: cargo test --bin <binary_name>"
echo ""

# Exit with appropriate code
if [ $crashed_tests -gt 0 ]; then
    echo -e "${RED}⚠️  Some tests crashed - investigation needed${NC}"
    exit 1
elif [ $failed_tests -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Some tests failed - check test logic${NC}"
    exit 2
else
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
fi