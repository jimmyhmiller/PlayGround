#!/bin/bash

# Simple build script for C++ projects
set -e

PROJECT_NAME="lang"
SRC_DIR="src"
BUILD_DIR="build"
CXX="c++"
CXXFLAGS="-std=c++20 -Wall -Wextra -g"
INCLUDES="-I${SRC_DIR}"

# LLVM configuration
LLVM_CONFIG="/opt/homebrew/opt/llvm/bin/llvm-config"
if [ -x "$LLVM_CONFIG" ]; then
    LLVM_CXXFLAGS="$($LLVM_CONFIG --cxxflags)"
    LLVM_LDFLAGS="$($LLVM_CONFIG --ldflags --libs core executionengine mcjit interpreter native)"
    # Remove conflicting flags and adjust for our project
    LLVM_CXXFLAGS=$(echo "$LLVM_CXXFLAGS" | sed 's/-std=c++17/-std=c++20/g')
    LLVM_CXXFLAGS=$(echo "$LLVM_CXXFLAGS" | sed 's/-fno-exceptions//g')
    # Suppress warnings from LLVM headers using system includes
    LLVM_CXXFLAGS=$(echo "$LLVM_CXXFLAGS" | sed 's/-I/-isystem/g')
    echo "LLVM found: version $($LLVM_CONFIG --version)"
else
    echo "Warning: LLVM not found at $LLVM_CONFIG"
    LLVM_CXXFLAGS=""
    LLVM_LDFLAGS=""
fi

show_help() {
    echo "Usage: ./build.sh [command] [args...]"
    echo ""
    echo "Commands:"
    echo "  build [--release] - Build the project (debug by default, --release for optimized build)"
    echo "  test [test_name]  - Build and run tests (all tests or specific test by name)"
    echo "  stress            - Build and run stress tests (random input generation)"
    echo "  run               - Build (if needed) and run the project"
    echo "  ast-to-json       - Parse input from stdin and output AST as JSON"
    echo "  ast-to-code       - Parse input from stdin and generate code from AST"
    echo "  ast-to-llvm       - Parse input from stdin and generate LLVM IR"
    echo ""
    echo "  reader-repl       - Interactive REPL showing parsed structure"
    echo "  tokenizer-debug   - Show all tokens from input"
    echo "  tools [tool]      - Build and run a specific tool (run 'tools' with no args to see available tools)"
    echo "  fmt               - Format all C++ source files using clang-format"
    echo "  clean             - Clean build artifacts"
    echo "  help              - Show this help message"
    echo ""
    echo "Available individual tests:"
    echo "  test_reader_simple_numbers      - Test basic number parsing"
    echo "  test_reader_binary_operations   - Test binary operators"
    echo "  test_reader_operator_precedence - Test operator precedence"
    echo "  test_reader_right_associative   - Test right-associative operators"
    echo "  test_reader_unary_minus         - Test unary minus operator"
    echo "  test_reader_postfix_operator    - Test postfix operators"
    echo "  test_reader_complex_expression  - Test complex nested expressions"
    echo "  test_reader_multiple_expressions - Test multiple expressions"
    echo "  test_reader_node_equality       - Test node comparison"
    echo "  test_reader_empty_input         - Test empty input handling"
    echo "  test_reader_whitespace_handling - Test whitespace parsing"
    echo ""
    # Automatically discover and show available tools
    available_tools=$(find "${SRC_DIR}/tools" -name "*.cc" 2>/dev/null | sed 's|.*/||' | sed 's|\.cc$||' | sort)
    if [ -n "$available_tools" ]; then
        echo "Available tools:"
        for tool in $available_tools; do
            echo "  $tool"
        done
    else
        echo "No tools found in ${SRC_DIR}/tools"
    fi
}

cmd_clean() {
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    echo "Clean complete"
}

cmd_fmt() {
    echo "Formatting C++ source files with clang-format..."
    
    # Check if clang-format is available
    if ! command -v clang-format >/dev/null 2>&1; then
        echo "Error: clang-format not found. Please install clang-format."
        exit 1
    fi
    
    # Find all C++ source files (including tools and tests)
    files=$(find . -name "*.cc" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | grep -v "/build/" | sort)
    
    if [ -z "$files" ]; then
        echo "No C++ files found to format"
        return 0
    fi
    
    echo "Formatting files:"
    for file in $files; do
        echo "  $file"
        clang-format -i "$file"
    done
    
    echo "Formatting complete"
}

cmd_build() {
    local build_mode="debug"
    local cxxflags="${CXXFLAGS}"
    local output_suffix=""
    
    # Check for --release flag
    if [ "$1" = "--release" ]; then
        build_mode="release"
        cxxflags="-std=c++20 -Wall -Wextra -O3 -DNDEBUG"
        output_suffix="_release"
    fi
    
    mkdir -p "${BUILD_DIR}"
    
    # Find all .cc and .cpp files in src, excluding tests and tools
    sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v "/tools/" | sort)
    
    if [ -z "$sources" ]; then
        echo "Error: No source files found in ${SRC_DIR}"
        exit 1
    fi
    
    echo "Compiling (${build_mode}):"
    for src in $sources; do
        echo "  $src"
    done
    
    ${CXX} ${cxxflags} ${INCLUDES} $sources -o "${BUILD_DIR}/${PROJECT_NAME}${output_suffix}"
    echo "Build complete: ${BUILD_DIR}/${PROJECT_NAME}${output_suffix}"
}

cmd_test() {
    local test_name="$1"
    
    mkdir -p "${BUILD_DIR}"
    
    # Build library sources (excluding main.cc, test files, and tools)
    lib_sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v main.cc | grep -v "/tools/" | sort)
    
    if [ -z "$lib_sources" ]; then
        echo "Error: No library source files found in ${SRC_DIR}"
        exit 1
    fi
    
    # Test runner with improved reporting
    local failed_suites=0
    local total_suites=0
    local all_tests_passed=0
    local all_tests_failed=0
    local all_tests_total=0
    
    echo "=========================================="
    echo "Running Test Suite"
    echo "=========================================="
    
    # Function to parse test stats from output
    parse_test_stats() {
        local output="$1"
        local stats_line=$(echo "$output" | grep "TEST_STATS:" | tail -1)
        if [ -n "$stats_line" ]; then
            local passed=$(echo "$stats_line" | sed 's/.*passed=\([0-9]*\).*/\1/')
            local failed=$(echo "$stats_line" | sed 's/.*failed=\([0-9]*\).*/\1/')
            local total=$(echo "$stats_line" | sed 's/.*total=\([0-9]*\).*/\1/')
            all_tests_passed=$((all_tests_passed + passed))
            all_tests_failed=$((all_tests_failed + failed))
            all_tests_total=$((all_tests_total + total))
        fi
    }
    
    # Test 1: Reader tests
    total_suites=$((total_suites + 1))
    echo ""
    echo "[$total_suites] Reader Tests"
    echo "----------------------------------------"
    if [ -n "$test_name" ]; then
        # Build with individual test flag
        ${CXX} ${CXXFLAGS} ${INCLUDES} "-DRUN_INDIVIDUAL_TEST=\"$test_name\"" $lib_sources tests/test_reader.cc -o "${BUILD_DIR}/${PROJECT_NAME}_test"
    else
        # Build normally for all tests
        ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources tests/test_reader.cc -o "${BUILD_DIR}/${PROJECT_NAME}_test"
    fi
    
    local reader_output
    if reader_output=$("${BUILD_DIR}/${PROJECT_NAME}_test" 2>&1); then
        echo "$reader_output"
        echo "✅ Reader tests PASSED"
        parse_test_stats "$reader_output"
    else
        echo "$reader_output"
        echo "❌ Reader tests FAILED"
        failed_suites=$((failed_suites + 1))
    fi
    
    # Test 2: AST tests
    total_suites=$((total_suites + 1))
    echo ""
    echo "[$total_suites] AST Tests"
    echo "----------------------------------------"
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources tests/test_ast.cc -o "${BUILD_DIR}/test_ast"
    
    local ast_output
    if ast_output=$("${BUILD_DIR}/test_ast" 2>&1); then
        echo "$ast_output"
        echo "✅ AST tests PASSED"
        parse_test_stats "$ast_output"
    else
        echo "$ast_output"
        echo "❌ AST tests FAILED"
        failed_suites=$((failed_suites + 1))
    fi
    
    # Test 3: Parameter tests
    total_suites=$((total_suites + 1))
    echo ""
    echo "[$total_suites] Parameter Tests"
    echo "----------------------------------------"
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources test_parameters.cc -o "${BUILD_DIR}/test_parameters"
    
    local param_output
    if param_output=$("${BUILD_DIR}/test_parameters" 2>&1); then
        echo "$param_output"
        echo "✅ Parameter tests PASSED"
        parse_test_stats "$param_output"
    else
        echo "$param_output"
        echo "❌ Parameter tests FAILED"
        failed_suites=$((failed_suites + 1))
        parse_test_stats "$param_output"  # Parse stats even if suite failed
    fi
    
    # Test 4: Example syntax tests
    total_suites=$((total_suites + 1))
    echo ""
    echo "[$total_suites] Example Syntax Tests"
    echo "----------------------------------------"
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources tests/test_example_syntax.cc -o "${BUILD_DIR}/test_example_syntax"
    
    local example_output
    if example_output=$("${BUILD_DIR}/test_example_syntax" 2>&1); then
        echo "$example_output"
        echo "✅ Example syntax tests PASSED"
        parse_test_stats "$example_output"
    else
        echo "$example_output"
        echo "❌ Example syntax tests FAILED"
        failed_suites=$((failed_suites + 1))
        parse_test_stats "$example_output"  # Parse stats even if suite failed
    fi
    
    # Test 5: Identifier validation tests
    total_suites=$((total_suites + 1))
    echo ""
    echo "[$total_suites] Identifier Validation Tests"
    echo "----------------------------------------"
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources tests/test_identifier_validation.cc -o "${BUILD_DIR}/test_identifier_validation"
    
    local identifier_output
    if identifier_output=$("${BUILD_DIR}/test_identifier_validation" 2>&1); then
        echo "$identifier_output"
        echo "✅ Identifier validation tests PASSED"
        parse_test_stats "$identifier_output"
    else
        echo "$identifier_output"
        echo "❌ Identifier validation tests FAILED"
        failed_suites=$((failed_suites + 1))
        parse_test_stats "$identifier_output"  # Parse stats even if suite failed
    fi
    
    # Test 6: Tokenizer tests
    total_suites=$((total_suites + 1))
    echo ""
    echo "[$total_suites] Tokenizer Tests"
    echo "----------------------------------------"
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources tests/test_tokenizer.cc -o "${BUILD_DIR}/test_tokenizer"
    
    local tokenizer_output
    if tokenizer_output=$("${BUILD_DIR}/test_tokenizer" 2>&1); then
        echo "$tokenizer_output"
        echo "✅ Tokenizer tests PASSED"
        parse_test_stats "$tokenizer_output"
    else
        echo "$tokenizer_output"
        echo "❌ Tokenizer tests FAILED"
        failed_suites=$((failed_suites + 1))
        parse_test_stats "$tokenizer_output"  # Parse stats even if suite failed
    fi
    
    # Test 7: Error handling and edge case tests
    total_suites=$((total_suites + 1))
    echo ""
    echo "[$total_suites] Error Handling & Edge Case Tests"
    echo "----------------------------------------"
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources tests/test_error_handling.cc -o "${BUILD_DIR}/test_error_handling"
    
    local error_output
    if error_output=$("${BUILD_DIR}/test_error_handling" 2>&1); then
        echo "$error_output"
        echo "✅ Error handling tests PASSED"
        parse_test_stats "$error_output"
    else
        echo "$error_output"
        echo "❌ Error handling tests FAILED"
        failed_suites=$((failed_suites + 1))
        parse_test_stats "$error_output"  # Parse stats even if suite failed
    fi
    
    # Test 8: LLVM tests (only if LLVM is available)
    if [ -n "$LLVM_CXXFLAGS" ]; then
        total_suites=$((total_suites + 1))
        echo ""
        echo "[$total_suites] LLVM Backend Tests"
        echo "----------------------------------------"
        llvm_sources=$(find "llvm" -name "*.cc" -o -name "*.cpp" 2>/dev/null | sort)
        ${CXX} ${CXXFLAGS} ${LLVM_CXXFLAGS} ${INCLUDES} $lib_sources $llvm_sources tests/test_llvm.cc ${LLVM_LDFLAGS} -o "${BUILD_DIR}/test_llvm"
        
        local llvm_output
        if llvm_output=$("${BUILD_DIR}/test_llvm" 2>&1); then
            echo "$llvm_output"
            echo "✅ LLVM tests PASSED"
            parse_test_stats "$llvm_output"
        else
            echo "$llvm_output"
            echo "❌ LLVM tests FAILED"
            failed_suites=$((failed_suites + 1))
            parse_test_stats "$llvm_output"  # Parse stats even if suite failed
        fi
    else
        echo ""
        echo "[SKIPPED] LLVM Backend Tests (LLVM not available)"
    fi
    
    # Print final summary
    echo ""
    echo "=========================================="
    echo "Test Suite Summary"
    echo "=========================================="
    local passed_suites=$((total_suites - failed_suites))
    echo "Test suites run:    $total_suites"
    echo "Test suites passed: $passed_suites" 
    echo "Test suites failed: $failed_suites"
    echo ""
    echo "Individual Test Results:"
    echo "Tests run:    $all_tests_total"
    echo "Tests passed: $all_tests_passed"
    echo "Tests failed: $all_tests_failed"
    echo ""
    echo "NOTE: Above numbers reflect actual individual test cases executed."
    echo "See detailed output above for specific test failures and issues."
    
    if [ $failed_suites -eq 0 ] && [ $all_tests_failed -eq 0 ]; then
        echo ""
        echo "🎉 ALL TESTS PASSED!"
        echo "=========================================="
        return 0
    else
        echo ""
        if [ $failed_suites -gt 0 ]; then
            echo "💥 $failed_suites test suite(s) failed"
        fi
        if [ $all_tests_failed -gt 0 ]; then
            echo "💥 $all_tests_failed individual test(s) failed"
        fi
        echo "Review the detailed output above to see specific failing tests."
        echo "=========================================="
        return 1
    fi
}

cmd_stress() {
    mkdir -p "${BUILD_DIR}"
    
    # Build library sources (excluding main.cc, test files, and tools)
    lib_sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v main.cc | grep -v "/tools/" | sort)
    
    if [ -z "$lib_sources" ]; then
        echo "Error: No library source files found in ${SRC_DIR}"
        exit 1
    fi
    
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources tests/reader_stress_test.cc -o "${BUILD_DIR}/${PROJECT_NAME}_stress"
    echo "Running stress tests..."
    "${BUILD_DIR}/${PROJECT_NAME}_stress"
}

cmd_tools() {
    if [ $# -lt 2 ]; then
        echo "Error: tools command requires a tool name"
        # Automatically discover available tools
        available_tools=$(find "${SRC_DIR}/tools" -name "*.cc" 2>/dev/null | sed 's|.*/||' | sed 's|\.cc$||' | sort | tr '\n' ' ')
        if [ -n "$available_tools" ]; then
            echo "Available tools: $available_tools"
        else
            echo "No tools found in ${SRC_DIR}/tools"
        fi
        exit 1
    fi
    
    tool_name="$2"
    tool_file="${SRC_DIR}/tools/${tool_name}.cc"
    
    if [ ! -f "$tool_file" ]; then
        echo "Error: Tool '$tool_name' not found"
        # Show available tools
        available_tools=$(find "${SRC_DIR}/tools" -name "*.cc" 2>/dev/null | sed 's|.*/||' | sed 's|\.cc$||' | sort | tr '\n' ' ')
        if [ -n "$available_tools" ]; then
            echo "Available tools: $available_tools"
        else
            echo "No tools found in ${SRC_DIR}/tools"
        fi
        exit 1
    fi
    
    mkdir -p "${BUILD_DIR}"
    
    # Build library sources (excluding main.cc and tools)
    lib_sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v main.cc | grep -v "/tools/" | sort)
    
    if [ -z "$lib_sources" ]; then
        echo "Error: No library source files found in ${SRC_DIR}"
        exit 1
    fi
    
    # Build the tool
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources "$tool_file" -o "${BUILD_DIR}/${tool_name}"
    echo "Running $tool_name..."
    "${BUILD_DIR}/${tool_name}"
}

cmd_run() {
    if [ ! -f "${BUILD_DIR}/${PROJECT_NAME}" ]; then
        cmd_build
    fi
    
    echo "Running ${PROJECT_NAME}..."
    shift # Remove 'run' from arguments
    "${BUILD_DIR}/${PROJECT_NAME}" "$@"
}

cmd_ast_to_json() {
    mkdir -p "${BUILD_DIR}"
    
    # Build library sources (excluding main.cc and tools)
    lib_sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v main.cc | grep -v "/tools/" | sort)
    
    if [ -z "$lib_sources" ]; then
        echo "Error: No library source files found in ${SRC_DIR}"
        exit 1
    fi
    
    # Build the ast_to_json tool
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources "${SRC_DIR}/tools/ast_to_json.cc" -o "${BUILD_DIR}/ast_to_json"
    
    # Run the tool with stdin
    "${BUILD_DIR}/ast_to_json"
}

cmd_ast_to_code() {
    mkdir -p "${BUILD_DIR}"
    
    # Build library sources (excluding main.cc and tools)
    lib_sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v main.cc | grep -v "/tools/" | sort)
    
    if [ -z "$lib_sources" ]; then
        echo "Error: No library source files found in ${SRC_DIR}"
        exit 1
    fi
    
    # Build the ast_to_code tool
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources "${SRC_DIR}/tools/ast_to_code.cc" -o "${BUILD_DIR}/ast_to_code"
    
    # Run the tool with stdin
    "${BUILD_DIR}/ast_to_code"
}

cmd_ast_to_llvm() {
    mkdir -p "${BUILD_DIR}"
    
    # Extract arguments to pass to the tool (everything after the command)
    shift # Remove 'ast-to-llvm' from arguments
    tool_args="$@"
    
    # Check if LLVM is available
    if [ -z "$LLVM_CXXFLAGS" ]; then
        echo "Error: LLVM not found. Please install LLVM or update LLVM_CONFIG path."
        exit 1
    fi
    
    # Build library sources (excluding main.cc and tools)
    lib_sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v main.cc | grep -v "/tools/" | sort)
    
    # Add LLVM sources
    llvm_sources=$(find "llvm" -name "*.cc" -o -name "*.cpp" | sort)
    
    if [ -z "$lib_sources" ]; then
        echo "Error: No library source files found in ${SRC_DIR}"
        exit 1
    fi
    
    echo "Building ast_to_llvm tool with LLVM support..."
    
    # Build the ast_to_llvm tool with LLVM flags
    ${CXX} ${CXXFLAGS} ${LLVM_CXXFLAGS} ${INCLUDES} $lib_sources $llvm_sources "${SRC_DIR}/tools/ast_to_llvm.cc" ${LLVM_LDFLAGS} -o "${BUILD_DIR}/ast_to_llvm"
    
    # Run the tool with stdin and any additional arguments
    "${BUILD_DIR}/ast_to_llvm" $tool_args
}


cmd_reader_repl() {
    mkdir -p "${BUILD_DIR}"
    
    # Build library sources (excluding main.cc and tools)
    lib_sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v main.cc | grep -v "/tools/" | sort)
    
    if [ -z "$lib_sources" ]; then
        echo "Error: No library source files found in ${SRC_DIR}"
        exit 1
    fi
    
    # Build the reader_repl tool
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources "${SRC_DIR}/tools/reader_repl.cc" -o "${BUILD_DIR}/reader_repl"
    
    # Run the REPL with any additional arguments (skip the first one which is "reader-repl")
    shift
    "${BUILD_DIR}/reader_repl" "$@"
}

cmd_tokenizer_debug() {
    mkdir -p "${BUILD_DIR}"
    
    # Build library sources (excluding main.cc and tools)
    lib_sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v main.cc | grep -v "/tools/" | sort)
    
    if [ -z "$lib_sources" ]; then
        echo "Error: No library source files found in ${SRC_DIR}"
        exit 1
    fi
    
    # Build the tokenizer_debug tool
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources "${SRC_DIR}/tools/tokenizer_debug.cc" -o "${BUILD_DIR}/tokenizer_debug"
    
    # Run the tool
    "${BUILD_DIR}/tokenizer_debug"
}

# Main script
case ${1:-help} in
    build)
        cmd_build "$2"
        ;;
    test)
        cmd_test "$2"
        ;;
    stress)
        cmd_stress
        ;;
    ast-to-json)
        cmd_ast_to_json
        ;;
    ast-to-code)
        cmd_ast_to_code
        ;;
    ast-to-llvm)
        cmd_ast_to_llvm "$@"
        ;;
    reader-repl)
        cmd_reader_repl "$@"
        ;;
    tokenizer-debug)
        cmd_tokenizer_debug
        ;;
    tools)
        cmd_tools "$@"
        ;;
    run)
        cmd_run "$@"
        ;;
    fmt)
        cmd_fmt
        ;;
    clean)
        cmd_clean
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "Error: Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac