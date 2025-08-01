#!/bin/bash

# Simple build script for C++ projects
set -e

PROJECT_NAME="lang"
SRC_DIR="src"
BUILD_DIR="build"
CXX="c++"
CXXFLAGS="-std=c++20 -Wall -Wextra -g"
INCLUDES="-I${SRC_DIR}"

show_help() {
    echo "Usage: ./build.sh [command] [args...]"
    echo ""
    echo "Commands:"
    echo "  build [--release] - Build the project (debug by default, --release for optimized build)"
    echo "  test [test_name]  - Build and run tests (all tests or specific test by name)"
    echo "  stress            - Build and run stress tests (random input generation)"
    echo "  run               - Build (if needed) and run the project"
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
        echo "Building ${PROJECT_NAME} (release mode)..."
    else
        echo "Building ${PROJECT_NAME} (debug mode)..."
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
    
    if [ -n "$test_name" ]; then
        echo "Building and running individual test: $test_name"
    else
        echo "Building and running all tests..."
    fi
    
    mkdir -p "${BUILD_DIR}"
    
    # Build library sources (excluding main.cc, test files, and tools)
    lib_sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v main.cc | grep -v "/tools/" | sort)
    
    if [ -z "$lib_sources" ]; then
        echo "Error: No library source files found in ${SRC_DIR}"
        exit 1
    fi
    
    if [ -n "$test_name" ]; then
        # Build with individual test flag
        ${CXX} ${CXXFLAGS} ${INCLUDES} "-DRUN_INDIVIDUAL_TEST=\"$test_name\"" $lib_sources test_main.cc -o "${BUILD_DIR}/${PROJECT_NAME}_test"
    else
        # Build normally for all tests
        ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources test_main.cc -o "${BUILD_DIR}/${PROJECT_NAME}_test"
    fi
    
    echo "Running tests..."
    "${BUILD_DIR}/${PROJECT_NAME}_test"
}

cmd_stress() {
    echo "Building and running stress tests..."
    mkdir -p "${BUILD_DIR}"
    
    # Build library sources (excluding main.cc, test files, and tools)
    lib_sources=$(find "${SRC_DIR}" -name "*.cc" -o -name "*.cpp" | grep -v main.cc | grep -v "/tools/" | sort)
    
    if [ -z "$lib_sources" ]; then
        echo "Error: No library source files found in ${SRC_DIR}"
        exit 1
    fi
    
    ${CXX} ${CXXFLAGS} ${INCLUDES} $lib_sources stress_test.cc -o "${BUILD_DIR}/${PROJECT_NAME}_stress"
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
    
    echo "Building and running $tool_name..."
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