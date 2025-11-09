#!/bin/bash

# Test script for mlir-lisp examples

echo "Testing .lisp examples..."
echo "========================"
echo ""

working=()
broken=()

# Test each .lisp file
while IFS= read -r file; do
    echo -n "Testing $file ... "
    if ./zig-out/bin/mlir_lisp "$file" > /dev/null 2>&1; then
        echo "✓ WORKING"
        working+=("$file")
    else
        echo "✗ BROKEN"
        broken+=("$file")
    fi
done < <(find examples -name "*.lisp" -type f | sort)

# Test .mlisp files
echo ""
echo "Testing .mlisp examples..."
echo "=========================="
echo ""

while IFS= read -r file; do
    echo -n "Testing $file ... "
    if ./zig-out/bin/mlir_lisp "$file" > /dev/null 2>&1; then
        echo "✓ WORKING"
        working+=("$file")
    else
        echo "✗ BROKEN"
        broken+=("$file")
    fi
done < <(find examples -name "*.mlisp" -type f | sort)

# Print summary
echo ""
echo "================================"
echo "SUMMARY"
echo "================================"
echo ""
echo "WORKING (${#working[@]}):"
for file in "${working[@]}"; do
    echo "  ✓ $file"
done

echo ""
echo "BROKEN (${#broken[@]}):"
for file in "${broken[@]}"; do
    echo "  ✗ $file"
done
