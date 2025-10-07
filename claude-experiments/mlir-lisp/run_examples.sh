#!/bin/bash

# Script to run all example files

echo "Building project..."
cargo build --release 2>&1 | grep -E "(Compiling|Finished)" || true
echo ""

failed=0
passed=0

for example in examples/*.lisp; do
    if [ -f "$example" ]; then
        basename=$(basename "$example")
        printf "%-30s ... " "$basename"

        output=$(cargo run --release "$example" 2>&1)
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            result=$(echo "$output" | grep "Execution result:" | sed 's/.*result: //' || echo "")
            if [ -n "$result" ]; then
                echo "✅ PASS (result: $result)"
            else
                echo "✅ PASS"
            fi
            ((passed++))
        else
            echo "❌ FAIL"
            echo "$output" | grep "Error:" || true
            ((failed++))
        fi
    fi
done

echo ""
echo "========================================="
echo "Summary: $passed passed, $failed failed"
echo "========================================="

if [ $failed -gt 0 ]; then
    exit 1
fi
