#!/bin/bash
# Test which ignored tests might now pass

echo "Testing ignored tests one by one..."
echo

# List of test names to try
tests=(
    "test_block_with_multiple_let_bindings"
    "test_block_with_var_binding"
    "test_block_with_typed_bindings"
    "test_nested_blocks_with_shadowing"
    "test_unary_not"
    "test_unary_minus"
)

passing=0
failing=0

for test in "${tests[@]}"; do
    echo "Testing: $test"
    if cargo test --test comparison_tests "$test" -- --ignored --nocapture 2>&1 | grep -q "test result: ok"; then
        echo "  ✅ PASSED"
        ((passing++))
    else
        echo "  ❌ FAILED"
        ((failing++))
    fi
    echo
done

echo "========================================="
echo "Results: $passing passing, $failing failing"
echo "========================================="
