#!/bin/bash
# Test multiple passes for a few functions

INPUT_FILE="/Users/jimmyhmiller/Documents/Code/open-source/iongraph2/examples/mega-complex.json"
TS_SRC="/Users/jimmyhmiller/Documents/Code/open-source/iongraph2"

echo "Testing multiple passes for selected functions..."
echo ""

total_tests=0
passed_tests=0
failed_tests=0

# Test function 5 (the one we debugged) across multiple passes
for pass_idx in {0..5}; do
    echo "Testing function 5, pass $pass_idx..."

    # Generate TS version
    (cd "$TS_SRC" && node generate-svg-function.mjs examples/mega-complex.json 5 $pass_idx output.svg) > /dev/null 2>&1
    ts_result=$?

    if [ $ts_result -ne 0 ]; then
        echo "  ⚠️  TypeScript generation failed, skipping"
        continue
    fi

    # Copy TS output
    cp "$TS_SRC/output.svg" /tmp/ts-func5-pass${pass_idx}.svg

    # Generate Rust version
    (cd "$TS_SRC" && node generate-svg-function.mjs examples/mega-complex.json 5 $pass_idx /tmp/rust-func5-pass${pass_idx}.svg) > /dev/null 2>&1
    rust_result=$?

    if [ $rust_result -ne 0 ]; then
        echo "  ❌ Rust generation failed"
        ((failed_tests++))
        ((total_tests++))
        continue
    fi

    # Compare
    if diff -q /tmp/ts-func5-pass${pass_idx}.svg /tmp/rust-func5-pass${pass_idx}.svg > /dev/null 2>&1; then
        echo "  ✅ PASS"
        ((passed_tests++))
    else
        echo "  ❌ FAIL - outputs differ"
        ((failed_tests++))
    fi
    ((total_tests++))
done

# Test a few other functions with different passes
for func_idx in 0 6 14; do
    for pass_idx in 0 10 20; do
        echo "Testing function $func_idx, pass $pass_idx..."

        (cd "$TS_SRC" && node generate-svg-function.mjs examples/mega-complex.json $func_idx $pass_idx output.svg) > /dev/null 2>&1
        ts_result=$?

        if [ $ts_result -ne 0 ]; then
            echo "  ⚠️  TypeScript generation failed, skipping"
            continue
        fi

        cp "$TS_SRC/output.svg" /tmp/ts-func${func_idx}-pass${pass_idx}.svg

        (cd "$TS_SRC" && node generate-svg-function.mjs examples/mega-complex.json $func_idx $pass_idx /tmp/rust-func${func_idx}-pass${pass_idx}.svg) > /dev/null 2>&1
        rust_result=$?

        if [ $rust_result -ne 0 ]; then
            echo "  ❌ Rust generation failed"
            ((failed_tests++))
            ((total_tests++))
            continue
        fi

        if diff -q /tmp/ts-func${func_idx}-pass${pass_idx}.svg /tmp/rust-func${func_idx}-pass${pass_idx}.svg > /dev/null 2>&1; then
            echo "  ✅ PASS"
            ((passed_tests++))
        else
            echo "  ❌ FAIL"
            ((failed_tests++))
        fi
        ((total_tests++))
    done
done

echo ""
echo "========================================="
echo "Results: $passed_tests/$total_tests passed"
if [ $failed_tests -eq 0 ]; then
    echo "🎉 All tests passed!"
    exit 0
else
    echo "⚠️  $failed_tests tests failed"
    exit 1
fi
