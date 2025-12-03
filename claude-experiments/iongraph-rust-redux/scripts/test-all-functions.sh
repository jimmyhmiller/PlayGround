#!/bin/bash
# Test all functions in mega-complex.json

INPUT_FILE="/Users/jimmyhmiller/Documents/Code/open-source/iongraph2/examples/mega-complex.json"
TS_SRC="/Users/jimmyhmiller/Documents/Code/open-source/iongraph2"

echo "Testing all functions from mega-complex.json..."
echo ""

total_tests=0
passed_tests=0
failed_tests=0

# Test each function's first pass (pass 0)
for func_idx in {0..14}; do
    echo "Testing function $func_idx (pass 0)..."

    # Generate TS version
    cd "$TS_SRC"
    node generate-svg-function.mjs examples/mega-complex.json $func_idx 0 output.svg > /dev/null 2>&1
    ts_result=$?

    if [ $ts_result -ne 0 ]; then
        echo "  ⚠️  TypeScript generation failed, skipping"
        continue
    fi

    # Copy TS output to comparison location
    cp output.svg /tmp/ts-func${func_idx}-pass0.svg

    # Generate Rust version
    cd - > /dev/null
    node generate-svg-function.mjs "$INPUT_FILE" $func_idx 0 /tmp/rust-func${func_idx}-pass0.svg > /dev/null 2>&1
    rust_result=$?

    if [ $rust_result -ne 0 ]; then
        echo "  ❌ Rust generation failed"
        ((failed_tests++))
        ((total_tests++))
        continue
    fi

    # Compare outputs
    if diff -q /tmp/ts-func${func_idx}-pass0.svg /tmp/rust-func${func_idx}-pass0.svg > /dev/null 2>&1; then
        echo "  ✅ PASS"
        ((passed_tests++))
    else
        echo "  ❌ FAIL - outputs differ"
        # Show first few lines of diff
        echo "     First difference:"
        diff /tmp/ts-func${func_idx}-pass0.svg /tmp/rust-func${func_idx}-pass0.svg | head -5 | sed 's/^/     /'
        ((failed_tests++))
    fi
    ((total_tests++))
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
