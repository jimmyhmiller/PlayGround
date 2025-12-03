#!/bin/bash
# Test all examples in ion-examples directory

ION_EXAMPLES_DIR="/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/iongraph-rust-redux/ion-examples"
TS_SRC="/Users/jimmyhmiller/Documents/Code/open-source/iongraph2"
RUST_BIN="/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/iongraph-rust-redux/target/release/generate_svg"

echo "Testing all examples in ion-examples/..."
echo ""

total_tests=0
passed_tests=0
failed_tests=0

# Test each JSON file in ion-examples
for json_file in "$ION_EXAMPLES_DIR"/*.json; do
    filename=$(basename "$json_file")
    echo "Testing $filename..."

    # Generate TS version
    (cd "$TS_SRC" && node generate-svg-function.mjs "$json_file" 0 0 output.svg) > /dev/null 2>&1
    ts_result=$?

    if [ $ts_result -ne 0 ]; then
        echo "  ⚠️  TypeScript generation failed, skipping"
        continue
    fi

    # Copy TS output to comparison location
    cp "$TS_SRC/output.svg" /tmp/ts-${filename}.svg

    # Generate Rust version
    $RUST_BIN "$json_file" 0 0 /tmp/rust-${filename}.svg > /dev/null 2>&1
    rust_result=$?

    if [ $rust_result -ne 0 ]; then
        echo "  ❌ Rust generation failed"
        ((failed_tests++))
        ((total_tests++))
        continue
    fi

    # Compare outputs
    if diff -q /tmp/ts-${filename}.svg /tmp/rust-${filename}.svg > /dev/null 2>&1; then
        echo "  ✅ PASS"
        ((passed_tests++))
    else
        echo "  ❌ FAIL - outputs differ"
        # Show dimensions for debugging
        echo "     TypeScript dimensions: $(grep -o 'width="[0-9]*" height="[0-9]*"' /tmp/ts-${filename}.svg | head -1)"
        echo "     Rust dimensions:       $(grep -o 'width="[0-9]*" height="[0-9]*"' /tmp/rust-${filename}.svg | head -1)"
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
