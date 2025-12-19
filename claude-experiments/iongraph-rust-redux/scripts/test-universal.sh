#!/bin/bash

# Test script to verify Universal format produces identical output to TypeScript

echo "Testing Universal IR format compatibility..."
echo

passed=0
failed=0

for func in 0 1 2 3 4; do
  for pass in 0 10 20 30; do
    echo -n "Testing function $func, pass $pass... "

    # Generate Rust version (via Universal)
    ./target/release/generate_svg ion-examples/mega-complex.json $func $pass /tmp/rust-$func-$pass.svg 2>&1 >/dev/null

    # Generate TypeScript version
    node /Users/jimmyhmiller/Documents/Code/open-source/iongraph2/generate-svg-function.mjs ion-examples/mega-complex.json $func $pass /tmp/ts-$func-$pass.svg 2>&1 >/dev/null

    # Compare
    if diff -q /tmp/ts-$func-$pass.svg /tmp/rust-$func-$pass.svg >/dev/null 2>&1; then
      echo "✓ MATCH"
      ((passed++))
    else
      echo "✗ DIFFERENT"
      ((failed++))
    fi
  done
done

echo
echo "Results: $passed passed, $failed failed"

if [ $failed -eq 0 ]; then
  echo "🎉 All tests passed! Universal format produces identical output!"
  exit 0
else
  echo "❌ Some tests failed"
  exit 1
fi
