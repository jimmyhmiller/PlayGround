#!/bin/bash

# Test script to verify Universal format files can be rendered directly

echo "Testing Universal IR format direct rendering..."
echo

passed=0
failed=0

test_file() {
  local name=$1
  local file=$2

  echo -n "Testing $name... "
  if ./target/release/render_universal "$file" "/tmp/test-$(basename $file .json).svg" 2>&1 >/dev/null; then
    if [ -f "/tmp/test-$(basename $file .json).svg" ]; then
      echo "✓ SUCCESS"
      ((passed++))
    else
      echo "✗ FAILED (no output file)"
      ((failed++))
    fi
  else
    echo "✗ FAILED (render error)"
    ((failed++))
  fi
}

# Test simple universal example
test_file "simple-universal.json" "examples/simple-universal.json"

# Test large synthetic example
test_file "large-universal.json" "examples/large-universal.json"

# Test mega-complex conversions
test_file "mega-complex func11 pass0" "examples/mega-complex-func11-pass0-universal.json"
test_file "mega-complex func6 pass0" "examples/mega-complex-func6-pass0-universal.json"
test_file "mega-complex func8 pass5" "examples/mega-complex-func8-pass5-universal.json"

echo
echo "Results: $passed passed, $failed failed"

if [ $failed -eq 0 ]; then
  echo "🎉 All universal format tests passed!"
  exit 0
else
  echo "❌ Some tests failed"
  exit 1
fi
