#!/bin/bash
set -e
cd "$(dirname "$0")"

COMPILER_CP="stages/stage0:asm.jar"
PASS=0
FAIL=0
ERRORS=""

for test_file in tests/*.jrs; do
    name=$(basename "$test_file" .jrs)
    expected="tests/${name}.expected"

    if [ ! -f "$expected" ]; then
        echo "SKIP $name (no .expected file)"
        continue
    fi

    # Compile
    rm -rf output && mkdir -p output
    if ! java -cp "$COMPILER_CP" Main "$test_file" > /dev/null 2>&1; then
        echo "FAIL $name (compile error)"
        FAIL=$((FAIL + 1))
        ERRORS="$ERRORS\n  $name: compile error"
        continue
    fi

    # Run
    actual=$(java -cp output Main 2>&1) || true

    if [ "$actual" = "$(cat "$expected")" ]; then
        echo "PASS $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL $name"
        FAIL=$((FAIL + 1))
        ERRORS="$ERRORS\n  $name:"
        ERRORS="$ERRORS\n    expected: $(cat "$expected" | head -3)"
        ERRORS="$ERRORS\n    actual:   $(echo "$actual" | head -3)"
    fi
done

echo ""
echo "$PASS passed, $FAIL failed"
if [ $FAIL -gt 0 ]; then
    echo -e "Failures:$ERRORS"
    exit 1
fi
