#!/bin/bash
# Test case for bug uncommon-unwieldy-emu
# Unit attributes with true value not being emitted in MLIR output

echo "Testing unit attribute emission bug..."
echo ""

# Compile the test file
ACTUAL=$(./zig-out/bin/mlir_lisp examples/bugs/uncommon-unwieldy-emu_unit_attrs.mlir-lisp -g 2>&1 | grep -A 6 '"builtin.module"')

echo "=== Actual Output ==="
echo "$ACTUAL"
echo ""

echo "=== Expected Output ==="
cat examples/bugs/uncommon-unwieldy-emu_expected.mlir
echo ""

# Check for missing unit attributes
if echo "$ACTUAL" | grep -q "constant"; then
    echo "✓ 'constant' attribute found"
else
    echo "✗ 'constant' attribute MISSING"
fi

if echo "$ACTUAL" | grep -q "dso_local"; then
    echo "✓ 'dso_local' attribute found"
else
    echo "✗ 'dso_local' attribute MISSING"
fi

if echo "$ACTUAL" | grep -q "no_unwind"; then
    echo "✓ 'no_unwind' attribute found"
else
    echo "✗ 'no_unwind' attribute MISSING"
fi
