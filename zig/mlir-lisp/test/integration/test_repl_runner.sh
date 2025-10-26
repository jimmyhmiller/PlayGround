#!/bin/bash
set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go to the project root (two levels up from test/integration)
PROJECT_ROOT="$SCRIPT_DIR/../.."
MLIR_LISP="$PROJECT_ROOT/zig-out/bin/mlir_lisp"

echo "=== Testing REPL help command ==="
echo ":help
:quit" | "$MLIR_LISP" --repl 2>&1 | grep -q "REPL Commands" && echo "✓ PASS" || echo "✗ FAIL"

echo ""
echo "=== Testing simple constant auto-execution ==="
echo "(operation
  (name arith.constant)
  (result-bindings [%x])
  (result-types i32)
  (attributes { :value (: 42 i32) }))
:quit" | "$MLIR_LISP" --repl 2>&1 | grep -q "Result:" && echo "✓ PASS" || echo "✗ FAIL"

echo ""
echo "=== Testing function definition ==="
echo "(operation
  (name func.func)
  (attributes {
    :sym_name @test
    :function_type (!function (inputs) (results i64))
  })
  (regions
    (region
      (block [^entry]
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%c42])
          (result-types i64)
          (attributes { :value (: 42 i64) }))
        (operation
          (name func.return)
          (operands %c42))))))
:quit" | "$MLIR_LISP" --repl 2>&1 | grep -q "Function defined" && echo "✓ PASS" || echo "✗ FAIL"

echo ""
echo "=== Testing :mlir command ==="
echo "(operation (name func.func) (attributes { :sym_name @test :function_type (!function (inputs) (results i64)) }) (regions (region (block [^entry] (arguments []) (operation (name arith.constant) (result-bindings [%c42]) (result-types i64) (attributes { :value (: 42 i64) })) (operation (name func.return) (operands %c42))))))
:mlir
:quit" | "$MLIR_LISP" --repl 2>&1 | grep -q "@test" && echo "✓ PASS" || echo "✗ FAIL"

echo ""
echo "=== Testing :clear command ==="
echo "(operation
  (name func.func)
  (attributes {
    :sym_name @test
    :function_type (!function (inputs) (results i64))
  })
  (regions
    (region
      (block [^entry]
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%c42])
          (result-types i64)
          (attributes { :value (: 42 i64) }))
        (operation
          (name func.return)
          (operands %c42))))))
:clear
:mlir
:quit" | "$MLIR_LISP" --repl 2>&1 | grep -q "No module compiled yet" && echo "✓ PASS" || echo "✗ FAIL"

echo ""
echo "=== Testing unbalanced parentheses error ==="
echo "(operation))
:quit" | "$MLIR_LISP" --repl 2>&1 | grep -q "Unbalanced" && echo "✓ PASS" || echo "✗ FAIL"

echo ""
echo "=== All tests complete ==="
