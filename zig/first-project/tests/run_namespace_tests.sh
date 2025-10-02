#!/bin/bash

echo "========================================="
echo "Namespace Compilation Test Suite"
echo "========================================="
echo ""

PASS=0
FAIL=0

# Helper function to run a test
run_test() {
    local test_name="$1"
    local lisp_file="$2"
    local expected_outputs="$3"
    
    echo "Running: $test_name"
    
    # Compile
    ./simple_c_compiler "$lisp_file" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "  ❌ FAIL: Lisp compilation failed"
        ((FAIL++))
        return 1
    fi
    
    # Get C filename
    c_file="${lisp_file%.lisp}.c"
    exe_file="${lisp_file%.lisp}"
    
    # Compile C (ignore warnings)
    gcc -o "$exe_file" "$c_file" 2>&1 | grep "error:" > /dev/null
    if [ $? -eq 0 ]; then
        echo "  ❌ FAIL: C compilation failed"
        gcc -o "$exe_file" "$c_file" 2>&1 | head -5
        ((FAIL++))
        return 1
    fi
    
    # Actually compile (quietly)
    gcc -o "$exe_file" "$c_file" 2>/dev/null
    
    # Run and check output
    actual_output=$(./"$exe_file")
    if [ "$actual_output" = "$expected_outputs" ]; then
        echo "  ✅ PASS"
        ((PASS++))
    else
        echo "  ❌ FAIL: Output mismatch"
        echo "     Expected: $expected_outputs"
        echo "     Got:      $actual_output"
        ((FAIL++))
    fi
}

# Test 1: Basic namespace variables
cat > test_basic_vars.lisp << 'LISP_EOF'
(def x (: Int) 10)
(def y (: Int) 20)
(+ x y)
LISP_EOF

run_test "Basic namespace variables" "test_basic_vars.lisp" "30"

# Test 2: Functions calling each other
cat > test_fn_calls.lisp << 'LISP_EOF'
(def add (: (-> [Int Int] Int))
  (fn [a b] (+ a b)))

(def sum3 (: (-> [Int Int Int] Int))
  (fn [a b c] (add a (add b c))))

(sum3 1 2 3)
LISP_EOF

run_test "Functions calling each other" "test_fn_calls.lisp" "6"

# Test 3: Recursive functions
cat > test_recursive.lisp << 'LISP_EOF'
(def fib (: (-> [Int] Int))
  (fn [n]
    (if (< n 2)
        n
        (+ (fib (- n 1)) (fib (- n 2))))))

(fib 10)
LISP_EOF

run_test "Recursive functions" "test_recursive.lisp" "55"

# Test 4: Explicit namespace
cat > test_explicit_ns.lisp << 'LISP_EOF'
(ns my.test)

(def value (: Int) 42)
value
LISP_EOF

run_test "Explicit namespace declaration" "test_explicit_ns.lisp" "42"

# Test 5: C keyword sanitization
cat > test_keywords.lisp << 'LISP_EOF'
(def double (: (-> [Int] Int))
  (fn [n] (+ n n)))

(def return (: Int) 21)

(+ return (double return))
LISP_EOF

run_test "C keyword sanitization" "test_keywords.lisp" "63"

# Test 6: Struct with namespace
cat > test_struct_ns.lisp << 'LISP_EOF'
(def Point (: Type)
  (Struct [x Int] [y Int]))

(def p (Point 3 4))
(def sum (: Int) (+ (. p x) (. p y)))
sum
LISP_EOF

run_test "Struct with namespace vars" "test_struct_ns.lisp" "7"

# Test 7: Mixed operations
cat > test_mixed.lisp << 'LISP_EOF'
(def x (: Int) 5)

(def square (: (-> [Int] Int))
  (fn [n] (* n n)))

(+ x (square x))
LISP_EOF

run_test "Mixed vars and functions" "test_mixed.lisp" "30"

echo ""
echo "========================================="
echo "Test Results: $PASS passed, $FAIL failed"
echo "========================================="

if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
