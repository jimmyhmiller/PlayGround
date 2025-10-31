#!/bin/bash
# Comprehensive syntax testing script
# Tests all verified syntax from PARSER_COMPARISON.md plus edge cases
# Usage: ./test_all_syntax.sh [--verbose]

set -e

VERBOSE=false
if [ "$1" = "--verbose" ]; then
    VERBOSE=true
fi

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
FAILURES=()

# Test a single expression
test_expr() {
    local expr="$1"
    local description="$2"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if $VERBOSE; then
        echo "Testing: $description"
        echo "  Expression: $expr"
    fi

    if ./compare_parsers.sh "$expr" > /dev/null 2>&1; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        if $VERBOSE; then
            echo "  ✅ PASS"
        else
            echo -n "."
        fi
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILURES+=("$description: $expr")
        if $VERBOSE; then
            echo "  ❌ FAIL"
        else
            echo -n "F"
        fi
    fi
}

echo "========================================"
echo "Comprehensive Pyret Parser Test Suite"
echo "========================================"
echo

# ============================================================================
# Category 1: Primitive Expressions
# ============================================================================

echo "Category 1: Primitives"

test_expr "42" "integer number"
test_expr "0" "zero"
test_expr "3.14" "decimal number"
test_expr "0.001" "small decimal"
test_expr "1000000" "large number"

test_expr "\"hello\"" "simple string"
test_expr "\"\"" "empty string"
test_expr "\"hello world\"" "string with spaces"
test_expr "\"123 abc\"" "string with numbers"

test_expr "true" "boolean true"
test_expr "false" "boolean false"

test_expr "x" "single letter identifier"
test_expr "myVariable" "camelCase identifier"
test_expr "foo123" "identifier with numbers"
test_expr "snake_case_var" "snake_case identifier"

echo

# ============================================================================
# Category 2: Binary Operators (All 15 operators)
# ============================================================================

echo "Category 2: Binary Operators"

# Arithmetic
test_expr "2 + 3" "addition"
test_expr "10 - 5" "subtraction"
test_expr "3 * 4" "multiplication"
test_expr "20 / 5" "division"

# Comparison
test_expr "x < 10" "less than"
test_expr "y > 5" "greater than"
test_expr "a <= b" "less than or equal"
test_expr "c >= d" "greater than or equal"
test_expr "x == y" "equality"
test_expr "a <> b" "inequality"

# Logical
test_expr "true and false" "logical and"
test_expr "x or y" "logical or"

# String
test_expr "\"hello\" ^ \" world\"" "string concatenation"

# Special
test_expr "x is y" "is operator"
test_expr "f(x) raises \"error\"" "raises operator"
test_expr "x satisfies pred" "satisfies operator"
test_expr "x violates pred" "violates operator"

echo

# ============================================================================
# Category 3: Left-Associativity (No Precedence!)
# ============================================================================

echo "Category 3: Left-Associativity"

test_expr "1 + 2 + 3" "chained addition (3 terms)"
test_expr "1 + 2 + 3 + 4 + 5" "chained addition (5 terms)"
test_expr "10 - 5 - 2" "chained subtraction"
test_expr "2 * 3 * 4" "chained multiplication"
test_expr "100 / 10 / 2" "chained division"

# Mixed operators (no precedence!)
test_expr "2 + 3 * 4" "mixed add/mult = (2+3)*4 = 20"
test_expr "10 - 5 + 3" "mixed sub/add"
test_expr "x * y + z" "mixed mult/add"
test_expr "a / b * c" "mixed div/mult"

# Comparisons
test_expr "x < y < z" "chained comparison"
test_expr "a == b == c" "chained equality"

# Logical
test_expr "a and b and c" "chained and"
test_expr "x or y or z" "chained or"

echo

# ============================================================================
# Category 4: Parenthesized Expressions
# ============================================================================

echo "Category 4: Parentheses"

test_expr "(42)" "simple paren number"
test_expr "(x)" "paren identifier"
test_expr "(2 + 3)" "paren binop"
test_expr "((5))" "double nested parens"
test_expr "(((x)))" "triple nested parens"
test_expr "((((1))))" "quad nested parens"

# Parens changing grouping
test_expr "1 + (2 * 3)" "parens change grouping"
test_expr "(1 + 2) * 3" "parens force precedence"
test_expr "a * (b + c)" "parens in second operand"
test_expr "(a + b) * (c + d)" "parens on both sides"

echo

# ============================================================================
# Category 5: Function Application
# ============================================================================

echo "Category 5: Function Calls"

test_expr "f(x)" "simple call"
test_expr "foo(bar)" "call with named args"
test_expr "f()" "no args call"
test_expr "f(x, y)" "two args"
test_expr "f(x, y, z)" "three args"
test_expr "f(a, b, c, d, e)" "five args"

# Expression arguments
test_expr "f(1 + 2)" "call with binop arg"
test_expr "g(x * y, a + b)" "call with multiple binop args"
test_expr "h((1), (2))" "call with paren args"

# Chained calls
test_expr "f(x)(y)" "two chained calls"
test_expr "f()()" "chained no-arg calls"
test_expr "f(1)(2)(3)" "three chained calls"
test_expr "f()(g())" "nested calls"
test_expr "f(g(h(x)))" "deeply nested calls"

echo

# ============================================================================
# Category 6: Whitespace Sensitivity
# ============================================================================

echo "Category 6: Whitespace Sensitivity"

test_expr "f(x)" "no space - direct call"
test_expr "f (x)" "with space - applied to paren"
test_expr "f(x, y)" "no space multiple args"
test_expr "f (x, y)" "with space - paren has args"

echo

# ============================================================================
# Category 7: Dot Access
# ============================================================================

echo "Category 7: Dot Access"

test_expr "obj.field" "simple dot access"
test_expr "x.y" "short dot access"
test_expr "obj.foo.bar" "chained dot (3 levels)"
test_expr "a.b.c.d" "chained dot (4 levels)"
test_expr "obj.field1.field2.field3.field4.field5" "long dot chain"

# Dot on function calls
test_expr "f(x).result" "dot on call result"
test_expr "getObject().field" "dot on call"
test_expr "f(x).foo.bar" "chained dot on call"

# Calls on dot access
test_expr "obj.method()" "call on dot (no args)"
test_expr "obj.foo(x)" "call on dot (one arg)"
test_expr "obj.method(x, y)" "call on dot (multiple args)"

echo

# ============================================================================
# Category 8: Mixed Postfix Operators
# ============================================================================

echo "Category 8: Mixed Postfix (Dot + Call)"

test_expr "f(x).result" "call then dot"
test_expr "obj.foo(a, b).bar" "dot, call, dot"
test_expr "x.y().z" "dot, call, dot"
test_expr "a().b().c()" "call, dot, call, dot, call"
test_expr "obj.foo.bar.baz" "all dots"
test_expr "f()()(g()())" "all calls"
test_expr "a.b().c.d()" "alternating dot/call"
test_expr "obj.method().result.value" "method chaining"

echo

# ============================================================================
# Category 9: Array Expressions
# ============================================================================

echo "Category 9: Arrays"

test_expr "[]" "empty array"
test_expr "[1]" "single element"
test_expr "[1, 2, 3]" "multiple numbers"
test_expr "[x, y, z]" "multiple identifiers"
test_expr "[true, false]" "array of booleans"
test_expr "[\"a\", \"b\"]" "array of strings"

# Nested arrays
test_expr "[[1, 2], [3, 4]]" "2D array"
test_expr "[[], []]" "array of empty arrays"
test_expr "[[[[1]]]]" "deeply nested array"

# Arrays with expressions
test_expr "[1 + 2, 3 * 4]" "array with binops"
test_expr "[f(x), g(y)]" "array with calls"
test_expr "[obj.foo, obj.bar]" "array with dot access"
test_expr "[(1 + 2), (3 * 4)]" "array with parens"

echo

# ============================================================================
# Category 10: Complex Mixed Expressions
# ============================================================================

echo "Category 10: Complex Combinations"

# Calls in binops
test_expr "f(x) + g(y)" "calls in addition"
test_expr "obj.foo() * obj.bar()" "dot calls in mult"
test_expr "f(a + b) + g(c * d)" "complex call args in binop"

# Dots in binops
test_expr "obj.x + obj.y" "dots in addition"
test_expr "a.b * c.d" "dots in mult"
test_expr "obj.field1 + obj.field2 + obj.field3" "chained dots in binops"

# Nested complexity
test_expr "f(x + 1).result" "call with binop arg, then dot"
test_expr "obj.method(a * b).field" "dot, call with binop, dot"
test_expr "(f(x) + g(y)).value" "binop of calls in parens, then dot"

# Kitchen sink
test_expr "f(x).foo(y).bar" "call, dot, call, dot"
test_expr "obj.method(1 + 2).result.value" "full chain with binop"

echo

# ============================================================================
# Category 11: Edge Cases - Deeply Nested
# ============================================================================

echo "Category 11: Edge Cases - Deep Nesting"

test_expr "((((((5))))))" "6-level nested parens"
test_expr "[[[[[[1]]]]]]" "6-level nested arrays"
test_expr "f(g(h(i(j(k(x))))))" "6-level nested calls"
test_expr "obj.a.b.c.d.e.f.g.h.i.j" "10-level dot chain"

echo

# ============================================================================
# Category 12: Edge Cases - Long Expressions
# ============================================================================

echo "Category 12: Edge Cases - Long Expressions"

# Generate long chains
test_expr "1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10" "10-term addition"
test_expr "a and b and c and d and e and f" "6-term logical and"

# Long arrays
test_expr "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]" "15-element array"

echo

# ============================================================================
# Category 13: Real-World Patterns
# ============================================================================

echo "Category 13: Real-World Patterns"

test_expr "obj.foo().bar().baz()" "method chaining"
test_expr "builder.setX(1).setY(2).build()" "builder pattern"
test_expr "data.filter(pred).map(transform)" "functional pipeline"
test_expr "(a + b) * (c - d)" "arithmetic expression"
test_expr "x / (y + z)" "division with grouping"
test_expr "point.x * point.x + point.y * point.y" "distance formula"

echo

# ============================================================================
# Summary
# ============================================================================

echo
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo

if [ $FAILED_TESTS -gt 0 ]; then
    echo "Failed tests:"
    for failure in "${FAILURES[@]}"; do
        echo "  ❌ $failure"
    done
    echo
    exit 1
else
    echo "✅ All tests passed!"
    exit 0
fi
