# Bugs

This file tracks bugs discovered during development.

## Bool type inference fails in def statements [organic-dark-ocelot]

**ID:** organic-dark-ocelot
**Timestamp:** 2025-10-11 00:19:55
**Severity:** high
**Location:** tests/integration/09_bool_test.lisp
**Tags:** typechecker, bool, inference

### Description

Test 09_bool_test fails. When defining a boolean value and using it in an if expression, type checking fails with TypeMismatch error expecting Int instead of Bool.

### Minimal Reproducing Case

Define: (def x (: Bool) true) (def result (: Int) (if x 1 0)) - Expected to work but fails with TypeMismatch

---

## Enum namespace qualification fails in basic usage [electric-best-damselfly]

**ID:** electric-best-damselfly
**Timestamp:** 2025-10-11 00:20:12
**Severity:** high
**Location:** tests/integration/11_enum_basic.lisp
**Tags:** enum, namespace, typechecker

### Description

Test 11_enum_basic fails. Using qualified enum names like Color/Red in type annotations causes compilation failures.

### Minimal Reproducing Case

(ns test) (def Color (: Type) (Enum Red Green Blue)) (def my_color (: Color) Color/Red)

---

## Array mutation with array-set! not working [inborn-illegal-krill]

**ID:** inborn-illegal-krill
**Timestamp:** 2025-10-11 00:20:24
**Severity:** high
**Location:** tests/integration/13_array_basic.lisp
**Tags:** arrays, mutation, array-set

### Description

Test 13_array_basic fails. Creating arrays and mutating them with array-set! appears to fail at runtime or during type checking.

### Minimal Reproducing Case

Create array with (array Int 3 0), then use (array-set! arr 0 10)

---

## Multi-dimensional arrays fail [motionless-royal-tahr]

**ID:** motionless-royal-tahr
**Timestamp:** 2025-10-11 00:20:35
**Severity:** medium
**Location:** tests/integration/16_array_multidim.lisp
**Tags:** arrays, multidimensional

### Description

Test 16_array_multidim fails. Creating and manipulating multi-dimensional arrays (Array (Array Int 2) 2) doesn't work properly.

### Minimal Reproducing Case

Define matrix: (def matrix (: (Array (Array Int 2) 2)) (array (Array Int 2) 2))

---

## C-style for loops not working [vapid-insubstantial-panda]

**ID:** vapid-insubstantial-panda
**Timestamp:** 2025-10-11 00:21:37
**Severity:** high
**Location:** tests/integration/21_c_for_loop.lisp
**Tags:** control-flow, c-for, mutation, set\!

### Description

Test 21_c_for_loop fails. c-for loops with mutations don't execute or compile correctly.

### Minimal Reproducing Case

(c-for [i (: Int) 0] (< i 10) (set\! i (+ i 1)) (set\! sum (+ sum i)))

---

## Nested loops fail [mad-overcooked-canid]

**ID:** mad-overcooked-canid
**Timestamp:** 2025-10-11 00:21:58
**Severity:** medium
**Location:** tests/integration/22_nested_loops.lisp
**Tags:** control-flow, c-for, nested, mutation

### Description

Test 22_nested_loops fails. Nested c-for loops don't work correctly, likely related to mutation or scoping issues.

### Minimal Reproducing Case

Nested c-for loops with mutations

---

## Higher-order functions fail [fitting-last-bear]

**ID:** fitting-last-bear
**Timestamp:** 2025-10-11 00:22:07
**Severity:** high
**Location:** tests/integration/24_higher_order.lisp
**Tags:** functions, higher-order, closures

### Description

Test 24_higher_order fails. Passing functions as arguments to other functions doesn't work correctly.

### Minimal Reproducing Case

(def apply_twice (: (-> [(-> [Int] Int) Int] Int)) (fn [f x] (f (f x))))

---

## Nested pointers (Pointer (Pointer T)) fail [unkempt-live-coral]

**ID:** unkempt-live-coral
**Timestamp:** 2025-10-11 00:22:35
**Severity:** medium
**Location:** tests/integration/28_nested_pointers.lisp
**Tags:** pointers, nested, types

### Description

Test 28_nested_pointers fails. Creating and dereferencing nested pointer types doesn't work.

### Minimal Reproducing Case

(def ptr2 (: (Pointer (Pointer Int))) (allocate (Pointer Int) ptr1))

---

## Let binding shadowing doesn't work correctly [damp-memorable-cricket]

**ID:** damp-memorable-cricket
**Timestamp:** 2025-10-11 00:23:03
**Severity:** medium
**Location:** tests/integration/32_let_shadowing.lisp
**Tags:** let, shadowing, scoping

### Description

Test 32_let_shadowing fails. Shadowing variables in nested let expressions produces wrong results.

### Minimal Reproducing Case

(def x (: Int) 100) (let [x (: Int) 10] (let [x (: Int) 20] x)) - Should return 20

---

## Boolean 'and' operator doesn't work as expression [hollow-high-crab]

**ID:** hollow-high-crab
**Timestamp:** 2025-10-11 00:23:11
**Severity:** high
**Location:** tests/integration/34_and_operator.lisp
**Tags:** operators, boolean, and

### Description

Test 34_and_operator fails. Using 'and' operator in expressions produces incorrect results.

### Minimal Reproducing Case

(if (and true true) 1 0) - Should return 1

---

## Manual gensym in macros produces invalid type annotation [superficial-regular-snail]

**ID:** superficial-regular-snail
**Timestamp:** 2025-10-11 00:23:50
**Severity:** high
**Location:** tests/integration/42_macro_gensym.lisp
**Tags:** macros, gensym, type-annotation

### Description

Test 42_macro_gensym fails. Using manual (gensym) in macro expansion creates invalid type annotation 'Symbol' which fails type checking.

### Minimal Reproducing Case

Macro using (let [temp (: Symbol) (gensym)] ...) fails with InvalidTypeAnnotation

### Code Snippet

```
(defmacro swap-add [a b] (let [temp (: Symbol) (gensym "temp")] ...))
```

---

## Structs containing arrays fail [boiling-smoggy-hedgehog]

**ID:** boiling-smoggy-hedgehog
**Timestamp:** 2025-10-11 00:24:09
**Severity:** medium
**Location:** tests/integration/43_struct_with_array.lisp
**Tags:** structs, arrays, fields

### Description

Test 43_struct_with_array fails. Defining structs with array fields and accessing them doesn't work correctly.

### Minimal Reproducing Case

(def Vec3 (: Type) (Struct [data (Array Int 3)])) - Struct field access of array fails

---

## Structs with pointer fields fail type checking [sophisticated-miserable-loon]

**ID:** sophisticated-miserable-loon
**Timestamp:** 2025-10-11 00:24:18
**Severity:** high
**Location:** tests/integration/44_struct_with_pointer.lisp
**Tags:** structs, pointers, fields, allocate

### Description

Test 44_struct_with_pointer fails with TypeMismatch error. Allocating structs that contain pointer fields produces type errors expecting struct but getting Type.

### Minimal Reproducing Case

(def Node (: Type) (Struct [value Int] [next (Pointer Node)])) (allocate Node n1) - TypeMismatch error

### Code Snippet

```
ERROR: TypeMismatch - expected: struct_type, actual: type_type
```

---

## Negative test err_04 expects TypeMismatch but test expects different error format [clever-insecure-rat]

**ID:** clever-insecure-rat
**Timestamp:** 2025-10-11 00:25:47
**Severity:** low
**Location:** tests/integration/err_04_wrong_type_in_if.error
**Tags:** tests, error-messages, negative-tests

### Description

Test err_04_wrong_type_in_if.error expects substring 'TypeMismatch' but actual error is 'mergeBranchTypes failed: error.TypeMismatch' which contains more context. Need to update expected error or error format.

### Minimal Reproducing Case

Test checks for 'TypeMismatch' but gets 'ERROR: mergeBranchTypes failed: error.TypeMismatch'

---

## Negative test err_05 expects UnboundVariable but gets different format [sunny-confused-mite]

**ID:** sunny-confused-mite
**Timestamp:** 2025-10-11 00:25:56
**Severity:** low
**Location:** tests/integration/err_05_undefined_function.error
**Tags:** tests, error-messages, negative-tests

### Description

Test err_05_undefined_function.error expects substring 'UnboundVariable' but actual error includes more context. Error message format doesn't match test expectations.

### Minimal Reproducing Case

Expected 'UnboundVariable' but error includes 'ERROR: Unbound variable: undefined_func'

---

## Negative test err_07 array index type mismatch format [cooperative-nutritious-panther]

**ID:** cooperative-nutritious-panther
**Timestamp:** 2025-10-11 00:26:05
**Severity:** low
**Location:** tests/integration/err_07_array_wrong_index_type.error
**Tags:** tests, error-messages, negative-tests, arrays

### Description

Test err_07_array_wrong_index_type.error expects 'TypeMismatch' but test framework expects exact substring match with error output.

### Minimal Reproducing Case

Test file expects 'TypeMismatch' substring in error

---

## Negative test err_08 pointer type mismatch format [woozy-snappy-canary]

**ID:** woozy-snappy-canary
**Timestamp:** 2025-10-11 00:26:31
**Severity:** low
**Location:** tests/integration/err_08_pointer_type_mismatch.error
**Tags:** tests, error-messages, negative-tests, pointers

### Description

Test err_08_pointer_type_mismatch.error expects 'TypeMismatch' but error output includes additional type information that may not match.

### Minimal Reproducing Case

Expected error substring 'TypeMismatch' vs actual 'ERROR: TypeMismatch - expected: pointer string, actual: pointer int'

---

## Negative test err_10 struct field type mismatch format [extroverted-remorseful-porpoise]

**ID:** extroverted-remorseful-porpoise
**Timestamp:** 2025-10-11 00:26:58
**Severity:** low
**Location:** tests/integration/err_10_struct_wrong_field_type.error
**Tags:** tests, error-messages, negative-tests, structs

### Description

Test err_10_struct_wrong_field_type.error expects 'TypeMismatch' substring but error output format may not match exactly.

### Minimal Reproducing Case

Expected 'TypeMismatch' vs 'ERROR: TypeMismatch - expected: int, actual: string'

---

## allocate-array, pointer-index-read, and pointer-index-write! operations fail [super-aged-grasshopper]

**ID:** super-aged-grasshopper
**Timestamp:** 2025-10-11 00:28:11
**Severity:** high
**Location:** tests/integration/17_heap_array.lisp (heap array operations)
**Tags:** arrays, heap, pointers, allocate-array, pointer-index-read, pointer-index-write!

### Description

Integration test 17_heap_array fails. The test allocates a heap array with allocate-array, writes values using pointer-index-write!, reads them with pointer-index-read, then deallocates. Expected output: 600 (sum of 100+200+300). Test fails at compilation or runtime.

### Minimal Reproducing Case

allocate-array Int 3 0; pointer-index-write! ptr 0 100; pointer-index-read ptr 0

### Code Snippet

```
Lines 1-10 of test file show full test case with array allocation, indexing, and deallocation
```

---

## Boolean 'not' operator produces incorrect negation results [illegal-harsh-lynx]

**ID:** illegal-harsh-lynx
**Timestamp:** 2025-10-11 00:28:57
**Severity:** high
**Location:** tests/integration/36_not_operator.lisp (not operator in boolean expressions)
**Tags:** operators, boolean, not, negation

### Description

Test 36_not_operator fails. Test evaluates (not true) and (not false) in if expressions. Expected output: '0 1' (false, true). Test produces wrong results, indicating 'not' operator logic is broken.

### Minimal Reproducing Case

Test (not true)→0, (not false)→1; actual results differ

### Code Snippet

```
2 test cases: not(true)→0, not(false)→1
```

---

## Structs with enum-typed fields [unwitting-jaded-cattle]

**ID:** unwitting-jaded-cattle
**Timestamp:** 2025-10-11 00:31:02
**Severity:** medium
**Location:** tests/integration/48_enum_struct.lisp
**Tags:** structs, enums, fields, struct-construction

### Description

Test 48_enum_struct.lisp defines ColoredPoint struct with Color enum field. Creating and using structs with enum fields fails, likely during struct construction or field access.

---

## Recursive functions with heap arrays and pointer indexing [excited-staid-sawfish]

**ID:** excited-staid-sawfish
**Timestamp:** 2025-10-11 00:31:29
**Severity:** medium
**Location:** tests/integration/50_complex_recursion.lisp
**Tags:** recursion, heap-arrays, pointer-index-read, functions

### Description

Test 50_complex_recursion.lisp implements recursive sum_array_recursive function using pointer-index-read on heap-allocated array. Combination of recursion, heap arrays, and pointer indexing fails. Expected sum: 15 (1+2+3+4+5).

---

