# Fibonacci Example: Terse Syntax Conversion

This document shows the conversion of the most complex example (`fibonacci.mlir-lisp`) from verbose to terse syntax, demonstrating what currently works and what features are still needed.

---

## File Comparison

### Original (Verbose)
- **File:** `fibonacci.mlir-lisp`
- **Lines:** 158
- **Status:** ✅ Works perfectly, JIT compiles and returns 55

### Simplified Terse (Constants Only)
- **File:** `fibonacci_simplified_terse.lisp`
- **Lines:** 128
- **Reduction:** 30 lines (19% reduction)
- **Status:** ✅ Works perfectly, JIT compiles and returns 55
- **Uses:** Only terse constants with `declare`

### Partial Terse (Attempted)
- **File:** `fibonacci_partial_terse.lisp`
- **Lines:** 132
- **Reduction:** 26 lines (16% reduction)
- **Status:** ❌ Has errors (Unknown Value issues)
- **Problem:** Some operations don't have full type inference yet

### Future Full Terse
- **File:** `fibonacci_terse.lisp`
- **Lines:** ~40 (estimated)
- **Reduction:** 118 lines (75% reduction!)
- **Status:** ⏳ Waiting for additional features
- **Requires:** let, regions, scf.if, implicit yields/returns

---

## Side-by-Side: Original vs Simplified Terse

### Verbose (158 lines)
```lisp
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @fibonacci
      :function_type (!function (inputs i32) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [ (: %n i32) ])

          ;; Check if n <= 1
          (operation
            (name arith.constant)
            (result-bindings [%c1])
            (result-types i32)
            (attributes { :value (: 1 i32) }))

          (operation
            (name arith.cmpi)
            (result-bindings [%cond])
            (result-types i1)
            (operands %n %c1)
            (attributes { :predicate (: 3 i64) }))

          ;; ... continues for 100+ more lines
```

### Simplified Terse (128 lines)
```lisp
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @fibonacci
      :function_type (!function (inputs i32) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [ (: %n i32) ])

          ;; ✅ TERSE: Check if n <= 1
          (declare c1 (arith.constant {:value (: 1 i32)}))

          (operation
            (name arith.cmpi)
            (result-bindings [%cond])
            (result-types i1)
            (operands %n %c1)
            (attributes { :predicate (: 3 i64) }))

          ;; ... 80+ more lines
```

**Savings:** 30 lines (19%) just from using terse constants!

---

## What Currently Works ✅

### 1. Terse Constants with Declare

**Before:**
```lisp
(operation
  (name arith.constant)
  (result-bindings [%c1])
  (result-types i32)
  (attributes { :value (: 1 i32) }))
```

**After:**
```lisp
(declare c1 (arith.constant {:value (: 1 i32)}))
```

**Benefits:**
- 5 lines → 1 line (80% reduction)
- Type automatically inferred from `:value` attribute
- Name automatically gets `%` prefix
- Much more readable

### 2. Type Inference for Constants

The result type is automatically inferred from the typed attribute value:
- `{:value (: 42 i64)}` → result type is `i64`
- `{:value (: 1 i32)}` → result type is `i32`

### 3. Optional Attributes

Can write `(arith.constant {:value 42})` or `(arith.constant {} {:value 42})` - both work!

---

## What Doesn't Work Yet ❌

### 1. Arithmetic Operations Without Explicit Types

**Attempted:**
```lisp
(declare n_minus_1 (arith.subi %n %c1))
```

**Problem:** Type inference for `arith.subi` from operands doesn't work in all contexts

**Workaround:** Use verbose syntax:
```lisp
(operation
  (name arith.subi)
  (result-bindings [%n_minus_1])
  (result-types i32)
  (operands %n %c1))
```

### 2. Function Calls

**Attempted:**
```lisp
(declare fib_result (func.call {:callee @fibonacci} %n))
```

**Problem:** No type inference for `func.call` - result type can't be determined

**Workaround:** Use verbose syntax with explicit `result-types`

### 3. Comparison Operations

**Attempted:**
```lisp
(declare cond (arith.cmpi {:predicate sle} %n %c1))
```

**Problem:** No type inference for `arith.cmpi` - always returns `i1` but not implemented

**Workaround:** Use verbose syntax with explicit `result-types i1`

### 4. Type Conversions

**Attempted:**
```lisp
(declare result_i64 (arith.extsi %fib_result))
```

**Problem:** Target type needs to be specified, no inference

**Workaround:** Use verbose syntax with explicit `result-types i64`

---

## What's Missing for Full Terse Syntax

To achieve the projected 75% reduction (~40 lines), we need:

### 1. Let Bindings ⭐ CRITICAL

**Needed Syntax:**
```lisp
(let [(: c1 (arith.constant {:value 1}))
      (: n_minus_1 (arith.subi {} n c1))
      (: fib1 (func.call {:callee @fibonacci} n_minus_1))]
  ;; body that uses bindings
  )
```

**Benefits:**
- Scoped variables (no `%` prefix needed)
- Sequential evaluation
- Cleaner than multiple `declare` statements

### 2. Terse func.func Syntax ⭐ CRITICAL

**Needed Syntax:**
```lisp
(func.func {:sym_name fibonacci :function_type (-> (i32) (i32))}
  [(: n i32)]  ;; function arguments
  body)
```

**Current:** Must use verbose `(operation (name func.func) ...)`

### 3. Terse scf.if with Regions ⭐ CRITICAL

**Needed Syntax:**
```lisp
(scf.if {} cond
  (region  ;; Then
    n)     ;; Return n directly
  (region  ;; Else
    (let [...] ...)))
```

**Current:** Must use verbose `(operation (name scf.if) (regions ...))`

### 4. Implicit scf.yield ⭐ HIGH PRIORITY

**Needed Syntax:**
```lisp
(region
  (let [(: result (compute-something))]
    result))  ;; Automatically yields result
```

**Current:** Must explicitly write `(operation (name scf.yield) (operands %result))`

### 5. Implicit func.return ⭐ HIGH PRIORITY

**Needed Syntax:**
```lisp
(func.func ...
  body)  ;; Last expression automatically returned
```

**Current:** Must explicitly write `(operation (name func.return) (operands %result))`

### 6. Better Type Inference 🔶 MEDIUM PRIORITY

**Needed:**
- Infer `i1` for comparison operations
- Infer result types for `func.call` from function signature
- Infer target type for conversions when possible

**Current:** Only `arith.constant` and binary arithmetic have inference

---

## Projected Full Terse Version (~40 lines)

```lisp
;; Fibonacci with FULL terse syntax (NOT YET IMPLEMENTED)

(mlir
  ;; Fibonacci function
  (func.func {:sym_name fibonacci :function_type (-> (i32) (i32))}
    [(: n i32)]

    (let [(: c1 (arith.constant {:value 1}))
          (: cond (arith.cmpi {:predicate sle} n c1))]

      (scf.if {} cond
        (region n)  ;; Base case: return n
        (region  ;; Recursive case
          (let [(: c1_rec (arith.constant {:value 1}))
                (: n_minus_1 (arith.subi {} n c1_rec))
                (: fib1 (func.call {:callee @fibonacci} n_minus_1))
                (: c2 (arith.constant {:value 2}))
                (: n_minus_2 (arith.subi {} n c2))
                (: fib2 (func.call {:callee @fibonacci} n_minus_2))]
            (arith.addi {} fib1 fib2))))))

  ;; Main function
  (func.func {:sym_name main :function_type (-> () (i64))}
    []
    (let [(: n (arith.constant {:value 10}))
          (: fib_result (func.call {:callee @fibonacci} n))
          (: result_i64 (arith.extsi {} fib_result))]
      result_i64)))
```

**Estimated: 40 lines (75% reduction from 158 lines!)**

---

## Implementation Priority

To get to full terse syntax, implement in this order:

1. **Let bindings** - Foundation for scoped variables
2. **func.func terse syntax** - Functions are essential
3. **scf.if with regions** - Control flow is critical
4. **Implicit yields/returns** - Sugar that makes it readable
5. **Better type inference** - Removes remaining verbose operations

---

## Current Achievement

✅ **19% line reduction** using only terse constants
✅ **Identical MLIR output** - generates same code
✅ **JIT compiles and runs** - returns correct result (55)

**With full terse syntax: 75% reduction projected**

---

## Testing

```bash
# Original verbose version (158 lines)
./zig-out/bin/mlir_lisp examples/fibonacci.mlir-lisp
# Output: Result: 55 ✅

# Simplified terse version (128 lines)
./zig-out/bin/mlir_lisp examples/fibonacci_simplified_terse.lisp
# Output: Result: 55 ✅

# Both generate identical MLIR and produce same result!
```

---

## Conclusion

Even with just terse constants (the simplest feature), we achieve **19% code reduction** while maintaining full compatibility and correctness.

Implementing the remaining features (let, func.func, scf.if, implicit returns) would enable **75% reduction** - turning 158 lines into ~40 lines of much more readable code!

The terse syntax makes MLIR-Lisp dramatically more approachable while maintaining the full power and precision of MLIR.
