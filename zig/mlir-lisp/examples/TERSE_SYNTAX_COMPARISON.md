# Terse vs Verbose Syntax Comparison

This document shows side-by-side comparisons of verbose and terse MLIR-Lisp syntax.

## Example 1: Simple Constant Return

### Verbose (`simple_test.lisp`)
```lisp
(operation
  (name func.func)
  (attributes {}
    :sym_name @main
    :function_type (!function (inputs) (results i64)))

  (regions
    (region
      (block
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%result])
          (result-types i64)
          (attributes { :value (: 42 i64)}))
        (operation
          (name func.return)
          (operands %result))))))
```

### Terse (`simple_test_terse.lisp`)
```lisp
(operation
  (name func.func)
  (attributes {}
    :sym_name @main
    :function_type (!function (inputs) (results i64)))

  (regions
    (region
      (block
        (arguments [])
        (declare result (arith.constant {:value (: 42 i64)}))
        (func.return %result)))))
```

**Savings:** 18 lines → 12 lines (33% reduction)

---

## Example 2: Addition Function

### Verbose (`add.lisp`)
```lisp
(operation
  (name func.func)
  (attributes {}
    :sym_name @main
    :function_type (!function (inputs) (results i64)))

  (regions
    (region
      (block [^entry]
        (arguments [])

        ;; Create constant 10
        (operation
          (name arith.constant)
          (result-bindings [%c10])
          (result-types i64)
          (attributes { :value (: 10 i64)}))

        ;; Create constant 32
        (operation
          (name arith.constant)
          (result-bindings [%c32])
          (result-types i64)
          (attributes { :value (: 32 i64)}))

        ;; Add them together
        (operation
          (name arith.addi)
          (result-bindings [%result])
          (result-types i64)
          (operands %c10 %c32))

        ;; Return the result
        (operation
          (name func.return)
          (operands %result))))))
```

### Terse (`add_terse.lisp`)
```lisp
(operation
  (name func.func)
  (attributes {}
    :sym_name @main
    :function_type (!function (inputs) (results i64)))

  (regions
    (region
      (block [^entry]
        (arguments [])

        ;; Create constants using terse syntax - type inferred from :value attribute
        (declare c10 (arith.constant {:value (: 10 i64)}))
        (declare c32 (arith.constant {:value (: 32 i64)}))

        ;; Add them together - result type inferred from operands
        (declare result (arith.addi %c10 %c32))

        ;; Return the result
        (func.return %result)))))
```

**Savings:** 39 lines → 23 lines (41% reduction)

---

## Example 3: Multiple Arithmetic Operations

### Terse Only (`terse_demo.lisp`)
```lisp
;; Demonstration of terse operation syntax

(mlir
  ;; Declare constants with inferred types from attributes
  (declare c1 (arith.constant {:value (: 42 i64)}))
  (declare c2 (arith.constant {:value (: 10 i64)}))

  ;; Arithmetic operations with type inferred from operands
  (declare sum (arith.addi %c1 %c2))
  (declare product (arith.muli %sum %c2))

  ;; Operations with attributes
  (declare c3 (arith.constant {:value (: 100 i64)}))

  ;; More arithmetic
  (declare difference (arith.subi %c3 %product)))
```

### Generated MLIR (identical for verbose or terse)
```mlir
module {
  %c42_i64 = arith.constant 42 : i64
  %c10_i64 = arith.constant 10 : i64
  %0 = arith.addi %c42_i64, %c10_i64 : i64
  %1 = arith.muli %0, %c10_i64 : i64
  %c100_i64 = arith.constant 100 : i64
  %2 = arith.subi %c100_i64, %1 : i64
}
```

---

## Key Differences

### Verbose Syntax
- Explicit `(operation (name op.name) ...)` wrapper
- Explicit `result-bindings` and `result-types` sections
- More verbose, but very explicit
- 5-6 lines per operation on average

### Terse Syntax
- Direct operation name: `(op.name {attrs} operands...)`
- `declare` form for named results: `(declare name expr)`
- Type inference from attributes and operands
- 1-2 lines per operation on average
- **33-41% reduction in line count**

### What's Inferred

1. **For `arith.constant`:**
   - Result type inferred from `:value` attribute type
   - Example: `{:value (: 42 i64)}` → result type is `i64`

2. **For arithmetic operations (`arith.addi`, `arith.subi`, etc.):**
   - Result type inferred from operand types
   - Example: `(arith.addi %c1 %c2)` where `%c1` and `%c2` are `i64` → result is `i64`

3. **Variable names:**
   - `(declare my-var expr)` → creates SSA value `%my-var`
   - Automatically prepends `%` to create proper value ID

---

## Benefits of Terse Syntax

1. ✅ **Less boilerplate** - 33-41% fewer lines
2. ✅ **More readable** - Focus on what operations do, not structure
3. ✅ **Type inference** - Compiler figures out obvious types
4. ✅ **Named values** - Declare gives semantic names inline
5. ✅ **Backward compatible** - Verbose syntax still works
6. ✅ **Identical output** - Generates exactly the same MLIR

---

## When to Use Each

### Use Terse Syntax When:
- Writing hand-crafted MLIR-Lisp code
- Prototyping and experimenting
- Operations have clear type inference paths
- Readability is important

### Use Verbose Syntax When:
- Generating code programmatically
- Operations need explicit type annotations
- Working with complex type hierarchies
- Debugging type issues
- Interfacing with external tools
