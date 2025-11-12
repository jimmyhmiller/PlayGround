# Terse Regions Guide

## Overview

The terse syntax now supports **generic region handling** for ANY operation with regions, including `scf.if`, `scf.for`, `scf.while`, and any custom MLIR operations that use regions.

## Key Features

### 1. **Explicit Region Syntax**
Use `(region ...)` to define regions in terse operations:

```lisp
(scf.if %cond
  (region then-expr)
  (region else-expr))
```

### 2. **Implicit Yield/Return Insertion**
The parser automatically inserts appropriate terminators (`scf.yield`, `func.return`, etc.) at the end of regions:

```lisp
;; BEFORE (what you write):
(region
  (declare x (arith.addi %a %b))
  %x)

;; AFTER (what the parser generates):
(region
  (declare x (arith.addi %a %b))
  (scf.yield %x))    ;; ← Automatically inserted!
```

### 3. **Bare Value IDs**
You can use bare value IDs in regions - they're automatically yielded:

```lisp
(scf.if %cond
  (region %n)           ;; ← Becomes (scf.yield %n)
  (region %default))    ;; ← Becomes (scf.yield %default)
```

### 4. **Generic - Works for Any Operation**
This isn't just for `scf.if` - it works for **any MLIR operation with regions**:

- `scf.if`, `scf.for`, `scf.while`
- `func.func` (though currently uses verbose syntax for top-level)
- Any custom dialects with region-based operations

## Examples

### Example 1: Simple Conditional

```lisp
;; Verbose syntax:
(operation
  (name scf.if)
  (result-bindings [%result])
  (result-types i32)
  (operands %cond)
  (regions
    (region
      (block
        (arguments [])
        (operation
          (name scf.yield)
          (operands %val))))
    (region
      (block
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%c0])
          (result-types i32)
          (attributes {:value (: 0 i32)}))
        (operation
          (name scf.yield)
          (operands %c0))))))

;; Terse syntax with regions:
(declare result (:
  (scf.if %cond
    (region %val)
    (region
      (declare c0 (arith.constant {:value (: 0 i32)}))
      %c0))
  i32))
```

**Reduction**: 21 lines → 7 lines (67% reduction!)

### Example 2: Nested Operations in Regions

```lisp
(declare result (:
  (scf.if %cond
    (region
      (declare doubled (arith.muli %val %val))
      %doubled)             ;; Implicit (scf.yield %doubled)
    (region
      (declare c1 (arith.constant {:value (: 1 i32)}))
      (declare incremented (arith.addi %val %c1))
      %incremented))        ;; Implicit (scf.yield %incremented)
  i32))
```

### Example 3: Fibonacci (Real-World Usage)

```lisp
;; Check if n <= 1
(declare c1 (arith.constant {:value (: 1 i32)}))
(declare cond (: (arith.cmpi {:predicate (: 3 i64)} %n %c1) i1))

;; Terse scf.if with regions and implicit yields
(declare result (:
  (scf.if %cond
    (region %n)              ;; Base case: return n
    (region                   ;; Recursive case: fib(n-1) + fib(n-2)
      (declare c1_rec (arith.constant {:value (: 1 i32)}))
      (declare n_minus_1 (arith.subi %n %c1_rec))
      (declare fib_n_minus_1 (: (func.call {:callee @fibonacci} %n_minus_1) i32))

      (declare c2 (arith.constant {:value (: 2 i32)}))
      (declare n_minus_2 (arith.subi %n %c2))
      (declare fib_n_minus_2 (: (func.call {:callee @fibonacci} %n_minus_2) i32))

      (declare sum (arith.addi %fib_n_minus_1 %fib_n_minus_2))
      %sum))                ;; Implicit (scf.yield %sum)
  i32))
```

## Type Annotations

Currently, operations with regions in `declare` forms require **explicit type annotations**:

```lisp
;; ✓ CORRECT - explicit type annotation
(declare result (:
  (scf.if %cond
    (region %then_val)
    (region %else_val))
  i32))

;; ✗ INCORRECT - type inference not yet supported for scf.if
(declare result
  (scf.if %cond
    (region %then_val)
    (region %else_val)))
```

This is because type inference requires building the regions first, which creates a chicken-and-egg problem. Type inference from yielded values may be added in a future version.

## Terminator Selection

The parser automatically selects the appropriate terminator based on the parent operation:

- `scf.*` operations → `scf.yield`
- `func.*` operations → `func.return`
- Unknown operations → `scf.yield` (default)

## How It Works

### 1. Parser Extension
`parseTerseOperation()` now detects `(region ...)` forms mixed with operands:

```zig
(scf.if %cond          // Operand: %cond
  (region ...)         // Region 1
  (region ...))        // Region 2
```

### 2. Region Parsing
`parseTerseRegionWithParent()` handles terse regions:
- Creates an implicit block with no arguments
- Parses operations (or bare value IDs)
- Inserts appropriate terminator if missing

### 3. Implicit Terminator Insertion
`insertImplicitTerminator()` checks the last operation:
- If it's already a terminator → do nothing
- If it has result bindings → create `yield` with those results
- If no operations → create `yield` with no operands

## Advantages

1. **Concise**: 40-70% code reduction for control flow
2. **Generic**: Works for any operation with regions
3. **Consistent**: Same pattern for all region-based operations
4. **Safe**: Automatic terminator insertion prevents errors
5. **Clear**: Explicit `(region ...)` markers show structure

## Comparison: Verbose vs Terse

| Verbose | Terse | Reduction |
|---------|-------|-----------|
| 15+ lines | 3 lines | 80% |
| Explicit blocks | Implicit blocks | ✓ |
| Explicit yields | Implicit yields | ✓ |
| Explicit terminators | Auto-inserted | ✓ |

## Files

See these examples:
- `examples/test_scf_if_terse.lisp` - Basic examples
- `examples/test_scf_if_terse_simple.lisp` - Simplest case
- `examples/fibonacci_fully_terse.lisp` - Real-world usage

## Future Enhancements

Potential improvements for future versions:

1. **Type inference for scf.if**: Infer result type from yielded values
2. **Implicit regions**: `(scf.if %cond then else)` without `(region ...)`
3. **Terse func.func**: Top-level function definitions with terse regions
4. **Block arguments**: Support for region blocks with arguments

## Related

- See `docs/terse-syntax-spec.md` for the full specification
- See `TERSE_SYNTAX_SUMMARY.md` for overall terse syntax features
