# MLIR-Lisp Syntax Reference
**Version:** 1.0
**Date:** 2025-01-10
**Status:** Authoritative Reference

This document is the **single, canonical reference** for all MLIR-Lisp syntax variants. It consolidates the verbose syntax, terse syntax, and all implementation details into one authoritative source.

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Lexical Tokens](#lexical-tokens)
4. [Type System](#type-system)
5. [Verbose Syntax (Canonical)](#verbose-syntax-canonical)
6. [Terse Syntax (Modern)](#terse-syntax-modern)
7. [Attribute Syntax](#attribute-syntax)
8. [Regions and Control Flow](#regions-and-control-flow)
9. [Operation Flattening](#operation-flattening)
10. [Type Inference](#type-inference)
11. [Macro System](#macro-system)
12. [Complete Examples](#complete-examples)
13. [Grammar Summary](#grammar-summary)

---

## Overview

MLIR-Lisp provides three syntax layers for representing MLIR operations:

1. **Verbose Syntax** - Explicit, self-documenting, round-trip safe
2. **Terse Syntax** - Concise, functional, with type inference
3. **Pretty Syntax** - High-level surface syntax (design spec only, not implemented)

All syntax variants compile to identical MLIR IR. The verbose syntax is always available as an escape hatch and for generated code.

---

## Design Principles

### Core Philosophy

1. **Concise** - Minimize boilerplate without losing clarity
2. **Functional** - Expression-oriented with scoped bindings
3. **Type-safe** - Explicit annotations where needed, inference where possible
4. **MLIR-native** - Direct mapping to MLIR constructs, no abstraction loss
5. **Macro-friendly** - Built from composable primitives
6. **Round-trip safe** - Can parse and regenerate identical output

### Key Design Decisions

- **Symbols not strings** - Use `@func` not `"func"` for symbol references
- **Uniform type annotations** - Always `(: value type)` format
- **Scoped bindings** - Variables are locally scoped within blocks/regions
- **Implicit terminators** - Last expression in region is automatically yielded/returned
- **No loss of information** - Verbose syntax preserves all MLIR details

---

## Lexical Tokens

### Identifiers
```
Pattern: [A-Za-z_+*/<>=!?&|-][A-Za-z0-9_.$:+*/<>=!?&|-]*
```

**Examples:**
```lisp
add              ;; Simple identifier
my-func          ;; Hyphenated (Lisp style)
+                ;; Operator as identifier
>>=              ;; Multi-character operator
foo.bar          ;; Dotted identifier
```

**Note:** The following characters at the start are reserved for special syntax:
- `%` - SSA value IDs
- `^` - Block labels
- `@` - Symbol references
- `#` - Attribute markers
- `:` - Keywords
- `!` - Dialect types

### SSA Value IDs
```
Pattern: % + (number | identifier)
```

**Examples:**
```lisp
%0               ;; Anonymous value
%arg0            ;; Named argument
%result          ;; Named result
%c10             ;; Constant binding
```

### Block Labels
```
Pattern: ^ + (number | identifier)
```

**Examples:**
```lisp
^bb0             ;; Numbered block
^entry           ;; Named entry block
^loop            ;; Loop block
^exit            ;; Exit block
```

### Symbol References
```
Pattern: @ + (number | identifier)
```

**Examples:**
```lisp
@main            ;; Function name
@fibonacci       ;; User-defined function
@42              ;; Numeric symbol (valid)
```

### Keywords (Attribute Keys)
```
Pattern: : + identifier (with optional dots)
```

**Examples:**
```lisp
:sym_name                ;; Function name attribute
:value                   ;; Value attribute
:predicate               ;; Comparison predicate
:llvm.linkage            ;; Dotted keyword for LLVM
```

### Attribute Markers (Dialect Attributes)
```
Pattern: # + opaque-text-until-delimiter
```

Uses **bracket-aware scanning** to handle complex nested attributes with spaces:
- Tracks depth of `<>` and `()` brackets
- Handles string literals `"..."` (ignores brackets inside strings)
- Stops at whitespace/delimiters `{`, `}`, `[`, `]`, `;` only when bracket depth is 0

**Examples:**
```lisp
#arith.overflow<none>
#llvm.linkage<internal>
#llvm.noalias
#dlti.dl_spec<i1 = dense<8> : vector<2xi64>, i64 = dense<64>>
#attr<"key" = "value with spaces">
```

---

## Type System

### Builtin Types

Simple identifiers (no `!` prefix):

```lisp
;; Integer types
i1 i8 i16 i32 i64 i128

;; Floating-point types
f16 f32 f64

;; Index type (machine-word sized integer)
index
```

### Dialect Types

Require `!` prefix. Uses **bracket-aware scanning** for complex types:

```lisp
;; Simple dialect types
!llvm.ptr                           ;; LLVM pointer
!transform.any_op                   ;; Transform dialect
!pdl.operation                      ;; PDL operation type
!pdl.type                           ;; PDL type

;; Complex types with spaces and nested brackets
!llvm.array<10 x i8>               ;; Array with dimensions
!llvm.array<20 x f32>              ;; Float array
!llvm.struct<(i32, i64)>           ;; Struct with fields
!llvm.struct<(i32, array<5 x f32>)> ;; Nested types
!llvm.ptr<270>                      ;; Pointer with address space
!llvm.func<ptr (ptr, ptr)>         ;; Function type

;; Multi-dimensional types
!llvm.array<10 x array<20 x i32>>  ;; 2D array
```

### Function Types

**Verbose syntax:**
```lisp
(!function (inputs TYPE*) (results TYPE*))
```

**Examples:**
```lisp
;; Single input, single output
(!function (inputs i32) (results i32))

;; Multiple inputs
(!function (inputs i32 i64 f32) (results i64))

;; Multiple outputs
(!function (inputs i32) (results i32 i1))

;; No inputs
(!function (inputs) (results i32))

;; No outputs (void)
(!function (inputs i32 i64) (results))
```

**Terse syntax (in spec, not yet implemented):**
```lisp
;; Arrow notation
(-> (i32) (i32))
(-> (i32 i64 f32) (i64))
(-> (i32) (i32 i1))
(-> () (i32))
(-> (i32 i64) ())
```

### Type Annotations

**Uniform syntax:** `(: value type)`

```lisp
;; Function arguments
[(: %arg0 i32) (: %arg1 i64)]

;; Attribute values (typed literals)
{:value (: 42 i32)}
{:value (: 1.23 f32)}

;; Block arguments
(arguments [(: %arg0 i32) (: %iter index)])

;; Explicit type on expression (terse syntax)
(: (arith.addi {} a b) i32)

;; Let bindings (terse syntax spec)
(let [(: result (arith.addi {} a b))]
  result)
```

### Tensor/Vector Types

```lisp
tensor<10xi32>              ;; 10-element i32 tensor
vector<4xf32>               ;; 4-element f32 vector
tensor<*xi32>               ;; Unranked i32 tensor
memref<5xf32>               ;; Memory reference
memref<?xf32>               ;; Dynamic dimension
memref<*xf32>               ;; Unranked memref
```

---

## Verbose Syntax (Canonical)

The verbose syntax is **explicit, self-documenting, and round-trip safe**. It serves as the canonical representation and is always supported.

### Top-Level Structure

```lisp
(mlir TOP_LEVEL_ITEM*)

TOP_LEVEL_ITEM ::= TYPE_ALIAS | OPERATION
```

### Type Aliases

```lisp
(type-alias TYPE_ID STRING)

;; Examples:
(type-alias !my_vec "vector<4xf32>")
(type-alias !my_tensor "tensor<10x20xf32>")
```

### Operation Form

```lisp
(operation
  (name OP_NAME)
  SECTION*)

SECTION ::= RESULT_BINDINGS
          | RESULT_TYPES
          | OPERANDS
          | ATTRIBUTES
          | SUCCESSORS
          | REGIONS
          | LOCATION
```

**All sections are optional and order-flexible.**

### Sections

#### Result Bindings
```lisp
(result-bindings [ VALUE_ID* ])
```

**Examples:**
```lisp
(result-bindings [%result])
(result-bindings [%sum %overflow])
(result-bindings [])  ;; No results
```

#### Result Types
```lisp
(result-types TYPE*)
```

**Examples:**
```lisp
(result-types i32)
(result-types i32 i1)
(result-types)  ;; No results (void operation)
```

#### Operands
```lisp
(operands VALUE_ID*)
```

**Examples:**
```lisp
(operands %x %y)
(operands %arg0 %arg1 %arg2)
(operands)  ;; No operands
```

#### Attributes
```lisp
(attributes { KEYWORD VALUE* })
```

**Examples:**
```lisp
(attributes {:sym_name @main})
(attributes {:value (: 42 i32)})
(attributes {:predicate (: 3 i64)})
(attributes {:callee @fibonacci})
(attributes {})  ;; No attributes
```

#### Successors
```lisp
(successors SUCCESSOR*)

SUCCESSOR ::= (successor BLOCK_ID (VALUE_ID*))
```

**Examples:**
```lisp
(successors
  (successor ^then (%x))
  (successor ^else (%y)))

(successors
  (successor ^bb1)
  (successor ^bb2 (%arg1 %arg2)))
```

#### Regions
```lisp
(regions REGION*)

REGION ::= (region BLOCK+)

BLOCK ::= (block
            [ BLOCK_ID ]
            (arguments [ (: VALUE_ID TYPE)* ])
            OPERATION*)
```

**Examples:**
```lisp
(regions
  (region
    (block [^entry]
      (arguments [(: %arg0 i32) (: %arg1 i32)])
      (operation ...)
      (operation ...))))
```

### Complete Verbose Example

```lisp
;; Simple addition function
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @add
      :function_type (!function (inputs i32 i32) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [(: %x i32) (: %y i32)])
          (operation
            (name arith.addi)
            (result-bindings [%sum])
            (result-types i32)
            (operands %x %y))
          (operation
            (name func.return)
            (operands %sum)))))))
```

---

## Terse Syntax (Modern)

The terse syntax is **concise, functional, and designed for human readability**. It removes boilerplate while maintaining full MLIR expressiveness.

### Implementation Status

**✅ Currently Implemented:**
- Terse operations: `(op.name {attrs?} operands...)`
- Declare form: `(declare name expr)`
- Type inference: `arith.constant`, binary arithmetic operations
- Terse regions: `(region ...)` with implicit terminators
- Implicit yield/return insertion

**⏳ Not Yet Implemented (from spec):**
- Let bindings: `(let [bindings...] body)`
- Terse func.func: `(func.func {...} [args] body)`
- Do form: `(do expr1 expr2 ...)`
- Inline successors: `^{:key1 expr1 :key2 expr2}`
- Arrow function types: `(-> (inputs) (results))`
- Extended type inference (comparisons, function calls)

### Terse Operations

**Pattern:** `(op.name {attributes?} operands...)`

Detected by presence of `.` in operation name.

**Examples:**
```lisp
;; Binary arithmetic (attributes optional)
(arith.addi %a %b)
(arith.muli %x %y)

;; With attributes
(arith.constant {:value (: 42 i64)})
(func.call {:callee @my-func} %arg1 %arg2)
(arith.cmpi {:predicate (: 3 i64)} %n %c1)

;; No operands
(func.return %result)
(scf.yield)
```

### Declare Form

**Pattern:** `(declare name expr)`

Auto-prepends `%` to create SSA value IDs.

**Examples:**
```lisp
;; Simple constant
(declare c10 (arith.constant {:value (: 10 i64)}))
;; Creates: %c10 = arith.constant 10 : i64

;; With type inference
(declare sum (arith.addi %c1 %c2))
;; Creates: %sum = arith.addi %c1, %c2 : i64

;; Explicit type annotation
(declare result (: (func.call {:callee @foo} %arg) i32))
;; Creates: %result = func.call @foo(%arg) : i32
```

**Key Point:** Variables in `declare` are plain identifiers. The `%` is automatically added.

### Terse Regions

**Pattern:** `(region BODY...)`

- Used with operations that have regions: `scf.if`, `scf.for`, `func.func`, etc.
- **Generic** - works with ANY operation that has regions
- Implicit terminator insertion based on parent operation

**Examples:**
```lisp
;; Simple region with bare value
(scf.if %cond
  (region %then_val)      ;; Implicit: (scf.yield %then_val)
  (region %else_val))     ;; Implicit: (scf.yield %else_val)

;; Region with operations
(scf.if %cond
  (region
    (declare doubled (arith.muli %val %val))
    %doubled)            ;; Implicit: (scf.yield %doubled)
  (region
    (declare c0 (arith.constant {:value (: 0 i32)}))
    %c0))               ;; Implicit: (scf.yield %c0)
```

**Type annotation requirement:**
Currently, operations with regions in `declare` forms require **explicit type annotations**:

```lisp
;; ✓ CORRECT - explicit type
(declare result (:
  (scf.if %cond
    (region %val1)
    (region %val2))
  i32))

;; ✗ INCORRECT - inference not yet supported
(declare result
  (scf.if %cond
    (region %val1)
    (region %val2)))
```

### Implicit Terminators

The parser automatically inserts appropriate terminators at the end of regions:

| Parent Operation | Inserted Terminator |
|-----------------|---------------------|
| `scf.*`         | `scf.yield`        |
| `func.*`        | `func.return`      |
| Unknown         | `scf.yield` (default) |

**Rules:**
1. If region already ends with a terminator → do nothing
2. If last operation has result bindings → yield those results
3. If bare value ID at end → yield that value
4. If region is empty → yield with no operands

### Terse Syntax Comparison

**Verbose (21 lines):**
```lisp
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
```

**Terse (7 lines):**
```lisp
(declare result (:
  (scf.if %cond
    (region %val)
    (region
      (declare c0 (arith.constant {:value (: 0 i32)}))
      %c0))
  i32))
```

**67% reduction in code size!**

---

## Attribute Syntax

Attributes are key-value pairs used to configure operations.

### Attribute Values

```lisp
;; Integers (untyped)
{:value 42}

;; Typed integers (typed literals)
{:value (: 42 i32)}
{:value (: 10 i64)}
{:predicate (: 3 i64)}

;; Floats
{:value 3.14}
{:value (: 1.23 f32)}

;; Booleans
{:some_flag true}
{:other_flag false}

;; Symbols (not strings!)
{:sym_name @my-function}
{:callee @fibonacci}

;; Keywords (for enum-like values)
{:predicate slt}
{:predicate eq}

;; Arrays
{:case_values [0 1 2]}
{:operandSegmentSizes [1 2 4]}

;; Dialect attributes (with # marker)
{:linkage #llvm.linkage<internal>}
{:frame_pointer #llvm.framePointerKind<none>}
{:overflow #arith.overflow<none>}

;; Complex nested attributes
{:dlti.dl_spec #dlti.dl_spec<
  i1 = dense<8> : vector<2xi64>,
  i64 = dense<64> : vector<2xi64>,
  "dlti.endianness" = "little"
>}

;; Function types
{:function_type (!function (inputs i32) (results i32))}
```

### Attribute Syntax Rules

1. **Keys are keywords** - Always start with `:`
2. **Symbols use `@`** - For function/symbol references
3. **No string quotes** - Use `@name`, not `"name"`
4. **Typed literals** - Use `(: value type)` for typed constants
5. **Dialect markers** - Use `#` prefix for dialect-specific attributes

---

## Regions and Control Flow

### Regions

Regions contain one or more blocks and define scope boundaries.

**Verbose syntax:**
```lisp
(regions
  (region
    (block [^label]
      (arguments [(: %arg TYPE) ...])
      OPERATION*)))
```

**Terse syntax:**
```lisp
(region BODY*)
```

### Blocks

Blocks are basic blocks in the control flow graph.

**Verbose syntax:**
```lisp
(block [^label]
  (arguments [(: %arg0 TYPE) ...])
  OPERATION*)
```

**Features:**
- Optional block label (for branches)
- Optional arguments (with types)
- List of operations (sequential execution)

### Control Flow Operations

#### Conditional Branch (verbose)
```lisp
(operation
  (name cf.cond_br)
  (operands %cond)
  (successors
    (successor ^then (%val1))
    (successor ^else (%val2))))
```

#### Unconditional Branch
```lisp
(operation
  (name cf.br)
  (successors
    (successor ^target (%arg1 %arg2))))
```

#### SCF If-Else (terse)
```lisp
(scf.if %cond
  (region then-expr)    ;; Implicit scf.yield
  (region else-expr))   ;; Implicit scf.yield
```

#### Loop with Explicit Blocks
```lisp
(block ^loop [(: %iter i32) (: %sum i32)]
  (declare cond (arith.cmpi {:predicate slt} %iter %limit))
  (operation
    (name cf.cond_br)
    (operands %cond)
    (successors
      (successor ^continue)
      (successor ^exit))))
```

---

## Operation Flattening

MLIR-Lisp supports **WAST-style nested operations** that are automatically flattened to SSA form.

### What is Operation Flattening?

Allows operations to be nested in operand positions (like WebAssembly text format):

**Input (nested):**
```lisp
(operation
  (name arith.addi)
  (result-types i64)
  (operands
    (operation
      (name arith.constant)
      (result-types i64)
      (attributes {:value (: 10 i64)}))
    (operation
      (name arith.constant)
      (result-types i64)
      (attributes {:value (: 32 i64)}))))
```

**Output (flattened):**
```lisp
(operation
  (name arith.constant)
  (result-bindings [%result_G0])
  (result-types i64)
  (attributes {:value (: 10 i64)}))
(operation
  (name arith.constant)
  (result-bindings [%result_G1])
  (result-types i64)
  (attributes {:value (: 32 i64)}))
(operation
  (name arith.addi)
  (result-types i64)
  (operands %result_G0 %result_G1))
```

### Pipeline Position

```
Reader → Macro Expander → Operation Flattener → Parser → Builder
```

Runs **after macro expansion** (macros can generate nested operations) and **before parsing** (parser expects flat operations).

### Gensym (Auto-Generated Bindings)

When a nested operation lacks explicit `(result-bindings ...)`, the flattener auto-generates:

**Format:** `%result_G0`, `%result_G1`, `%result_G2`, ...

**Rules:**
1. User-provided bindings are **always preserved**
2. Only auto-generate when `(result-bindings ...)` is missing
3. Counter increments globally across the module

### Evaluation Order

**Depth-first, left-to-right:**
- Inner nested operations are flattened before outer ones
- Left operands are evaluated before right operands
- Ensures dependencies are satisfied

**Example:**
```lisp
;; Input: ((5 + 3) * 2)
(arith.muli
  (arith.addi
    (arith.constant 5)
    (arith.constant 3))
  (arith.constant 2))

;; Evaluation order: 5, 3, (5+3), 2, ((5+3)*2)
```

### Multiple Results

If a nested operation produces multiple results, the **first result** is used:

```lisp
(operation
  (name some.consumer)
  (operands
    (operation
      (result-bindings [%r1 %r2 %r3])  ;; Multiple results
      ...)))

;; Becomes:
;; ... (hoisted operation)
;; (operation (name some.consumer) (operands %r1))  ;; Uses first result
```

---

## Type Inference

The terse syntax supports **type inference** for certain operations, reducing the need for explicit type annotations.

### Currently Implemented

#### 1. arith.constant

Type inferred from `:value` attribute:

```lisp
;; Explicit type in attribute
(declare c1 (arith.constant {:value (: 42 i64)}))
;; Result type: i64 (inferred)

;; Compiles to:
;; %c1 = arith.constant 42 : i64
```

**Rule:** If `:value` attribute has type annotation `(: val type)`, use that type as result type.

#### 2. Binary Arithmetic Operations

Type inferred from operand types:

```lisp
;; Both operands are i64
(declare c1 (arith.constant {:value (: 10 i64)}))
(declare c2 (arith.constant {:value (: 32 i64)}))
(declare sum (arith.addi %c1 %c2))
;; Result type: i64 (inferred from operands)
```

**Supported operations:**
- `arith.addi`, `arith.subi`, `arith.muli`
- `arith.divsi`, `arith.divui`, `arith.remsi`, `arith.remui`
- `arith.andi`, `arith.ori`, `arith.xori`

**Rule:** Result type matches the common type of all operands.

### Not Yet Implemented

From the terse syntax spec (future work):

#### Comparison Operations
```lisp
;; Should infer i1 result type
(declare cond (arith.cmpi {:predicate slt} %n %c1))
```

#### Function Calls
```lisp
;; Should infer result type from function signature
(declare result (func.call {:callee @foo} %arg))
```

#### Type Conversions
```lisp
;; Should infer from context
(declare wider (arith.extsi %narrow))
```

---

## Macro System

### Current Status

The macro system operates on the `Value` AST type at compile-time.

**Capabilities:**
- Macros work with Reader's `Value` type
- Can use LLVM operations to manipulate `Value` structs
- `CValueLayout` struct has known offsets for fields
- Struct field access via `llvm.getelementptr`
- Load/store via `llvm.load`/`llvm.store`
- Create `Value` structs via `llvm.alloca` + `llvm.store`

**Missing (minor):**
- String global syntax for macro-generated strings
- Quasiquote/unquote for better ergonomics (could be added)

### Macro Capabilities

Current macros can:
1. Read and manipulate `Value` structures
2. Generate new operations
3. Transform AST nodes
4. Access compile-time values

**Example use cases:**
- `call` macro - expands function calls
- `constant` macro - creates constants with bindings
- Custom dialect macros - transform domain-specific syntax

---

## Complete Examples

### Example 1: Simple Function (Verbose)

```lisp
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @add
      :function_type (!function (inputs i32 i32) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [(: %x i32) (: %y i32)])
          (operation
            (name arith.addi)
            (result-bindings [%sum])
            (result-types i32)
            (operands %x %y))
          (operation
            (name func.return)
            (operands %sum)))))))
```

**MLIR output:**
```mlir
func.func @add(%x: i32, %y: i32) -> i32 {
  %sum = arith.addi %x, %y : i32
  return %sum : i32
}
```

### Example 2: Conditional (Verbose)

```lisp
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @abs
      :function_type (!function (inputs i32) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [(: %val i32)])

          (operation
            (name arith.constant)
            (result-bindings [%c0])
            (result-types i32)
            (attributes {:value (: 0 i32)}))

          (operation
            (name arith.cmpi)
            (result-bindings [%is_neg])
            (result-types i1)
            (operands %val %c0)
            (attributes {:predicate (: 3 i64)}))

          (operation
            (name cf.cond_br)
            (operands %is_neg)
            (successors
              (successor ^negate)
              (successor ^positive))))

        (block [^negate]
          (arguments [])
          (operation
            (name arith.subi)
            (result-bindings [%negated])
            (result-types i32)
            (operands %c0 %val))
          (operation
            (name func.return)
            (operands %negated)))

        (block [^positive]
          (arguments [])
          (operation
            (name func.return)
            (operands %val))))))))
```

### Example 3: Fibonacci (Terse with Regions)

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
          (arguments [(: %n i32)])

          ;; Check if n <= 1
          (declare c1 (arith.constant {:value (: 1 i32)}))
          (declare cond (: (arith.cmpi {:predicate (: 3 i64)} %n %c1) i1))

          ;; Terse scf.if with regions
          (declare result (:
            (scf.if %cond
              (region %n)              ;; Base case: return n
              (region                   ;; Recursive case
                ;; Compute fib(n-1)
                (declare c1_rec (arith.constant {:value (: 1 i32)}))
                (declare n_minus_1 (arith.subi %n %c1_rec))
                (declare fib_n_minus_1 (: (func.call {:callee @fibonacci} %n_minus_1) i32))

                ;; Compute fib(n-2)
                (declare c2 (arith.constant {:value (: 2 i32)}))
                (declare n_minus_2 (arith.subi %n %c2))
                (declare fib_n_minus_2 (: (func.call {:callee @fibonacci} %n_minus_2) i32))

                ;; Add results
                (declare sum (arith.addi %fib_n_minus_1 %fib_n_minus_2))
                %sum))                ;; Implicit scf.yield
            i32))

          (func.return %result))))))
```

### Example 4: Nested Operations (Operation Flattening)

```lisp
;; Before flattening
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @nested
      :function_type (!function (inputs) (results i64))
    })
    (regions
      (region
        (block [^entry]
          (arguments [])
          (operation
            (name func.return)
            (operands
              (operation
                (name arith.addi)
                (result-types i64)
                (operands
                  (operation
                    (name arith.constant)
                    (result-types i64)
                    (attributes {:value (: 10 i64)}))
                  (operation
                    (name arith.constant)
                    (result-types i64)
                    (attributes {:value (: 32 i64)})))))))))))

;; After flattening
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @nested
      :function_type (!function (inputs) (results i64))
    })
    (regions
      (region
        (block [^entry]
          (arguments [])
          (operation
            (name arith.constant)
            (result-bindings [%result_G0])
            (result-types i64)
            (attributes {:value (: 10 i64)}))
          (operation
            (name arith.constant)
            (result-bindings [%result_G1])
            (result-types i64)
            (attributes {:value (: 32 i64)}))
          (operation
            (name arith.addi)
            (result-bindings [%result_G2])
            (result-types i64)
            (operands %result_G0 %result_G1))
          (operation
            (name func.return)
            (operands %result_G2)))))))
```

---

## Grammar Summary

### Top-Level Grammar

```
Program       ::= (mlir Form*)

Form          ::= Operation
                | TypeAlias
```

### Verbose Operation Grammar

```
Operation     ::= (operation
                    (name OpName)
                    Section*)

OpName        ::= IDENT ("." IDENT)*

Section       ::= ResultBindings
                | ResultTypes
                | Operands
                | Attributes
                | Successors
                | Regions
                | Location

ResultBindings ::= (result-bindings "[" ValueId* "]")
ResultTypes    ::= (result-types Type*)
Operands       ::= (operands ValueId*)
Attributes     ::= (attributes "{" (Keyword Value)* "}")
Successors     ::= (successors Successor*)
Regions        ::= (regions Region*)

Region         ::= (region Block+)

Block          ::= (block
                      "[" BlockId "]"
                      (arguments "[" (: ValueId Type)* "]")
                      Operation*)

Successor      ::= (successor BlockId ("(" ValueId* ")")?)
```

### Terse Operation Grammar

```
TerseOp       ::= (OpName Attributes? Operands? TerseRegion*)

OpName        ::= IDENT "." IDENT ("." IDENT)*    ;; Must contain "."

Attributes    ::= "{" (Keyword Value)* "}"

Operands      ::= Value*

TerseRegion   ::= (region Form+)

Declare       ::= (declare IDENT Expr)

TypeAnnot     ::= (: Expr Type)
```

### Type Grammar

```
Type          ::= IDENT                              ;; Builtin: i32, f64, index
                | "!" IDENT ("<" TypeArgs ">")?       ;; Dialect: !llvm.ptr
                | "(" "!function"
                     "(inputs" Type* ")"
                     "(results" Type* ")" ")"        ;; Function type
```

### Value Grammar

```
Value         ::= ValueId          ;; %result, %arg0
                | BlockId          ;; ^bb0, ^entry
                | Symbol           ;; @main, @fibonacci
                | Keyword          ;; :sym_name, :value
                | AttrMarker       ;; #llvm.linkage<internal>
                | Number           ;; 42, 3.14
                | String           ;; "text"
                | Boolean          ;; true, false
                | TypedLiteral     ;; (: 42 i32)
                | Operation        ;; Nested operation
                | TerseOp          ;; Terse operation
                | Declare          ;; Declare form
                | TypeAnnot        ;; Type annotation

ValueId       ::= "%" (NUMBER | IDENT)
BlockId       ::= "^" (NUMBER | IDENT)
Symbol        ::= "@" (NUMBER | IDENT)
Keyword       ::= ":" IDENT ("." IDENT)*
AttrMarker    ::= "#" <opaque-text>
```

---

## Future Extensions

### From Terse Syntax Spec (Not Yet Implemented)

#### 1. Let Bindings
```lisp
(let [binding1 expr1
      binding2 expr2
      binding3 expr3]
  body)
```

#### 2. Do Form (Sequencing)
```lisp
(do
  (llvm.store {} val ptr)
  (llvm.store {} val2 ptr2)
  (arith.constant {:value 0}))  ;; Returned
```

#### 3. Terse Function Definition
```lisp
(func.func {:sym_name add :function_type (-> (i32 i32) (i32))}
  [(: a i32) (: b i32)]
  (arith.addi {} a b))
```

#### 4. Inline Successors
```lisp
(cf.cond_br {} cond
  ^{:then (func.return {} doubled)
    :else (func.return {} val)})
```

#### 5. Arrow Function Types
```lisp
(-> (i32 i32) (i32))    ;; Instead of (!function ...)
```

### From Pretty Syntax Spec (Design Only)

#### 1. defn for Functions
```lisp
(defn add [(arg1 i32) (arg2 i32)] (-> (i32 i32) (i32))
  (arith.addi {} arg1 arg2))
```

#### 2. Multi-Binding Let
```lisp
(let [A? (memref.cast (memref.alloc memref<5xf32>) memref<?xf32>)
      B? (memref.cast A? memref<*xf32>)]
  body)
```

#### 3. For Loops
```lisp
(for [i 0 5 1]
  (memref.store 1.23 A? [i]))
```

---

## Implementation Notes

### Compiler Pipeline

```
Source File
    ↓
Reader (src/reader.zig)
    ↓
Value AST
    ↓
Macro Expander
    ↓
Expanded Value AST
    ↓
Operation Flattener (src/operation_flattener.zig)
    ↓
Flattened Value AST
    ↓
Parser (src/parser.zig)
    ↓
MLIR Operations
    ↓
Builder (src/builder.zig)
    ↓
MLIR Module
    ↓
MLIR Optimizer/JIT
```

### Key Files

**Core Implementation:**
- `src/reader.zig` - Lexer and reader (S-expression parsing)
- `src/parser.zig` - Parser (Value AST → MLIR operations)
- `src/builder.zig` - Builder (MLIR IR construction)
- `src/operation_flattener.zig` - Operation flattening pass

**Documentation:**
- `docs/grammar.md` - Verbose grammar specification
- `docs/terse-syntax-spec.md` - Terse syntax specification (full)
- `TERSE_SYNTAX_SUMMARY.md` - Implementation status
- `examples/TERSE_REGIONS_GUIDE.md` - Region syntax guide
- `docs/operation_flattening.md` - Operation flattening details
- `docs/pretty-syntax.md` - Future pretty syntax design

### Compatibility Notes

1. **Verbose and terse syntax can be mixed** - Use verbose for complex cases, terse for common operations
2. **Both generate identical MLIR IR** - No semantic difference
3. **Verbose syntax is always supported** - Guaranteed compatibility and escape hatch
4. **Round-trip safe** - Verbose syntax preserves all information

---

## Summary

MLIR-Lisp provides a **flexible, layered syntax** for writing MLIR programs:

1. **Verbose Syntax** - Explicit, self-documenting, always available
2. **Terse Syntax** - Concise, with type inference and implicit terminators
3. **Operation Flattening** - WAST-style nested operations for convenience
4. **Type Inference** - Reduces boilerplate for common operations
5. **Macro System** - Extensible with compile-time transformations

**Current Status:**
- ✅ Verbose syntax: **fully implemented**
- ✅ Terse operations: **fully implemented**
- ✅ Declare form: **fully implemented**
- ✅ Type inference: **basic implementation** (constants, binary arithmetic)
- ✅ Terse regions: **fully implemented**
- ✅ Operation flattening: **fully implemented**
- ⏳ Let bindings, do form, inline successors: **specified, not yet implemented**
- 📋 Pretty syntax: **design spec only**

**Philosophy:** Start with the verbose syntax for correctness, use terse syntax for ergonomics, and rely on type inference to reduce boilerplate. The macro system and operation flattening provide additional convenience without losing MLIR's expressiveness.

---

**End of MLIR-Lisp Syntax Reference**
