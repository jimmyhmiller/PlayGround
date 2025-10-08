# IRDL + Transform Dialect System

A meta-circular compiler infrastructure for defining MLIR dialects and transformations in Lisp itself!

## Overview

This system allows you to:

1. **Define MLIR dialects using IRDL in Lisp** - Like TableGen but in Lisp
2. **Write Transform dialect transformations in Lisp** - Declarative pattern matching
3. **Import dialects like Racket's `#lang`** - Modular dialect system
4. **Meta-circular compilation** - Define compiler in the language it compiles

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Lisp Source Code                          │
│              #lang lisp                                      │
│              (+ (* 10 20) 30)                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              IRDL Dialect Definition                         │
│         (defirdl-dialect lisp ...)                           │
│         Defines: lisp.constant, lisp.add, etc.               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           High-Level lisp.* Operations                       │
│         %0 = lisp.constant 10                                │
│         %1 = lisp.constant 20                                │
│         %2 = lisp.mul %0, %1                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Transform Dialect Transformations                   │
│         (deftransform lower-lisp-to-arith ...)               │
│         Pattern matching and rewriting                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Standard MLIR Dialects                             │
│         %0 = arith.constant 10                               │
│         %1 = arith.constant 20                               │
│         %2 = arith.muli %0, %1                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
                  LLVM IR
```

## Components

### 1. IRDL Dialect Definition (`irdl_dialect_definition.lisp`)

Define your dialect structure:

```lisp
(defirdl-dialect lisp
  :namespace "lisp"

  (defirdl-op constant
    :attributes [(value IntegerAttr)]
    :results [(result AnyInteger)]
    :traits [Pure NoMemoryEffect])

  (defirdl-op add
    :operands [(lhs AnyInteger) (rhs AnyInteger)]
    :results [(result AnyInteger)]
    :traits [Pure Commutative NoMemoryEffect]
    :constraints [(same-type lhs rhs result)]))
```

**IRDL Features:**
- Declarative operation definition
- Type constraints
- Trait specifications (Pure, Commutative, etc.)
- Verifiers
- Documentation embedded

### 2. Transform Patterns (`transform_patterns.lisp`)

Define how to transform your dialect:

```lisp
;; Transform sequence
(deftransform lower-lisp-to-arith
  (transform.sequence
    (let [adds (transform.match :ops ["lisp.add"])]
      (transform.apply-patterns :to adds
        (use-pattern add-lowering)))))

;; PDL pattern
(defpdl-pattern add-lowering
  :match
  (let [lhs (pdl.operand)
        rhs (pdl.operand)
        op (pdl.operation "lisp.add" :operands [lhs rhs])]

    :rewrite
    (pdl.replace op :with
      (pdl.operation "arith.addi" :operands [lhs rhs]))))
```

**Transform Features:**
- Pattern Descriptor Language (PDL)
- Declarative pattern matching
- Rewrite rules
- Benefit ordering
- Constraint checking

### 3. Dialect Usage (`using_lisp_dialect.lisp`)

Use your dialect:

```lisp
#lang lisp

(import-transform lower-lisp-to-arith)

(defn compute [] i32
  (+ (* 10 20) 30))  ; Emits lisp.* ops

(apply-transform lower-lisp-to-arith)  ; Lowers to arith.*
```

## Key Features

### Meta-Circular Design

The compiler is defined in the language it compiles:

1. Write IRDL definitions in Lisp → Generates dialect
2. Write Transform patterns in Lisp → Generates transformations
3. Write code using your dialect → Uses your definitions

### Racket-Style `#lang`

```lisp
#lang lisp          ; Import lisp dialect
#lang arith         ; Import arith dialect
#lang my-custom     ; Import custom dialect
```

Benefits:
- Modular dialect system
- Composable dialects
- Versioned dialects
- Namespace isolation

### Progressive Lowering

Each dialect level can be optimized independently:

```
High-Level (lisp dialect)
  ↓ [constant folding, tail-call opt]
Mid-Level (arith dialect)
  ↓ [strength reduction, CSE]
Low-Level (LLVM)
  ↓ [register allocation]
Machine Code
```

## Macro System

### Core Macros

#### `defirdl-dialect`
Define a new dialect with operations:

```lisp
(defirdl-dialect name
  :namespace "string"
  :description "..."
  (defirdl-op op-name ...))
```

#### `defirdl-op`
Define an operation in the dialect:

```lisp
(defirdl-op add
  :operands [(lhs Type) (rhs Type)]
  :results [(result Type)]
  :attributes [(name Type)]
  :traits [Pure Commutative]
  :constraints [(same-type lhs rhs)])
```

#### `deftransform`
Define a transform sequence:

```lisp
(deftransform name
  (transform.sequence
    ...transformations...))
```

#### `defpdl-pattern`
Define a PDL rewrite pattern:

```lisp
(defpdl-pattern name
  :benefit N
  :match (...)
  :rewrite (...))
```

#### `import-dialect`
Import a dialect (like Racket's `#lang`):

```lisp
(import-dialect lisp)       ; Import lisp dialect
(import-transform my-opt)   ; Import transformation
```

## Example Compilation Pipeline

### Input: `example.lisp`
```lisp
#lang lisp

(defn fib [n:i32] i32
  (if (< n 2)
    n
    (+ (fib (- n 1)) (fib (- n 2)))))
```

### Step 1: Parse & Emit High-Level IR
```mlir
func.func @fib(%arg0: i32) -> i32 {
  %c2 = lisp.constant 2 : i32
  %cond = lisp.lt %arg0, %c2 : i1
  %result = lisp.if %cond -> i32 {
    lisp.yield %arg0 : i32
  } else {
    %c1 = lisp.constant 1 : i32
    %nm1 = lisp.sub %arg0, %c1 : i32
    %fib1 = lisp.call @fib(%nm1) : i32
    %c2_0 = lisp.constant 2 : i32
    %nm2 = lisp.sub %arg0, %c2_0 : i32
    %fib2 = lisp.call @fib(%nm2) : i32
    %sum = lisp.add %fib1, %fib2 : i32
    lisp.yield %sum : i32
  }
  lisp.return %result : i32
}
```

### Step 2: Apply Transforms
```lisp
(apply-transform tail-call-optimize)    ; Detect tail calls
(apply-transform constant-fold)         ; Fold constants
(apply-transform lower-to-arith)        ; Lower to arith dialect
```

### Step 3: Lowered IR
```mlir
func.func @fib(%arg0: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %cond = arith.cmpi slt, %arg0, %c2 : i1
  %result = scf.if %cond -> i32 {
    scf.yield %arg0 : i32
  } else {
    %c1 = arith.constant 1 : i32
    %nm1 = arith.subi %arg0, %c1 : i32
    %fib1 = func.call @fib(%nm1) : i32
    %c2_0 = arith.constant 2 : i32
    %nm2 = arith.subi %arg0, %c2_0 : i32
    %fib2 = func.call @fib(%nm2) : i32
    %sum = arith.addi %fib1, %fib2 : i32
    scf.yield %sum : i32
  }
  func.return %result : i32
}
```

## Benefits

### For Language Designers

- **Fast Iteration**: Define dialects in Lisp, not C++
- **Experimentation**: Try different operation sets quickly
- **Documentation**: Dialect definition is self-documenting
- **Versioning**: Dialects as data, easy to version

### For Compiler Writers

- **Declarative Transforms**: Write what to match, not how
- **Composable Passes**: Chain transformations easily
- **Debuggable**: Inspect transform IR
- **Reusable**: Share transformation libraries

### For Users

- **#lang Style**: Import dialects as needed
- **Clear Semantics**: Each dialect level has meaning
- **Inspectable**: See IR at any level
- **Optimizable**: Optimizations at all levels

## Implementation Status

✅ **Conceptual Design Complete**
- IRDL macro syntax defined
- Transform macro syntax defined
- PDL pattern syntax defined
- #lang import system designed

🚧 **Implementation Needed**
- Macro expander for IRDL macros
- Macro expander for Transform macros
- PDL pattern compilation
- Dialect registration system
- Transform interpreter bindings

## Next Steps

1. **Implement IRDL Macros**
   - `defirdl-dialect` expands to MLIR IRDL operations
   - `defirdl-op` generates operation definitions
   - Register dialects with MLIR context

2. **Implement Transform Macros**
   - `deftransform` generates transform.sequence operations
   - `defpdl-pattern` generates PDL pattern IR
   - Bind to transform interpreter

3. **Implement #lang System**
   - `import-dialect` loads dialect definitions
   - Namespace management
   - Dialect composition

4. **Create Standard Library**
   - Common patterns (constant folding, DCE, etc.)
   - Standard dialects (lisp, array, tensor)
   - Transformation pipelines

## Philosophy

> "A language that doesn't affect the way you think about programming is not worth knowing."
> - Alan Perlis

By defining dialects and transformations in Lisp:
- The language becomes **self-describing**
- Transformations are **data**
- Compiler is **programmable**
- Users can **extend** the system

This is the **meta-circular** ideal: The compiler compiles itself!
