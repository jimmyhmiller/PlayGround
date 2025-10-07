# MLIR-Lisp Design Document

A minimalist Lisp that compiles to MLIR (Multi-Level Intermediate Representation).

## Philosophy

- **Minimal built-in functionality**: Only macros, basic control flow, and the primitive ability to emit MLIR
- **MLIR as the primitive**: Almost everything in the language emits MLIR operations
- **Build abstractions via macros**: Users define their own language features on top

## Syntax Examples

### Example 1: Basic Arithmetic (Builder Style)

```lisp
;; Define a function using MLIR builder-style syntax
(defn add-numbers []
  ;; Almost everything is just emitting MLIR operations
  (let [ten (mlir/arith.constant {:value 10 :type i32})
        thirty-two (mlir/arith.constant {:value 32 :type i32})
        result (mlir/arith.addi ten thirty-two)]
    (mlir/func.return result)))
```

### Example 2: More Explicit Builder Interface

```lisp
;; defn itself is just a macro that emits MLIR
(defn main [] i32
  (block entry
    ;; Direct MLIR operation emission
    (op arith.constant
        :attrs {:value (i32-attr 10)}
        :results [i32]
        :as %ten)

    (op arith.constant
        :attrs {:value (i32-attr 32)}
        :results [i32]
        :as %thirty-two)

    (op arith.addi
        :operands [%ten %thirty-two]
        :results [i32]
        :as %result)

    (op func.return
        :operands [%result])))
```

### Example 3: Direct MLIR Text Format Style

```lisp
;; Minimal s-expr wrapper around MLIR text format
(module
  (func.func :sym_name "main" :function_type (fn [] i32)
    (block
      (%0 (arith.constant :value (i32 10) -> i32))
      (%1 (arith.constant :value (i32 32) -> i32))
      (%2 (arith.addi %0 %1 -> i32))
      (func.return %2))))
```

### Example 4: Macros to Build Abstractions

```lisp
;; The language has macros, so you can build your own abstractions
(defmacro defun [name args ret-type & body]
  `(module
     (func.func :sym_name ~(str name)
                :function_type (fn [~@(map type-of args)] ~ret-type)
       (block ~@body))))

;; Now you can use your abstraction, but it still emits MLIR
(defun calculate [x y] i32
  (%result (arith.addi x y -> i32))
  (func.return %result))

;; You could even build a "normal" language on top
(defmacro + [a b]
  `(mlir/arith.addi ~a ~b))

;; And now:
(defun add-three [x y z] i32
  (let [tmp (+ x y)]
    (+ tmp z)))
```

### Example 5: Custom Dialect Definition

```lisp
;; Define your own dialect using IRDL, emitted as MLIR
(define-dialect mymath
  (irdl.dialect @mymath
    (irdl.operation @add
      (%arith-type (irdl.any_of (irdl.is i8) (irdl.is i16) (irdl.is i32)))
      (irdl.operands :lhs %arith-type :rhs %arith-type)
      (irdl.results :result %arith-type))))

;; Use it
(defn custom-add [] i32
  (%ten (arith.constant :value (i32 10) -> i32))
  (%thirty-two (arith.constant :value (i32 32) -> i32))
  (%result (mymath.add %ten %thirty-two -> i32))
  (func.return %result))
```

### Example 6: Transforms/Rewrites

```lisp
;; Define transforms as MLIR transform dialect code
(define-transform mymath-to-arith
  (transform.with_pdl_patterns
    (pdl.pattern @mymath_to_arith :benefit 1
      (let [%lhs (pdl.operand)
            %rhs (pdl.operand)
            %result-type (pdl.type)
            %mymath-op (pdl.operation "mymath.add" [%lhs %rhs] -> %result-type)]
        (pdl.rewrite %mymath-op
          (%arith-op (pdl.operation "arith.addi" [%lhs %rhs] -> %result-type))
          (pdl.replace %mymath-op :with %arith-op))))))

;; Apply transform
(apply-transform mymath-to-arith my-module)
```

## Core Language Features (Minimal Set)

### Special Forms
- `let` - local bindings
- `do` - sequencing
- `block` - MLIR block
- `module` - MLIR module
- `defmacro` - define macros
- `quote` / `quasiquote` / `unquote` - for macros

### MLIR Primitives
- `op` - emit an MLIR operation
- Type constructors: `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, etc.
- Attribute constructors: `i32-attr`, `string-attr`, `type-attr`, etc.
- Result/operand references: `%name` syntax

### Everything Else is Built via Macros
- `defn` - define functions
- `defun` - alternative function definition
- Arithmetic operators: `+`, `-`, `*`, `/`
- Control flow: `if`, `when`, `cond`
- Custom dialects: `define-dialect`, `define-transform`

## Implementation Architecture

1. **Parser** - S-expression parser (using nom)
2. **AST** - Lisp AST representation
3. **Macro Expander** - Expand macros recursively
4. **MLIR Emitter** - Translate AST to MLIR operations via melior
5. **Runtime** - Execute via MLIR JIT or AOT compilation

## Design Principles

1. **MLIR is the primitive** - Direct access to MLIR operations
2. **Minimal core** - Only essential features built-in
3. **Macros for abstraction** - Build your own language
4. **Dual syntax support** - Both builder-style and text-format-style
5. **No magic** - Everything compiles to visible MLIR operations
