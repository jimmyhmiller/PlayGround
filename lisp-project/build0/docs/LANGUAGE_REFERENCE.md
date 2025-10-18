# Lisp0 Language Reference

**Version:** 0.1.0
**Last Updated:** October 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Types](#basic-types)
3. [Type System](#type-system)
4. [Literals](#literals)
5. [Variables and Bindings](#variables-and-bindings)
6. [Functions](#functions)
7. [Control Flow](#control-flow)
8. [Operators](#operators)
9. [Composite Types](#composite-types)
10. [Memory Management](#memory-management)
11. [Arrays](#arrays)
12. [C Foreign Function Interface](#c-foreign-function-interface)
13. [Macros](#macros)
14. [Namespaces](#namespaces)
15. [Error Handling](#error-handling)

---

## Introduction

Lisp0 is a statically-typed Lisp dialect that compiles to C. It features:

- **Bidirectional type checking** with type inference
- **First-class functions** with support for recursion and mutual recursion
- **Algebraic data types** (structs and enums)
- **Manual memory management** with pointers and heap allocation
- **C FFI** for interfacing with C libraries
- **Compile-time macros** for metaprogramming
- **Arrays** with both stack and heap allocation

---

## Basic Types

### Integer Types

| Type | Description | C Type | Range |
|------|-------------|--------|-------|
| `Int` | Generic integer (polymorphic) | `int32_t` | Platform-dependent |
| `I8` | 8-bit signed integer | `int8_t` | -128 to 127 |
| `I16` | 16-bit signed integer | `int16_t` | -32,768 to 32,767 |
| `I32` | 32-bit signed integer | `int32_t` | -2^31 to 2^31-1 |
| `I64` | 64-bit signed integer | `int64_t` | -2^63 to 2^63-1 |
| `U8` | 8-bit unsigned integer | `uint8_t` | 0 to 255 |
| `U16` | 16-bit unsigned integer | `uint16_t` | 0 to 65,535 |
| `U32` | 32-bit unsigned integer | `uint32_t` | 0 to 2^32-1 |
| `U64` | 64-bit unsigned integer | `uint64_t` | 0 to 2^64-1 |

### Floating-Point Types

| Type | Description | C Type |
|------|-------------|--------|
| `Float` | Generic float (polymorphic) | `float` or `double` |
| `F32` | 32-bit floating-point | `float` |
| `F64` | 64-bit floating-point | `double` |

### Other Primitive Types

| Type | Description | C Type |
|------|-------------|--------|
| `Bool` | Boolean (true/false) | `bool` |
| `Nil` | Nil type (no value) | `void` |
| `String` | String literal type | `const char*` |

---

## Type System

### Type Annotations

Explicit type annotations use the `(: Type)` syntax:

```lisp
(def x (: I32) 42)
(def name (: String) "Alice")
```

### Type Inference

The type checker can infer types in many contexts:

```lisp
(def add1 (: (-> [Int] Int))
  (fn [x] (+ x 1)))  ; x's type is inferred as Int
```

### Subtyping

Sized integer and float types are subtypes of generic `Int` and `Float`:

```lisp
(def x (: I32) 100)
(def y (: Int) x)  ; OK: I32 <: Int
```

### Bidirectional Type Checking

Lisp0 uses bidirectional type checking with two modes:

- **Synthesis mode**: Infers the type of an expression
- **Checking mode**: Checks an expression against an expected type

This enables better type inference while maintaining type safety.

---

## Literals

### Integer Literals

```lisp
42        ; Int (generic)
-15       ; Int
0         ; Int
```

### Float Literals

```lisp
3.14      ; Float
-0.5      ; Float
2.0       ; Float
```

### String Literals

```lisp
"Hello, world!"
"Line 1\nLine 2"
""  ; empty string
```

### Boolean Literals

```lisp
true
false
```

### Nil Literal

```lisp
nil  ; represents absence of value
```

### Symbol and Keyword Literals

```lisp
:keyword
'symbol
```

---

## Variables and Bindings

### Global Definitions

Define global constants with `def`:

```lisp
(def pi (: F64) 3.14159)
(def max-count (: U32) 100)
```

### Let Bindings

Create local bindings with `let`:

```lisp
(let [x (: Int) 10]
  (+ x 5))  ; => 15
```

Let bindings MUST have a type

Multiple bindings in sequence:

```lisp
(let [x (: Int) 10]
  (let [y (: Int) (* x 2)]
    (+ x y)))  ; => 30
```

Sequential dependencies:

```lisp
(let [a (: Int) 5]
  (let [b (: Int) (+ a 3)]
    (* a b)))  ; a is in scope for b's initializer
```

### Mutation

Use `set!` to mutate variables:

```lisp
(def counter (: I32) 0)
(set! counter (+ counter 1))
```

---

## Functions

### Function Definitions

Functions are defined with `def` and `fn`:

```lisp
(def add (: (-> [Int Int] Int))
  (fn [x y]
    (+ x y)))
```

### Function Types

Function types use the `(-> [ParamTypes...] ReturnType)` syntax:

```lisp
(-> [Int] Int)           ; takes Int, returns Int
(-> [I32 I32] Bool)      ; takes two I32, returns Bool
(-> [] Nil)              ; no parameters, returns Nil
```

### Function Application

```lisp
(add 10 20)  ; => 30
```

### Recursion

Recursive functions are fully supported:

```lisp
(def factorial (: (-> [U32] U32))
  (fn [n]
    (if (= n 0)
        1
        (* n (factorial (- n 1))))))
```

### Mutual Recursion

Forward references enable mutually recursive functions:

```lisp
(def is-even (: (-> [I32] Bool))
  (fn [n]
    (if (= n 0)
        true
        (is-odd (- n 1)))))

(def is-odd (: (-> [I32] Bool))
  (fn [n]
    (if (= n 0)
        false
        (is-even (- n 1)))))
```

### Higher-Order Functions

Functions can be passed as values:

```lisp
(def apply-twice (: (-> [(-> [Int] Int) Int] Int))
  (fn [f x]
    (f (f x))))

(def add1 (: (-> [Int] Int))
  (fn [x] (+ x 1)))

(apply-twice add1 5)  ; => 7
```

---

## Control Flow

### Conditionals

The `if` expression evaluates a condition and returns one of two branches:

```lisp
(if condition
    then-expr
    else-expr)
```

Example:

```lisp
(def abs (: (-> [Int] Int))
  (fn [x]
    (if (< x 0)
        (- 0 x)
        x)))
```

Both branches must have compatible types:

```lisp
(if true 42 "no")  ; ERROR: branches have incompatible types
```

### While Loops

```lisp
(while condition
  body-expr1
  body-expr2
  ...)
```

Example:

```lisp
(def count-to-ten (: (-> [] Nil))
  (fn []
    (let [i (: I32) 0]
      (while (< i 10)
        (set! i (+ i 1))))))
```

### C-Style For Loops

```lisp
(c-for [var (: Type) init] test step
  body-expr1
  body-expr2
  ...)
```

Example:

```lisp
(c-for [i (: I32) 0] (< i 10) (set! i (+ i 1))
  (printf (c-str "i = %d\n") i))
```

---

## Operators

### Arithmetic Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `(+ 1 2 3)` => `6` |
| `-` | Subtraction | `(- 10 3)` => `7` |
| `*` | Multiplication | `(* 2 3 4)` => `24` |
| `/` | Division | `(/ 10 2)` => `5` |
| `%` | Modulo | `(% 10 3)` => `1` |

Supports multiple operands for `+` and `*`:

```lisp
(+ 1 2 3 4 5)  ; => 15
(* 2 3 4)      ; => 24
```

### Comparison Operators

| Operator | Description | Returns |
|----------|-------------|---------|
| `<` | Less than | `Bool` |
| `>` | Greater than | `Bool` |
| `<=` | Less than or equal | `Bool` |
| `>=` | Greater than or equal | `Bool` |
| `=` | Equal | `Bool` |
| `!=` | Not equal | `Bool` |

Example:

```lisp
(< 5 10)   ; => true
(= 42 42)  ; => true
```

### Logical Operators

| Operator | Description | Short-circuits |
|----------|-------------|----------------|
| `and` | Logical AND | Yes |
| `or` | Logical OR | Yes |
| `not` | Logical NOT | N/A |

Example:

```lisp
(and true false)   ; => false
(or false true)    ; => true
(not true)         ; => false
```

---

## Composite Types

### Structs

Define struct types with named fields:

```lisp
(def Point (: Type)
  (Struct [x Int] [y Int]))
```

Create struct instances:

```lisp
(def p (: Point) (Point 10 20))
```

Access struct fields with `.` syntax:

```lisp
(. p x)  ; => 10
(. p y)  ; => 20
```

Works with struct values, pointers, and extern types:

```lisp
(def ptr (: (Pointer Point))
  (allocate Point (Point 5 10)))

(. ptr x)  ; automatically dereferences pointer
```

Nested structs:

```lisp
(def Line (: Type)
  (Struct [start Point] [end Point]))

(def line (: Line)
  (Line (Point 0 0) (Point 10 10)))

(. line start)  ; => Point instance
```

Empty structs are allowed:

```lisp
(def Unit (: Type) (Struct))
```

### Enums

Define enumeration types:

```lisp
(def Color (: Type)
  (Enum Red Green Blue))
```

Use enum variants with qualified names:

```lisp
(def favorite (: Color) Color/Red)
```

---

## Memory Management

Lisp0 uses manual memory management with explicit allocation and deallocation.

### Pointer Types

```lisp
(Pointer T)         ; pointer to type T
(Pointer (Pointer T))  ; nested pointers
```

### Allocation and Deallocation

```lisp
; Allocate on heap
(allocate Type initial-value)

; Deallocate
(deallocate ptr)
```

Example:

```lisp
(def p (: (Pointer I32))
  (allocate I32 42))

; Use the pointer...

(deallocate p)
```

### Dereferencing

```lisp
(dereference ptr)  ; read value through pointer
```

Example:

```lisp
(def p (: (Pointer I32)) (allocate I32 100))
(dereference p)  ; => 100
```

### Writing Through Pointers

```lisp
(pointer-write! ptr value)
```

Example:

```lisp
(def p (: (Pointer I32)) (allocate I32 0))
(pointer-write! p 42)
(dereference p)  ; => 42
```

### Address-Of

```lisp
(address-of value)
```

Example:

```lisp
(def x (: I32) 10)
(def ptr (: (Pointer I32)) (address-of x))
```

### Null Pointers

```lisp
pointer-null  ; null pointer literal
```

Example:

```lisp
(def p (: (Pointer I32)) pointer-null)
```

### Pointer Equality

```lisp
(pointer-equal? ptr1 ptr2)
```

### Pointer Casting

Convert pointer types for C FFI:

```lisp
(cast TargetType value)
```

Example:

```lisp
(def i32-ptr (: (Pointer I32)) (allocate I32 42))
(def u8-ptr (: (Pointer U8)) (cast (Pointer U8) i32-ptr))
```

Generates C-style cast: `((uint8_t*)ptr)`

### Struct Field Operations

Read and write struct fields through pointers:

```lisp
(pointer-field-read ptr field-name)
(pointer-field-write! ptr field-name value)
```

Example:

```lisp
(def Point (: Type) (Struct [x Int] [y Int]))
(def p (: (Pointer Point)) (allocate Point (Point 0 0)))

(pointer-field-write! p x 10)
(pointer-field-read p x)  ; => 10
```

---

## Arrays

Lisp0 supports fixed-size arrays with both stack and heap allocation.

### Array Types

```lisp
(Array ElementType Size)
```

Example:

```lisp
(Array Int 10)         ; array of 10 integers
(Array F32 5)          ; array of 5 floats
(Array (Array I32 3) 2)  ; 2x3 matrix
```

### Stack Array Creation

```lisp
; Uninitialized array
(array Type Size)

; Initialized array
(array Type Size InitValue)
```

Example:

```lisp
(def arr (: (Array I32 5))
  (array I32 5 0))  ; [0, 0, 0, 0, 0]
```

### Heap Array Allocation

```lisp
; Allocate uninitialized array
(allocate-array Type Size)

; Allocate initialized array
(allocate-array Type Size InitValue)

; Deallocate array
(deallocate-array ptr)
```

Example:

```lisp
(def arr (: (Pointer (Array I32 100)))
  (allocate-array I32 100 0))

; Use array...

(deallocate-array arr)
```

### Array Operations

```lisp
; Read element
(array-ref arr index)

; Write element
(array-set! arr index value)

; Get array length (compile-time constant)
(array-length arr)

; Get pointer to element
(array-ptr arr index)
```

Example:

```lisp
(def numbers (: (Array I32 5))
  (array I32 5 0))

(array-set! numbers 0 10)
(array-set! numbers 1 20)

(array-ref numbers 0)  ; => 10
(array-length numbers) ; => 5
```

### Pointer Indexing

For heap-allocated arrays accessed through pointers:

```lisp
(pointer-index-read ptr index)
(pointer-index-write! ptr index value)
```

Example:

```lisp
(def arr (: (Pointer (Array I32 10)))
  (allocate-array I32 10 0))

(pointer-index-write! arr 3 42)
(pointer-index-read arr 3)  ; => 42
```

### Multi-Dimensional Arrays

```lisp
(def matrix (: (Array (Array I32 3) 2))
  (array (Array I32 3) 2 (array I32 3 0)))

; Access elements
(def row (: (Array I32 3)) (array-ref matrix 0))
(array-ref row 0)  ; => 0
```

### Arrays with Functions

Arrays can be passed as parameters and returned from functions:

```lisp
; Pass array as parameter
(def sum-array (: (-> [(Array I32 3)] I32))
  (fn [arr]
    (+ (array-ref arr 0)
       (+ (array-ref arr 1)
          (array-ref arr 2)))))

(def nums (: (Array I32 3)) (array I32 3 0))
(array-set! nums 0 10)
(array-set! nums 1 20)
(array-set! nums 2 30)
(sum-array nums)  ; => 60

; Return array from function
(def make-array (: (-> [] (Array I32 3)))
  (fn []
    (let [arr (: (Array I32 3)) (array I32 3 0)]
      (array-set! arr 0 1)
      (array-set! arr 1 2)
      (array-set! arr 2 3)
      arr)))

(def result (: (Array I32 3)) (make-array))
(array-ref result 0)  ; => 1
```

---

## C Foreign Function Interface

Lisp0 provides two sets of forms for C interoperability.

### Include Headers

```lisp
(include-header "header.h")
```

Example:

```lisp
(include-header "stdio.h")
(include-header "stdlib.h")
```

### `extern-*` Forms (Emit C Declarations)

Use when you need the compiler to generate extern declarations:

```lisp
; Function declaration (generates extern)
(extern-fn name [param-types...] -> return-type)

; Opaque type
(extern-type name)

; Union type
(extern-union name)

; Struct type (from header)
(extern-struct name)

; Variable declaration
(extern-var name type)
```

Example:

```lisp
(extern-fn my_custom_fn [x I32] -> I32)
; Generates: extern int32_t my_custom_fn(int32_t);
```

### `declare-*` Forms (Type-Only, No Emission)

Use when C headers provide declarations - avoids duplicate declarations:

```lisp
; Function (type info only)
(declare-fn name [param-types...] -> return-type)

; Type info only
(declare-type name)
(declare-union name)
(declare-struct name)
(declare-var name type)
```

Example:

```lisp
(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)
; No extern emitted - header provides it
```

### Best Practices

- **Use `declare-*` with `include-header`** to avoid duplicate declarations
- **Use `extern-*` for custom C functions** not in standard headers
- **Mixing is allowed**: Use `declare-*` for stdlib, `extern-*` for custom code

### Complete Example

```lisp
; Standard library - use declare with headers
(include-header "stdio.h")
(include-header "stdlib.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))

; Custom C function - use extern (no header)
(extern-fn my_custom_parser [input (Pointer U8)] -> I32)

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Hello from Lisp0!\n"))
    (my_custom_parser (c-str "data"))
    0))
```

### C String Helper

```lisp
(c-str "string literal")
```

Converts string literal to `(Pointer U8)` for C FFI.

---

## Macros

Lisp0 supports compile-time macros for metaprogramming.

### Macro Definition

```lisp
(defmacro name [params]
  body)
```

Example:

```lisp
(defmacro add1 [x]
  `(+ ~x 1))

(add1 5)  ; expands to (+ 5 1)
```

### Syntax Quote

Quote code structure with backtick or `syntax-quote`:

```lisp
`expr
(syntax-quote expr)
```

### Unquote

Substitute parameter value:

```lisp
~x
(unquote x)
```

### Unquote-Splicing

Splice list elements into surrounding context:

```lisp
~@xs
(unquote-splicing xs)
```

Example:

```lisp
(defmacro sum-list [xs]
  `(+ ~@xs))

(sum-list (1 2 3))  ; expands to (+ 1 2 3)
```

### Hygiene: Manual Gensym

Generate unique symbols to avoid variable capture:

```lisp
(gensym)          ; generates G__0, G__1, etc.
(gensym "prefix") ; generates prefix__0, prefix__1, etc.
(gensym 'symbol)  ; uses symbol name as prefix
```

Example:

```lisp
(defmacro let-unique [var val body]
  (let [tmp (gensym "tmp")]
    `(let [~tmp ~val] ~body)))
```

### Hygiene: Auto-Gensym

Automatically generate unique symbols inside syntax-quote:

```lisp
sym#  ; auto-generates unique symbol
```

All instances of the same `sym#` within one syntax-quote get the same unique symbol. Different `sym#` names get different symbols.

Example:

```lisp
(defmacro swap [a b]
  `(let [tmp# ~a]
     (set! ~a ~b)
     (set! ~b tmp#)))

; tmp# is unique, prevents variable capture
```

### Macro Expansion

Macros expand before type checking:

```
Parse → Expand Macros → Type Check → Codegen
```

### Recursive Macros

Macros can call other macros:

```lisp
(defmacro double [x] `(+ ~x ~x))
(defmacro quadruple [x] `(double (double ~x)))
```

---

## Namespaces

Basic namespace support with `ns`:

```lisp
(ns my-namespace)
```

Qualified name resolution for enum variants:

```lisp
(def Color (: Type) (Enum Red Green Blue))
Color/Red
Color/Green
```

---

## Error Handling

The Lisp0 compiler provides detailed error messages.

### Type Errors

```lisp
(def x (: I32) "hello")
; ERROR: TypeMismatch - expected I32, got String
```

### Unbound Variables

```lisp
(def x (: Int) undefined-var)
; ERROR: UnboundVariable - 'undefined-var' not in scope
```

### Argument Count Mismatch

```lisp
(def add (: (-> [Int Int] Int))
  (fn [x y] (+ x y)))

(add 1 2 3)
; ERROR: ArgumentCountMismatch - expected 2 arguments, got 3
```

### Type Synthesis Failure

```lisp
(def x unknown-expr)
; ERROR: CannotSynthesize - unable to infer type
```

### Invalid Type Annotations

```lisp
(def x (: NotAType) 42)
; ERROR: InvalidTypeAnnotation - 'NotAType' is not a valid type
```

---

## Compilation

### Compile to Executable

```bash
lisp0 program.lisp
./output
```

### Compile and Run

```bash
lisp0 program.lisp --run
```

### Compile to Bundle (Dynamic Library)

```bash
lisp0 program.lisp --bundle
```

### Bundle and Run

```bash
lisp0 program.lisp --bundle --run
```

---

## REPL

Launch the interactive REPL:

```bash
lisp0-repl
```

Features:
- Live evaluation of expressions
- Function and macro redefinition
- **Note:** REPL recompiles from scratch each input - no runtime state persistence

---

## Examples

See the `examples/` directory for complete example programs demonstrating all language features.

---

**End of Language Reference**
