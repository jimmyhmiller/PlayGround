# Array API Design Document

## Overview

This document specifies the array feature for the language, enabling fixed-size and dynamic arrays needed for numerical computing applications like machine learning.

## Type System

### Array Types

```lisp
;; Fixed-size array type
(Array ElementType Size)

;; Examples
(Array Int 10)                    ; 10 integers
(Array Float 100)                 ; 100 floats
(Array Bool 5)                    ; 5 booleans
(Array (Pointer Int) 20)          ; 20 pointers to int
```

#### Multi-dimensional Arrays

Multi-dimensional arrays are represented as nested array types:

```lisp
(Array (Array Float 10) 5)        ; 5x10 matrix of floats
(Array (Array (Array Int 3) 4) 2) ; 2x4x3 tensor of ints
```

### Type Checking Rules

1. **Size must be a compile-time constant**: Array sizes must be known at compile time (integer literal or defined constant)
2. **Element type can be any valid type**: Including structs, pointers, and other arrays
3. **Subtyping**: Arrays are invariant in their element type (no subtyping)
4. **Array types are value types**: Passing an array passes the entire array (or use pointers for reference semantics)

## Array Creation

### Stack-Allocated Arrays

```lisp
;; Uninitialized array
(array ElementType Size)

;; Examples
(def weights (: (Array Float 100))
  (array Float 100))

;; Zero-initialized array
(array ElementType Size InitValue)

;; Examples
(def counters (: (Array Int 10))
  (array Int 10 0))
```

### Heap-Allocated Arrays

Use existing pointer operations with array-specific allocators:

```lisp
;; Allocate array on heap
(allocate-array ElementType Size)
;; Returns: (Pointer ElementType)

;; Allocate with initialization
(allocate-array ElementType Size InitValue)

;; Examples
(def large-buffer (: (Pointer Float))
  (allocate-array Float 10000))

(def zeros (: (Pointer Int))
  (allocate-array Int 1000 0))

;; Deallocate when done
(deallocate-array large-buffer)
```

## Array Operations

### Indexing (Read)

```lisp
;; Single-dimensional indexing
(array-ref array index)
;; Returns: ElementType

;; Examples
(def arr (: (Array Int 5)) (array Int 5))
(array-set! arr 0 10)
(array-set! arr 1 20)
(def second (: Int) (array-ref arr 1))  ; 20

;; Multi-dimensional indexing (via nested array-ref)
(def matrix (: (Array (Array Float 3) 2))
  (array (Array Float 3) 2))
(def val (: Float) (array-ref (array-ref matrix 1) 2))
```

#### Alternative Syntax (Future Consideration)

```lisp
;; Reuse struct field access syntax for consistency
(. arr 1)           ; equivalent to (array-ref arr 1)
(. matrix 1 2)      ; equivalent to (array-ref matrix 1 2)
```

### Indexing (Write)

```lisp
;; Single-dimensional write
(array-set! array index value)
;; Returns: Nil (mutates in place)

;; Examples
(def scores (: (Array Int 10)) (array Int 10 0))
(array-set! scores 0 95)
(array-set! scores 1 87)

;; Multi-dimensional write (via nested array-ref)
;; Example
(def grid (: (Array (Array Int 3) 3))
  (array (Array Int 3) 3 (array Int 3 0)))
(array-set! (array-ref grid 1) 1 42)  ; grid[1][1] = 42
```

### Length

```lisp
;; Get array length (always known at compile time)
(array-length array)
;; Returns: Int (the compile-time size constant)

;; Example
(def arr (: (Array Float 100)) (array Float 100))
(def len (: Int) (array-length arr))  ; 100
```

### Pointer to Element

```lisp
;; Get pointer to array element (for passing slices to functions)
(array-ptr array index)
;; Returns: (Pointer ElementType)

;; Example
(def data (: (Array Float 100)) (array Float 100 0.0))
(def ptr-to-middle (: (Pointer Float))
  (array-ptr data 50))

;; Can pass to functions expecting pointers
(some-function ptr-to-middle)
```

### Pointer Indexing

For heap-allocated arrays (pointers), use pointer arithmetic:

```lisp
;; Read from pointer-based array
(pointer-index-read ptr index)
;; Equivalent to: (dereference (pointer-offset ptr index))

;; Write to pointer-based array
(pointer-index-write! ptr index value)

;; Example
(def buffer (: (Pointer Float))
  (allocate-array Float 1000))
(pointer-index-write! buffer 0 3.14)
(pointer-index-write! buffer 1 2.71)
(def first (: Float) (pointer-index-read buffer 0))
```

## Memory Model

### Stack vs Heap

- **Stack arrays**: Created with `(array T N)` or literals, automatically deallocated when out of scope
- **Heap arrays**: Created with `(allocate-array T N)`, must be manually deallocated with `(deallocate-array ptr)`

### Ownership and Copying

- Arrays are **value types**: Assignment and passing copies the entire array
- For reference semantics, use pointers: `(Pointer (Array T N))` or heap-allocated `(Pointer T)`
- Multi-dimensional arrays involve nested copying

```lisp
;; Value semantics - copies entire array
(def a (: (Array Int 3)) (array Int 3 0))
(array-set! a 0 1)
(array-set! a 1 2)
(array-set! a 2 3)
(def b (: (Array Int 3)) a)  ; b is a copy
(array-set! b 0 99)
;; a[0] is still 1, b[0] is now 99

;; Reference semantics via pointer
(def arr (: (Array Int 3)) (array Int 3))
(def ptr (: (Pointer (Array Int 3))) (address-of arr))
;; Can mutate through pointer
```

### Bounds Checking

- **Static bounds**: Index must be `Int` type at compile time
- **Runtime bounds checking**: Optional (implementation choice)
  - Debug mode: Insert runtime checks, panic on out-of-bounds
  - Release mode: No checks (C-like behavior for performance)

## Code Generation (C Backend)

### Type Mapping

```lisp
(Array Int 10)           → int[10]
(Array Float 100)        → float[100]
(Array (Array Int 5) 3)  → int[3][5]
(Pointer Int)            → int*
```

### Operation Mapping

```lisp
(array Int 10)                      → int arr[10]
(array Int 10 0)                    → int arr[10]; for loop init
(array-ref arr 5)                   → arr[5]
(array-set! arr 5 10)               → arr[5] = 10
(array-ref (array-ref matrix 2) 3)  → matrix[2][3]
(array-length arr)                  → (compile-time constant)
(array-ptr arr 10)                  → &arr[10]
(allocate-array Float 100)          → (float*)malloc(100 * sizeof(float))
(deallocate-array ptr)              → free(ptr)
(pointer-index-read ptr 5)          → ptr[5]
(pointer-index-write! ptr 5 v)      → ptr[5] = v
```

## Integration with Existing Features

### With Pointers

Arrays and pointers interoperate naturally:

```lisp
;; Get pointer to array element
(def arr (: (Array Int 10)) (array Int 10 0))
(def ptr (: (Pointer Int)) (array-ptr arr 0))

;; Can index pointers like arrays
(pointer-index-write! ptr 5 42)
```

### With Structs

Arrays can be struct fields, and structs can be array elements:

```lisp
(def Matrix (: Type)
  (Struct
    [rows Int]
    [cols Int]
    [data (Pointer Float)]))

(def m (: Matrix)
  (Matrix 10 10 (allocate-array Float 100)))

(pointer-index-write! (. m data) 0 3.14)
```

### With Functions

Arrays can be function parameters and return values:

```lisp
;; Pass by value (copies entire array)
(def sum-array (: (-> [(Array Int 10)] Int))
  (fn [arr]
    (let [total (: Int) 0]
      (c-for [i (: Int) 0] (< i 10) (+ i 1)
        (set! total (+ total (array-ref arr i))))
      total)))

;; Pass by reference (via pointer)
(def fill-array (: (-> [(Pointer Float) Int Float] Nil))
  (fn [arr size value]
    (c-for [i (: Int) 0] (< i size) (+ i 1)
      (pointer-index-write! arr i value))
    nil))
```

## Usage Examples

### Simple Array Operations

```lisp
;; Create and initialize
(def scores (: (Array Int 5)) (array Int 5))
(array-set! scores 0 85)
(array-set! scores 1 92)
(array-set! scores 2 78)
(array-set! scores 3 95)
(array-set! scores 4 88)

;; Read
(def first-score (: Int) (array-ref scores 0))  ; 85

;; Write
(array-set! scores 2 80)  ; Update third score

;; Iterate
(c-for [i (: Int) 0] (< i (array-length scores)) (+ i 1)
  (printf "Score %d: %d\n" i (array-ref scores i)))
```

### Matrix Operations

```lisp
;; 3x3 identity matrix
(def matrix (: (Array (Array Float 3) 3))
  (array (Array Float 3) 3 (array Float 3 0.0)))

;; Set diagonal to 1.0
(array-set! (array-ref matrix 0) 0 1.0)
(array-set! (array-ref matrix 1) 1 1.0)
(array-set! (array-ref matrix 2) 2 1.0)

;; Access element
(def center (: Float) (array-ref (array-ref matrix 1) 1))  ; 1.0

;; Nested iteration
(c-for [i (: Int) 0] (< i 3) (+ i 1)
  (c-for [j (: Int) 0] (< j 3) (+ j 1)
    (array-set! (array-ref matrix i) j
      (* (array-ref (array-ref matrix i) j) 2.0))))
```

### Large Dynamic Arrays

```lisp
;; Neural network weights (heap-allocated)
(def layer-size (: Int) 1000)
(def weights (: (Pointer Float))
  (allocate-array Float (* layer-size layer-size)))

;; Initialize
(c-for [i (: Int) 0] (< i (* layer-size layer-size)) (+ i 1)
  (pointer-index-write! weights i 0.01))

;; Use
(def weight-0-1 (: Float)
  (pointer-index-read weights (+ (* 0 layer-size) 1)))

;; Clean up
(deallocate-array weights)
```

### Passing Arrays to Functions

```lisp
;; Stack array passed by value
(def double-elements (: (-> [(Array Int 5)] (Array Int 5)))
  (fn [arr]
    (let [result (: (Array Int 5)) (array Int 5)]
      (c-for [i (: Int) 0] (< i 5) (+ i 1)
        (array-set! result i (* (array-ref arr i) 2)))
      result)))

;; Heap array passed by reference
(def scale-in-place (: (-> [(Pointer Float) Int Float] Nil))
  (fn [arr len scale]
    (c-for [i (: Int) 0] (< i len) (+ i 1)
      (pointer-index-write! arr i
        (* (pointer-index-read arr i) scale)))
    nil))
```

## Implementation Phases

### Phase 1: Fixed-Size Stack Arrays (Core)
- Type representation: `(Array T N)`
- Type checking for array types
- `(array T N)` and `(array T N init)` constructors
- `(array-ref arr idx)` for reading
- `(array-set! arr idx val)` for writing
- `(array-length arr)`
- C code generation for 1D arrays

### Phase 2: Multi-dimensional Arrays
- Nested array type support
- Nested `array-ref` and `array-set!` calls
- C code generation for N-dimensional arrays

### Phase 3: Pointer Integration (Heap Arrays)
- `(array-ptr arr idx)` for element pointers
- `(allocate-array T N)` heap allocation
- `(deallocate-array ptr)` deallocation
- `(pointer-index-read ptr idx)` and `(pointer-index-write! ptr idx val)`

### Phase 4: Optimizations & Sugar (Future)
- Optional bounds checking
- Possible syntax sugar: `(. arr idx)`
- Array literal syntax `[...]`
- Array initialization helpers
- Array copy/fill utilities

## Open Questions

1. **Array comparison**: Should `(= arr1 arr2)` do element-wise comparison or pointer equality?
2. **Array slicing**: Support `(array-slice arr start end)` → `(Pointer T)`?
3. **Bounds checking policy**: Runtime checks always, debug-only, or never?
4. **Zero-initialization default**: Should `(array T N)` zero-initialize by default?
5. **Const arrays**: Read-only array types for literals and constants?
6. **Array return optimization**: Pass-by-hidden-pointer for large array returns?

## Future Extensions

- **Dynamic sizing**: True dynamic arrays with runtime-known sizes (like C99 VLAs)
- **Array comprehensions**: `[expr | var <- range, condition]` syntax
- **Built-in array functions**: `map`, `fold`, `zip` for functional array operations
- **SIMD support**: Vector types and operations for numerical performance
- **Bounds-checked safe mode**: Compiler flag for debug builds
