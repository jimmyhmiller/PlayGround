# Lispier: Lispy Syntax for MLIR

A succinct, complete S-expression based syntax for MLIR that maps directly to the generic form and C API.

## Design Principles

1. **Generic form mapping**: Syntax maps 1:1 to MLIR's generic operation form
2. **Type inference**: Infer types where possible, default to i64 for integers
3. **Clean names**: No `%` prefix for SSA values
4. **S-expression based**: Everything is a form, minimal special syntax
5. **C API compatible**: Designed to work with MLIR's C API, not custom parsers

## Core Syntax

### Operations

```lisp
(operation-name operand1 operand2 ...)
(operation-name {:attributes...} operand1 operand2 ...)
(operation-name {:attributes...} operand1 operand2 ... (region...) (region...))
```

**Rules:**
- Operation name is a symbol (e.g., `arith.addi`, `func.call`)
- Operations require their dialect to be imported (see "Dialects and Namespaces" section)
- Attributes are optional, represented as a map `{:key value}`
- Operands follow attributes (or come first if no attributes)
- Regions come after operands as `(do ...)` forms

**Examples:**
```lisp
(require-dialect arith func)

(arith.addi x y)
(arith.constant {:value 42})
(func.call "add" a b)
```

### Attributes

Attributes are compile-time constant data in a map:

```lisp
{:key1 value1 :key2 value2}
```

**Special attributes:**
- `:successors` - array of block labels for control flow
- `:operand_segment_sizes` - array indicating how to partition operands

**Examples:**
```lisp
{:predicate "sgt"}
{:sym_name "main" :function_type (-> [i64 i64] [i64])}
{:successors [^bb1 ^bb2] :operand_segment_sizes [1 2 1]}
```

**AI worries: attribute round-trip fidelity**

MLIR’s generic form distinguishes attribute *kinds* (not just values). To round-trip losslessly, the syntax may need explicit literals for these cases:

- **Elements attrs (dense/sparse) with element type**: `"arith.constant"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>`. Needs both the `dense` kind and the trailing `tensor<2xi32>`.
- **Typed arrays/dictionaries**: `array<i32: 1, 2>` and `dictionary [a = 1, b = 2]` carry element typing and ordering beyond a generic vector/map.
- **Booleans and unit flags**: `true`/`false`, and presence-only `unit` such as `arith.fastmath = unit`.
- **Symbol references**: `symbol_ref @foo` or nested `symbol_ref @root::@leaf` (e.g. `"func.call"() {callee = @foo}`). They are not strings.
- **Opaque/dialect-specific**: `opaque<"mydialect", "payload">` retains dialect and payload.
- **Type attributes**: attributes whose value is itself a type, e.g. `{element_type = i32}`.

One possible Lispy spelling for these, if desired, could be `(dense [1 2] tensor<2xi32>)`, `(sparse [[0] [42]] tensor<?xi64>)`, `(array i32 [1 2 3])`, `(dictionary {:a 1 :b 2})`, `(unit)`, `(symbol_ref foo)`, `(opaque "mydialect" "payload")`, etc.

### Successors and Control Flow

Successors are block references used for control flow, represented via the `:successors` attribute:

```lisp
(cf.br {:successors [^bb1]})
(cf.cond_br {:successors [^bb1 ^bb2] :operand_segment_sizes [1 2 1]}
            cond x y z)
```

The `:operand_segment_sizes` attribute specifies how operands are partitioned:
- For `cf.cond_br` with sizes `[1 2 1]`: 1 operand for condition, 2 for ^bb1, 1 for ^bb2

### Regions and Blocks

**Regions** contain blocks of operations:
```lisp
(do
  (block [args...]
    operations...))
```

**Blocks** can have labels and arguments:
```lisp
(block ^label [arg1 arg2]
  operations...)
```

Block arguments act like phi nodes in traditional SSA - they receive values from predecessor blocks.

**Examples:**
```lisp
(do
  (block
    (cf.br {:successors [^loop]} 0))

  (block ^loop [iter]
    (def next (arith.addi iter 1))
    (cf.br {:successors [^loop]} next)))
```

## Bindings

### def - Single Binding

```lisp
(def name expression)
```

Binds the result of an expression to a name.

**Examples:**
```lisp
(def x 42)
(def sum (arith.addi x y))
(def result (func.call "compute" a b))
```

### let - Multiple Bindings

```lisp
(let [name1 expr1
      name2 expr2
      ...]
  body)
```

Creates multiple bindings, each can reference previous bindings.

**Examples:**
```lisp
(let [x 42
      y 10
      sum (arith.addi x y)]
  (func.return sum))
```

### Destructuring - Multiple Return Values

```lisp
(let [[name1 name2] multi-return-expr]
  body)
```

Operations that return multiple values can be destructured:

**Examples:**
```lisp
(let [[token result] (async.execute (do (async.yield 42)))]
  (async.await result))
```

## Types

### Type Representation

Types are symbols that may contain angle brackets `<>`:

```lisp
i64
f32
i1
index
memref<128x128xf32>
!llvm.ptr<i8>
!llvm.ptr<array<14xi8>>
tensor<10xi32>
```

Types with `<>` are tokenized as complete units, respecting nesting.

### Type Annotations

Use `(: value type)` to explicitly specify types:

```lisp
(: 42 i32)                    ; 42 as i32 instead of default i64
(: arg0 i64)                  ; function argument with type
```

**Examples:**
```lisp
(let [x (: 42 i32)
      y (: 10 i32)]
  (arith.addi x y))

(block [(: arg0 i64) (: arg1 i64)]
  ...)
```

### Type Inference

- **Numbers**: Default to `i64` for integers, `f64` for floating point
- **Operations**: Infer result types from operand types when possible
- **Context**: Types can be inferred from function signatures or expected types

### Function Types

Function types use special arrow syntax:

```lisp
(-> [arg-types...] [return-types...])
```

**Examples:**
```lisp
(-> [i64 i64] [i64])                    ; (i64, i64) -> i64
(-> [] [i32])                           ; () -> i32
(-> [i32 f64] [i32 i32])                ; multiple returns
(-> [(varargs !llvm.ptr<i8>)] [i32])    ; variadic arguments
```

### Affine Maps

Affine maps are symbols like types:

```lisp
affine_map<()->(0)>                     ; constant 0
affine_map<()->(128)>                   ; constant 128
affine_map<(d0,d1)->(d0*16+d1)>        ; d0*16 + d1
affine_map<()[s0]->(s0)>                ; symbol s0
```

Types may contain embedded attribute or layout payloads; these are valid and should be treated as single type tokens:

```lisp
tensor<?x?xf32, #map>                       ; encoding after a comma
memref<4x4xf16, affine_map<(i, j) -> (j, i)>>
!mhlo.complex<f32>
!tt.ptr<space = #tt.cpu>
!shape.value_shape<i32, tensor<2xi64>>
```

Used in affine operations:
```lisp
(affine.for {:lower_bound affine_map<()->(0)>
             :upper_bound affine_map<()->(128)>
             :step 1}
  (do (block [i] ...)))
```

## Dialects and Namespaces

MLIR organizes operations into **dialects** (e.g., `arith`, `func`, `memref`, `scf`). Lispier requires **explicit dialect imports** to use any operations. This enables better tooling, early error detection, and clear dependency tracking.

### Importing Dialects

Before using any operations, you must import the dialect. Lispier provides three import styles:

#### 1. Basic Import (Fully Qualified Access)

```lisp
(require-dialect arith)

(def sum (arith.addi x y))     ; dot notation for fully qualified names
(def product (arith.muli a b))
```

**When to use:** When you want explicit, self-documenting code, or when using a dialect sparingly.

#### 2. Aliased Import (Slash Notation)

```lisp
(require-dialect [arith :as a])

(def sum (a/addi x y))         ; slash notation with alias
(def product (a/muli a b))
```

**When to use:** When you use a dialect frequently and want concise code. This is the most common style.

#### 3. Unqualified Import (Bare Names)

```lisp
(use-dialect arith)

(def sum (addi x y))           ; no prefix needed
(def product (muli a b))
```

**When to use:** When writing DSL-like code focused on a single dialect, or for arithmetic-heavy sections.

**Warning:** Using multiple dialects with `use-dialect` can cause ambiguity if operations have the same name. Prefer aliases for multi-dialect code.

### Import Syntax

You can combine different import styles:

```lisp
(require-dialect arith             ; basic import: arith.addi
                 [memref :as m]    ; aliased: m/alloc
                 [scf :as s])      ; aliased: s/for
(use-dialect func)                 ; unqualified: call, return
```

Or write them separately:

```lisp
(require-dialect arith)
(require-dialect [memref :as m])
(use-dialect func)
```

### Symbol Resolution

When you use an operation, Lispier resolves it in this order:

1. **Slash notation** (`alias/operation`) → Resolve from `require-dialect` alias
   - `a/addi` where `[arith :as a]` → `arith.addi`

2. **Dot notation** (`dialect.operation`) → Fully qualified name (dialect must be imported)
   - `arith.addi` with `(require-dialect arith)` → `arith.addi`
   - **Error** if `arith` not imported

3. **Unqualified** (`operation`) → Search `use-dialect` imports
   - `addi` with `(use-dialect arith)` → `arith.addi`
   - **Error** if not found in any used dialect
   - **Error** if found in multiple used dialects (ambiguous)

### Scope

Dialect imports are **scoped** to the module or function where they appear:

**Module-level imports** (available to all functions):
```lisp
(module
  (require-dialect [arith :as a]
                   [memref :as m])
  (do
    (func.func {:sym_name "example" ...}
      (do
        (block [...]
          (def sum (a/addi x y))      ; uses module-level import
          ...)))))
```

**Function-level imports** (override/extend module scope):
```lisp
(module
  (require-dialect [arith :as a])     ; module-level

  (do
    (func.func {:sym_name "specialized" ...}
      (do
        (require-dialect [vector :as v])  ; function-level
        (use-dialect scf)                 ; function-level

        (block [...]
          (def x (a/addi 1 2))        ; module-level arith
          (def y (v/broadcast x))     ; function-level vector
          (def z (for ...))           ; function-level scf via use-dialect
          ...)))))
```

### Why Explicit Imports?

**Early error detection:** Typos fail at parse time, not runtime
```lisp
(require-dialect arth)   ; ERROR: unknown dialect 'arth'. Did you mean 'arith'?
```

**Better tooling:** IDEs can provide accurate auto-complete, jump-to-definition, and unused import warnings.

**Clear dependencies:** Every module declares exactly what it uses, making code easier to understand and analyze.

**Scales well:** As MLIR grows, explicit imports keep your namespace clean and focused.

### Custom Dialects

Custom MLIR dialects integrate seamlessly. After registering a dialect `myproject` with MLIR:

```lisp
(require-dialect myproject)           ; basic
(def x (myproject.custom_op args))

(require-dialect [myproject :as mp])  ; aliased
(def x (mp/custom_op args))

(use-dialect myproject)               ; unqualified
(def x (custom_op args))
```

## Literals

### Numbers

Numbers are written directly:

```lisp
42          ; i64 by default
3.14159     ; f64 by default
0           ; i64
-10         ; i64
```

For specific types, use type annotations:
```lisp
(: 42 i32)
(: 3.14 f32)
```

`(: expr type)` is **not** a runtime cast. It is a compile-time type annotation that controls the declared result type of the wrapped expression/operation. Use it whenever the operation cannot infer its result type (zero-operand ops, ops whose type comes only from attributes, multi-result ops):

```lisp
(: (arith.constant {:value 42}) i32)             ; sets the op result type to i32
(: (foo.new_handle) !foo.handle)                 ; zero-operand op that must declare its result type
(: (foo.split value) [i32 i1])                   ; multi-result op annotated as a tuple of result types
```

### Strings

String literals are written in double quotes:

```lisp
"hello"
"Hello, World!\00"
```

## Complete Examples

### Simple Function

**MLIR:**
```mlir
func.func @add(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.addi %arg0, %arg1 : i64
  func.return %0 : i64
}
```

**Lispier (fully qualified):**
```lisp
(require-dialect arith func)

(func.func {:sym_name "add"
            :function_type (-> [i64 i64] [i64])}
  (do
    (block [(: arg0 i64) (: arg1 i64)]
      (let [result (arith.addi arg0 arg1)]
        (func.return result)))))
```

**Lispier (with aliases):**
```lisp
(require-dialect [arith :as a]
                 [func :as f])

(f/func {:sym_name "add"
         :function_type (-> [i64 i64] [i64])}
  (do
    (block [(: arg0 i64) (: arg1 i64)]
      (let [result (a/addi arg0 arg1)]
        (f/return result)))))
```

**Lispier (with use-dialect):**
```lisp
(require-dialect func)
(use-dialect arith)

(func.func {:sym_name "add"
            :function_type (-> [i64 i64] [i64])}
  (do
    (block [(: arg0 i64) (: arg1 i64)]
      (let [result (addi arg0 arg1)]
        (func.return result)))))
```

### Conditional Branch

**MLIR:**
```mlir
func.func @max(%a: i64, %b: i64) -> i64 {
  %cond = arith.cmpi sgt, %a, %b : i64
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  func.return %a : i64
^bb2:
  func.return %b : i64
}
```

**Lispier:**
```lisp
(require-dialect arith cf func)

(func.func {:sym_name "max"
            :function_type (-> [i64 i64] [i64])}
  (do
    (block [(: a i64) (: b i64)]
      (def cond (arith.cmpi {:predicate "sgt"} a b))
      (cf.cond_br {:successors [^return_a ^return_b]
                   :operand_segment_sizes [1 0 0]}
                  cond))
    (block ^return_a
      (func.return a))
    (block ^return_b
      (func.return b))))
```

**Lispier (with aliases):**
```lisp
(require-dialect [arith :as a]
                 [cf :as c]
                 [func :as f])

(f/func {:sym_name "max"
         :function_type (-> [i64 i64] [i64])}
  (do
    (block [(: a i64) (: b i64)]
      (def cond (a/cmpi {:predicate "sgt"} a b))
      (c/cond_br {:successors [^return_a ^return_b]
                  :operand_segment_sizes [1 0 0]}
                 cond))
    (block ^return_a
      (f/return a))
    (block ^return_b
      (f/return b))))
```

### Loop with Block Arguments

**MLIR:**
```mlir
func.func @countdown(%n: i64) -> i64 {
  cf.br ^loop(%n : i64)
^loop(%iter: i64):
  %is_zero = arith.cmpi eq, %iter, 0 : i64
  cf.cond_br %is_zero, ^done, ^continue(%iter : i64)
^continue(%val: i64):
  %next = arith.subi %val, 1 : i64
  cf.br ^loop(%next : i64)
^done:
  func.return 0 : i64
}
```

**Lispier:**
```lisp
(require-dialect arith cf func)

(func.func {:sym_name "countdown"
            :function_type (-> [i64] [i64])}
  (do
    (block [(: n i64)]
      (cf.br {:successors [^loop]} n))

    (block ^loop [(: iter i64)]
      (def is_zero (arith.cmpi {:predicate "eq"} iter 0))
      (cf.cond_br {:successors [^done ^continue]
                   :operand_segment_sizes [1 0 1]}
                  is_zero iter))

    (block ^continue [(: val i64)]
      (def next (arith.subi val 1))
      (cf.br {:successors [^loop]} next))

    (block ^done
      (func.return 0))))
```

### Switch Statement

**MLIR:**
```mlir
"cf.switch"(%flag, %a, %b, %c) [^bb1, ^bb2, ^bb3] {
  case_values = dense<[42, 43]> : tensor<2xi32>,
  operand_segment_sizes = dense<[1, 1, 1, 1]> : vector<4xi32>
} : (i32, i32, i32, i32) -> ()
```

**Lispier:**
```lisp
(require-dialect cf)

(cf.switch {:successors [^bb1 ^bb2 ^bb3]
            :case_values [42 43]
            :operand_segment_sizes [1 1 1 1]}
           flag a b c)
```

### SCF If-Else

**MLIR:**
```mlir
%result = scf.if %condition -> i64 {
  %c1 = arith.constant 1 : i64
  scf.yield %c1 : i64
} else {
  %c0 = arith.constant 0 : i64
  scf.yield %c0 : i64
}
```

**Lispier:**
```lisp
(require-dialect scf)

(def result
  (scf.if condition
    (do
      (def c1 1)
      (scf.yield c1))
    (do
      (def c0 0)
      (scf.yield c0))))
```

### Affine Loop

**MLIR:**
```mlir
affine.for %i = 0 to 10 {
  affine.for %j = 0 to 20 {
    %sum = arith.addi %i, %j : index
    affine.store %sum, %buffer[%i, %j] : memref<10x20xindex>
  }
}
```

**Lispier:**
```lisp
(require-dialect affine arith)

(affine.for {:lower_bound affine_map<()->(0)>
             :upper_bound affine_map<()->(10)>
             :step 1}
  (do
    (block [(: i index)]
      (affine.for {:lower_bound affine_map<()->(0)>
                   :upper_bound affine_map<()->(20)>
                   :step 1}
        (do
          (block [(: j index)]
            (def sum (arith.addi i j))
            (affine.store sum buffer [i j])))))))
```

### Module with Multiple Functions

**Lispier:**
```lisp
(module
  (require-dialect arith func)

  (do
    (func.func {:sym_name "helper"
                :function_type (-> [i64] [i64])}
      (do
        (block [(: x i64)]
          (def result (arith.muli x 2))
          (func.return result))))

    (func.func {:sym_name "main"
                :function_type (-> [] [i64])}
      (do
        (block []
          (def val (func.call "helper" 21))
          (func.return val))))))
```

**Lispier (with aliases):**
```lisp
(module
  (require-dialect [arith :as a]
                   [func :as f])

  (do
    (f/func {:sym_name "helper"
             :function_type (-> [i64] [i64])}
      (do
        (block [(: x i64)]
          (def result (a/muli x 2))
          (f/return result))))

    (f/func {:sym_name "main"
             :function_type (-> [] [i64])}
      (do
        (block []
          (def val (f/call "helper" 21))
          (f/return val))))))
```

### Memory Operations

**MLIR:**
```mlir
%buffer = memref.alloc() : memref<10xi64>
%c0 = arith.constant 0 : index
%val = arith.constant 42 : i64
memref.store %val, %buffer[%c0] : memref<10xi64>
%loaded = memref.load %buffer[%c0] : memref<10xi64>
```

**Lispier:**
```lisp
(require-dialect memref func)

(let [buffer (memref.alloc)
      c0 (: 0 index)
      val 42
      _ (memref.store val buffer [c0])
      loaded (memref.load buffer [c0])]
  (func.return loaded))
```

### Complete Example: Matrix Multiplication with All Dialect Styles

This example demonstrates all three dialect notation styles working together:

```lisp
(module
  ;; Module-level dialect declarations
  (require-dialect [arith :as a]
                   [memref :as m]
                   [scf :as s])

  (do
    ;; Function using fully qualified names for clarity
    (func.func {:sym_name "init_matrix"
                :function_type (-> [memref<128x128xf32>] [])}
      (do
        (block [(: mat memref<128x128xf32>)]
          (def zero (: 0.0 f32))
          (def c0 (: 0 index))
          (def c128 (: 128 index))

          ;; Using fully qualified dialect names
          (scf.for {:lower_bound affine_map<()->(0)>
                    :upper_bound affine_map<()->(128)>
                    :step 1}
            (do
              (block [(: i index)]
                (scf.for {:lower_bound affine_map<()->(0)>
                          :upper_bound affine_map<()->(128)>
                          :step 1}
                  (do
                    (block [(: j index)]
                      (memref.store zero mat [i j])))))))
          (func.return))))

    ;; Function using dialect aliases for conciseness
    (func.func {:sym_name "matmul"
                :function_type (-> [memref<128x128xf32>
                                    memref<128x128xf32>
                                    memref<128x128xf32>] [])}
      (do
        (block [(: a memref<128x128xf32>)
                (: b memref<128x128xf32>)
                (: c memref<128x128xf32>)]

          ;; Using aliases (a/, m/, s/) for readability
          (s/for {:lower_bound affine_map<()->(0)>
                  :upper_bound affine_map<()->(128)>
                  :step 1}
            (do
              (block [(: i index)]
                (s/for {:lower_bound affine_map<()->(0)>
                        :upper_bound affine_map<()->(128)>
                        :step 1}
                  (do
                    (block [(: j index)]
                      (def sum_init (: 0.0 f32))

                      (def sum
                        (s/for {:lower_bound affine_map<()->(0)>
                                :upper_bound affine_map<()->(128)>
                                :step 1}
                          (do
                            (block [(: k index) (: sum_iter f32)]
                              (def a_val (m/load a [i k]))
                              (def b_val (m/load b [k j]))
                              (def prod (a/mulf a_val b_val))
                              (def new_sum (a/addf sum_iter prod))
                              (s/yield new_sum)))
                          sum_init))

                      (m/store sum c [i j])))))))
          (func.return))))

    ;; Function using use-dialect for arithmetic-heavy code
    (func.func {:sym_name "scale_matrix"
                :function_type (-> [memref<128x128xf32> f32] [])}
      (do
        ;; Function-scoped use-dialect
        (use-dialect arith)
        (require-dialect [memref :as m]
                        [scf :as s])

        (block [(: mat memref<128x128xf32>) (: scale f32)]
          (s/for {:lower_bound affine_map<()->(0)>
                  :upper_bound affine_map<()->(128)>
                  :step 1}
            (do
              (block [(: i index)]
                (s/for {:lower_bound affine_map<()->(0)>
                        :upper_bound affine_map<()->(128)>
                        :step 1}
                  (do
                    (block [(: j index)]
                      (def val (m/load mat [i j]))
                      ;; Using unqualified arith ops
                      (def scaled (mulf val scale))
                      (m/store scaled mat [i j])))))))
          (func.return))))

    ;; Main function mixing all styles
    (func.func {:sym_name "main"
                :function_type (-> [] [i32])}
      (do
        (block []
          ;; Fully qualified for one-off operations
          (def a (memref.alloc))
          (def b (memref.alloc))
          (def c (memref.alloc))

          ;; Using full names for inter-dialect calls
          (func.call "init_matrix" a)
          (func.call "init_matrix" b)
          (func.call "init_matrix" c)
          (func.call "matmul" a b c)

          ;; Using alias for final arithmetic
          (def scale (: 2.0 f32))
          (func.call "scale_matrix" c scale)

          ;; Return
          (def result (: 0 i32))
          (func.return result))))))
```

**This example shows:**
- **Module-level declarations** that apply to all functions
- **Function-level declarations** that override or add to module scope
- **Fully qualified names** (`memref.alloc`) for clarity when mixing many dialects
- **Aliases** (`a/mulf`, `m/load`, `s/for`) for frequently used operations
- **Unqualified** (`mulf`) in arithmetic-heavy sections with `use-dialect`
- **All three styles mixed** in the same module based on what's most readable

## Locations (proposed)

MLIR ops, operands, blocks, and regions can carry `loc(...)` metadata for source mapping. One possible spelling is to wrap any form with `loc` so it remains distinct from attributes:

```lisp
(loc (arith.addi x y) (file "foo.mlir" 12 3))             ; op location
(loc x (name "arg0"))                                     ; operand-specific location
(loc (block ^bb1 [...]) (fused "loop" [(file "f.mlir" 4 2)]))
```

Location payloads can express file/line/column, named, fused, callsite stacks, or unknown, matching MLIR's `loc(...)` forms.

## Summary

This syntax provides:
- **1:1 mapping** to MLIR's generic form and C API
- **Clean, Lispy style** with minimal special syntax
- **Explicit dialect imports** for better tooling, early error detection, and clear dependencies
- **Flexible notation** supporting three styles:
  - Fully qualified: `(require-dialect arith)` → `arith.addi`
  - Aliased: `(require-dialect [arith :as a])` → `a/addi`
  - Unqualified: `(use-dialect arith)` → `addi`
- **Type inference** where possible, explicit annotations when needed
- **Uniform treatment** of types, affine maps, and other structured data as symbols
- **Clear representation** of regions, blocks, successors, and control flow

The syntax is designed to be both human-readable and straightforward to implement using standard S-expression parsing techniques. The dialect system provides the ergonomics and safety of Clojure's namespace system while maintaining the clarity and explicit structure of MLIR's dialect organization.
