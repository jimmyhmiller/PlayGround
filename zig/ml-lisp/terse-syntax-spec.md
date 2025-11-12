# MLIR-Lisp Terse Syntax Specification

**Version:** 1.0 (Draft)
**Date:** 2025-01-09

## Overview

This document specifies the terse, Lisp-like syntax for MLIR operations. The design goals are:

1. **Concise** - Remove verbose `(operation ...)` wrappers
2. **Functional** - Scoped `let` bindings, implicit returns, expression-oriented
3. **Type-safe** - Explicit type annotations where needed, inference where possible
4. **MLIR-native** - Direct mapping to MLIR constructs, no abstraction loss
5. **Macro-friendly** - Built from primitives, high-level constructs via macros

## Core Principles

- **No `%` prefix** - Variables are plain identifiers: `arg0`, `result`, `iter`
- **Symbols not strings** - Attribute values like `{:sym_name add-one}`, not `"add-one"`
- **Uniform type annotations** - Always `(: value type)` format
- **Function types** - `(-> (input-types...) (result-types...))` format
- **Scoped let** - Clojure-style: `(let [bindings...] body)`
- **Implicit returns** - Last expression in block is the result
- **Map insertion order** - For inline successors `^{:key1 ... :key2 ...}`

---

## 1. Operations (Terse Form)

### Basic Pattern
```lisp
;; (operation-name {attributes} operands...)
(arith.addi {} a b)
(arith.constant {:value 42})
(func.call {:callee @my-func} arg1 arg2)
```

### Optional Attributes
```lisp
;; Can omit empty attributes? (TBD)
(arith.addi a b)  ;; If we allow omitting {}
```

### Type Annotations
```lisp
;; Explicit result type when needed
(: (arith.addi {} a b) i32)

;; Most operations infer type from MLIR definition
(arith.addi {} a b)  ;; Type inferred from operand types
```

---

## 2. Let Bindings

### Scoped Bindings (Clojure-style)
```lisp
;; Single binding
(let [(: x (arith.addi {} a b))]
  (arith.muli {} x x))

;; Multiple bindings (sequential scope)
(let [(: x (arith.addi {} a b))
      (: y (arith.muli {} x 2))
      (: z (arith.addi {} y x))]
  z)  ;; Implicit return
```

### Type Inference in Let
```lisp
;; Type can be inferred from operation result
(let [x (arith.addi {} a b)]  ;; Type inferred
  x)

;; Or explicitly annotated
(let [(: x i32) (some.op {})]
  x)
```

---

## 3. Type Annotations

### Uniform Syntax: `(: value type)`

```lisp
;; Function arguments
[(: arg0 i32) (: arg1 i64)]

;; Let bindings
(let [(: result (arith.addi {} a b))]
  result)

;; Explicit type on expression
(: (some.op {} arg) i32)

;; Block arguments
[(: iter i32) (: sum i32)]
```

### Function Types
```lisp
;; Pattern: (-> (input-types...) (result-types...))

;; Single input, single output
(-> (i32) (i32))

;; Multiple inputs
(-> (i32 i64 f32) (i64))

;; Multiple outputs
(-> (i32) (i32 i1))

;; No inputs
(-> () (i32))

;; No outputs
(-> (i32 i64) ())
```

---

## 4. Do (Sequencing)

### Pattern: Evaluate Multiple Expressions, Return Last

```lisp
;; Simple sequencing
(do
  (llvm.store {} val ptr)
  (llvm.store {} val2 ptr2)
  (arith.constant {:value 0}))  ;; This is returned

;; Common in successor blocks
(do
  (: x (arith.addi {} a b))
  (: y (arith.muli {} x 2))
  (func.return {} y))
```

---

## 5. Functions

### Function Definition

```lisp
(func.func {:sym_name function-name
            :function_type (-> (input-types...) (result-types...))}
  [(: arg0 type0) (: arg1 type1) ...]
  body-expression)
```

### Examples

```lisp
;; Simple function
(func.func {:sym_name add-one :function_type (-> (i32) (i32))}
  [(: arg0 i32)]
  (let [(: c1 (arith.constant {:value 1}))]
    (arith.addi {} arg0 c1)))

;; Multiple arguments
(func.func {:sym_name add :function_type (-> (i32 i32) (i32))}
  [(: a i32) (: b i32)]
  (arith.addi {} a b))

;; No arguments
(func.func {:sym_name get-zero :function_type (-> () (i32))}
  []
  (arith.constant {:value 0}))
```

---

## 6. Inline Successor Blocks

### Pattern: `^{:key1 body1 :key2 body2 ...}`

Successor order follows map insertion order.

### Simple Conditional

```lisp
(cf.cond_br {} cond
  ^{:then true-val
    :else false-val})
```

### With Operations

```lisp
(cf.cond_br {} cond
  ^{:then (let [(: doubled (arith.muli {} val val))]
            (func.return {} doubled))
    :else (func.return {} val)})
```

### Switch Statement

```lisp
(llvm.switch {:case_values [0 1 2]} val
  ^{:default (func.return {} default-val)
    :case0 (func.return {} zero-val)
    :case1 (func.return {} one-val)
    :case2 (func.return {} two-val)})
```

### Block Arguments in Successors

```lisp
(cf.cond_br {} cond
  ^{:then ([(: x i32) (: y i32)]  ;; Block arguments
            (arith.addi {} x y))
    :else default})
```

### Key Order Semantics

For `cf.cond_br`:
- First key (`:then`) → successor[0] (true branch)
- Second key (`:else`) → successor[1] (false branch)

For `llvm.switch`:
- First key (`:default`) → successor[0] (default destination)
- Remaining keys (`:case0`, `:case1`, ...) → successor[1], successor[2], ...

---

## 7. Explicit Blocks

For loops, complex control flow, or forward references.

### Pattern

```lisp
(block ^label [(: arg0 type0) (: arg1 type1) ...]
  body)
```

### Loop Example

```lisp
(block ^loop [(: iter i32) (: sum i32)]
  (let [(: cond (arith.cmpi {:predicate slt} iter limit))]
    (cf.cond_br {} cond
      ^{:continue (let [(: new-sum (arith.addi {} sum iter))
                        (: next-iter (arith.addi {} iter one))]
                    (cf.br {} ^loop next-iter new-sum))
        :exit (func.return {} sum)})))
```

### Unconditional Branch

```lisp
;; Branch to named block
(cf.br {} ^target-block arg1 arg2)

;; Branch to block with no arguments
(cf.br {} ^target-block)
```

---

## 8. Regions

Operations like `scf.if`, `scf.for`, `func.func` contain regions.

### Pattern

```lisp
(operation {attributes} operands
  (region body...)
  (region body...))
```

### SCF If-Else Example

```lisp
(scf.if {} condition
  (region  ;; Then region
    (let [(: x (arith.muli {} val val))]
      x))  ;; Implicit scf.yield
  (region  ;; Else region
    val))  ;; Implicit scf.yield
```

### SCF For Loop Example

```lisp
(scf.for {:lower_bound 0 :upper_bound 10 :step 1}
  [(: iter i32)]
  (region
    (let [(: body-result (compute-something iter))]
      (scf.yield {} body-result))))
```

---

## 9. Complete Examples

### Example 1: Simple Function

**Terse Syntax:**
```lisp
(func.func {:sym_name add-one :function_type (-> (i32) (i32))}
  [(: arg0 i32)]
  (let [(: c1 (arith.constant {:value 1}))]
    (arith.addi {} arg0 c1)))
```

**Expands to MLIR:**
```mlir
func.func @add-one(%arg0: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %0 = arith.addi %arg0, %c1 : i32
  return %0 : i32
}
```

---

### Example 2: Absolute Value Function

**Terse Syntax:**
```lisp
(func.func {:sym_name abs :function_type (-> (i32) (i32))}
  [(: val i32)]
  (let [(: zero (arith.constant {:value 0}))
        (: is-neg (arith.cmpi {:predicate slt} val zero))]
    (cf.cond_br {} is-neg
      ^{:negate (let [(: negated (arith.subi {} zero val))]
                  (func.return {} negated))
        :positive (func.return {} val)})))
```

**Expands to MLIR:**
```mlir
func.func @abs(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %0 = arith.cmpi slt, %arg0, %c0 : i32
  cf.cond_br %0, ^bb1, ^bb2
^bb1:
  %1 = arith.subi %c0, %arg0 : i32
  return %1 : i32
^bb2:
  return %arg0 : i32
}
```

---

### Example 3: Sum Loop

**Terse Syntax:**
```lisp
(func.func {:sym_name sum-to-n :function_type (-> (i32) (i32))}
  [(: n i32)]
  (let [(: zero (arith.constant {:value 0}))
        (: one (arith.constant {:value 1}))]
    (block ^loop [(: i i32) (: sum i32)]
      (let [(: cond (arith.cmpi {:predicate slt} i n))]
        (cf.cond_br {} cond
          ^{:continue (let [(: new-sum (arith.addi {} sum i))
                            (: next-i (arith.addi {} i one))]
                        (cf.br {} ^loop next-i new-sum))
            :exit (func.return {} sum)})))))
```

**Expands to MLIR:**
```mlir
func.func @sum-to-n(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  cf.br ^bb1(%c0, %c0 : i32, i32)
^bb1(%0: i32, %1: i32):
  %2 = arith.cmpi slt, %0, %arg0 : i32
  cf.cond_br %2, ^bb2, ^bb3
^bb2:
  %3 = arith.addi %1, %0 : i32
  %4 = arith.addi %0, %c1 : i32
  cf.br ^bb1(%4, %3 : i32, i32)
^bb3:
  return %1 : i32
}
```

---

### Example 4: Nested Conditionals

**Terse Syntax:**
```lisp
(func.func {:sym_name nested-cond :function_type (-> (i32 i32) (i32))}
  [(: a i32) (: b i32)]
  (let [(: zero (arith.constant {:value 0}))
        (: a-pos (arith.cmpi {:predicate sgt} a zero))]
    (cf.cond_br {} a-pos
      ^{:check-b (let [(: b-pos (arith.cmpi {:predicate sgt} b zero))]
                   (cf.cond_br {} b-pos
                     ^{:both-pos (let [(: result (arith.addi {} a b))]
                                   (func.return {} result))
                       :b-neg (func.return {} a)}))
        :a-neg (func.return {} b)})))
```

---

### Example 5: SCF Structured Control Flow

**Terse Syntax:**
```lisp
(func.func {:sym_name scf-example :function_type (-> (i1 i32 i32) (i32))}
  [(: cond i1) (: a i32) (: b i32)]
  (scf.if {} cond
    (region
      (let [(: doubled (arith.muli {} a a))]
        doubled))  ;; Implicit scf.yield
    (region
      b)))  ;; Implicit scf.yield
```

---

### Example 6: Switch Statement

**Terse Syntax:**
```lisp
(func.func {:sym_name switch-example :function_type (-> (i32) (i32))}
  [(: val i32)]
  (llvm.switch {:case_values [0 1 2]} val
    ^{:default (arith.constant {:value -1})
      :case0 (arith.constant {:value 10})
      :case1 (arith.constant {:value 20})
      :case2 (arith.constant {:value 30})}))
```

---

### Example 7: Multiple Results

**Terse Syntax:**
```lisp
(func.func {:sym_name checked-add :function_type (-> (i32 i32) (i32 i1))}
  [(: a i32) (: b i32)]
  (let [[(: sum i32) (: overflow i1)]
        (arith.addui-extended {} a b)]
    (cf.cond_br {} overflow
      ^{:handle-overflow (do
                           (call-overflow-handler)
                           (func.return {} sum overflow))
        :normal (func.return {} sum overflow)})))
```

---

### Example 8: Nested Operations (Operation Flattening)

**Terse Syntax:**
```lisp
(func.func {:sym_name nested :function_type (-> (i32 i32) (i32))}
  [(: x i32) (: y i32)]
  (let [(: result (arith.addi {}
                    (arith.muli {} x y)
                    (arith.constant {:value 1})))]
    result))
```

**Note:** Nested operations in operand position are automatically flattened by the operation flattener.

---

## 10. Desugaring Rules

### Let → MLIR SSA

**Terse:**
```lisp
(let [(: x expr1)] body)
```

**Desugars to:**
```
%x = <expand expr1>
<expand body with x in scope>
```

---

### Implicit Returns → Explicit Terminators

**Terse:**
```lisp
(func.func {:sym_name foo :function_type (-> (i32) (i32))}
  [(: arg i32)]
  body)
```

**Desugars to:**
```mlir
func.func @foo(%arg: i32) -> i32 {
  <expand body>
  return <last-expr-result> : i32
}
```

---

### Inline Successors → Named Blocks

**Terse:**
```lisp
(cf.cond_br {} cond ^{:then expr1 :else expr2})
```

**Desugars to:**
```mlir
cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  <expand expr1>
^bb2:
  <expand expr2>
```

---

### Do → Sequential Operations

**Terse:**
```lisp
(do expr1 expr2 expr3)
```

**Desugars to:**
```
<expand expr1>
<expand expr2>
%result = <expand expr3>
%result  // Return this
```

---

## 11. Special Cases

### Operations Without Results

```lisp
;; Store has no result
(llvm.store {} val ptr)

;; Use in do for sequencing
(do
  (llvm.store {} val ptr)
  (arith.constant {:value 0}))  ;; This is returned
```

---

### Multiple Results

```lisp
;; Destructuring multiple results in let
(let [[(: sum i32) (: overflow i1)]
      (arith.addui-extended {} a b)]
  (use-both sum overflow))
```

---

### Nested Operations

```lisp
;; Operations can be nested in operand position
(arith.addi {}
  (arith.muli {} x y)
  (arith.constant {:value 1}))

;; Automatically flattened to SSA form
```

---

### Void/Side-Effect Operations

```lisp
;; Operations like store, print, etc. that don't produce values
(do
  (llvm.store {} value pointer)
  (llvm.call {:callee @printf} format-str value)
  final-result)
```

---

## 12. Attribute Syntax

### Attribute Values

```lisp
;; Integers
{:value 42}

;; Typed integers
{:value (: 42 i16)}

;; Floats
{:value 3.14}

;; Symbols (not strings!)
{:sym_name my-function}
{:callee @my-function}

;; Keywords (for enum-like values)
{:predicate slt}
{:predicate eq}

;; Booleans
{:some_flag true}
{:other_flag false}

;; Arrays
{:case_values [0 1 2]}
{:operandSegmentSizes [1 2 4]}

;; Nested attributes
{:llvm.linkage internal}

;; Typed attributes
{:value (: 1 i16)}

;; Function types
{:function_type (-> (i32 i64) (i32))}
```

---

## 13. Type Syntax

### Builtin Types

```lisp
i1 i8 i16 i32 i64 i128      ;; Integer types
f16 f32 f64                  ;; Float types
index                        ;; Index type
```

### Dialect Types

```lisp
!llvm.ptr                    ;; LLVM pointer
!llvm.struct<(i32, i64)>    ;; LLVM struct
!transform.any_op            ;; Transform dialect
!pdl.operation               ;; PDL dialect
!pdl.type                    ;; PDL type
```

### Function Types

```lisp
(-> (i32) (i32))            ;; i32 -> i32
(-> (i32 i64) (f32))        ;; (i32, i64) -> f32
(-> () (i32))               ;; () -> i32
(-> (i32) ())               ;; i32 -> void
```

### Tensor/Vector Types

```lisp
tensor<10xi32>              ;; 10-element i32 tensor
vector<4xf32>               ;; 4-element f32 vector
tensor<*xi32>               ;; Unranked i32 tensor
```

---

## 14. Open Questions / TBD

### Question 1: Empty Attributes

Should we allow omitting `{}` when no attributes?

```lisp
;; Option A: Always require {}
(arith.addi {} a b)

;; Option B: Optional when empty
(arith.addi a b)
```

**Decision:** TBD

---

### Question 2: Type Annotation Position in Let

Should we allow type before the binding?

```lisp
;; Current: type annotation wraps the value
(let [(: x i32) (some.op {})]
  x)

;; Alternative: type annotation in let syntax
(let [x : i32 (some.op {})]
  x)
```

**Decision:** Use `(: x type)` form for consistency

---

### Question 3: Block Initialization

How do we specify initial values for block arguments?

```lisp
;; Option A: Separate initialization
(let [(: zero (arith.constant {:value 0}))]
  (block ^loop [(: iter i32) (: sum i32)]
    (cf.br {} ^loop zero zero)))  ;; Initialize on first branch

;; Option B: Inline initialization
(block ^loop [(: iter i32 zero) (: sum i32 zero)]
  ...)
```

**Decision:** TBD

---

### Question 4: Macros

Which high-level constructs should be macros?

Candidates:
- `if` → expands to `cf.cond_br` or `scf.if`
- `when` → expands to `scf.if` with empty else
- `cond` → expands to nested `cf.cond_br`
- `for` → expands to `scf.for`
- `while` → expands to `scf.while`

**Decision:** TBD (implement after core syntax is working)

---

## 15. Grammar Summary

```
Program       ::= Form*

Form          ::= Operation
                | Let
                | Do
                | Block
                | Region
                | FuncDef
                | Literal
                | Identifier

Operation     ::= "(" OpName Attributes? Operands? Successors? ")"

OpName        ::= IDENTIFIER ("." IDENTIFIER)*

Attributes    ::= "{" (Keyword Value)* "}"

Operands      ::= Value*

Successors    ::= "^{" (Keyword Form)+ "}"

Let           ::= "(let" "[" Binding* "]" Form ")"

Binding       ::= "(:" IDENTIFIER Type? ")" Form
                | IDENTIFIER Form

Do            ::= "(do" Form+ ")"

Block         ::= "(block" BlockLabel "[" BlockArg* "]" Form ")"

BlockLabel    ::= "^" IDENTIFIER

BlockArg      ::= "(:" IDENTIFIER Type ")"

Region        ::= "(region" Form+ ")"

FuncDef       ::= "(func.func" Attributes "[" FuncArg* "]" Form ")"

FuncArg       ::= "(:" IDENTIFIER Type ")"

TypeAnnot     ::= "(:" Form Type ")"

Type          ::= IDENTIFIER
                | "!" IDENTIFIER ("<" TypeArgs ">")?
                | "(-> (" Type* ") (" Type* "))"

Value         ::= IDENTIFIER
                | NUMBER
                | STRING
                | SYMBOL
                | KEYWORD
                | Operation
                | TypeAnnot
                | Let
                | Do

Keyword       ::= ":" IDENTIFIER

Symbol        ::= "@" IDENTIFIER

Literal       ::= NUMBER | STRING | "true" | "false"
```

---

## 16. Implementation Notes

### Parser Changes Required

1. **Remove `%` prefix** from tokenizer for value IDs
2. **Parse function type syntax** `(-> (...) (...))`
3. **Parse inline successor maps** `^{:key expr ...}`
4. **Parse scoped let** with binding vector
5. **Parse do forms**
6. **Support symbol literals** without quotes
7. **Validate map insertion order** for successors

### Desugaring Pipeline

1. **Macro expansion** (when implemented)
2. **Let desugaring** - Convert to SSA bindings
3. **Inline successor expansion** - Generate named blocks
4. **Implicit return insertion** - Add terminators
5. **Operation flattening** - Flatten nested operations
6. **Type inference** - Infer types from MLIR operation definitions
7. **Builder** - Generate MLIR IR

---

## 17. Compatibility

### Backwards Compatibility

The verbose `(operation ...)` syntax should remain supported for:
- Generated code
- Complex cases where terse syntax is unclear
- Operations with non-standard structure
- Debugging and inspection

### Migration Path

1. **Phase 1:** Implement terse syntax parser alongside verbose parser
2. **Phase 2:** Add desugaring passes
3. **Phase 3:** Update examples to use terse syntax
4. **Phase 4:** Deprecate verbose syntax (optional)

---

## Appendix: Side-by-Side Comparison

### Verbose vs Terse

**Verbose:**
```lisp
(operation
  (name func.func)
  (attributes {
    :sym_name "add_one"
    :function_type (: (i32) -> i32)
  })
  (regions
    (region
      (block ^entry
        (arguments [%arg0 : i32])
        (operation
          (name arith.constant)
          (result-bindings [%c1])
          (result-types i32)
          (attributes {:value (: 1 i32)}))
        (operation
          (name arith.addi)
          (result-bindings [%0])
          (result-types i32)
          (operands %arg0 %c1))
        (operation
          (name func.return)
          (operands %0))))))
```

**Terse:**
```lisp
(func.func {:sym_name add-one :function_type (-> (i32) (i32))}
  [(: arg0 i32)]
  (let [(: c1 (arith.constant {:value 1}))]
    (arith.addi {} arg0 c1)))
```

**MLIR Output (same for both):**
```mlir
func.func @add-one(%arg0: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %0 = arith.addi %arg0, %c1 : i32
  return %0 : i32
}
```

---

**End of Specification**
