

```clojure
;============================================================
; Minimal S-expr MLIR grammar (explicit, round-trip-safe)
;============================================================

;----------------------------------------
; 0) Lexical tokens (opaque, dialect-agnostic)
;----------------------------------------
IDENT      ::= /[A-Za-z_][A-Za-z0-9_.$:-]*/
SUFFIX_ID  ::= /[0-9]+/ | IDENT      ; pure digits OR named identifier
NUMBER     ::= integer | float | hex | binary
STRING     ::= "…"
VALUE_ID   ::= "%" SUFFIX_ID         ; SSA value id (binding/ref), e.g. %0, %arg0
BLOCK_ID   ::= "^" SUFFIX_ID         ; block label (binding/ref), e.g. ^bb0, ^entry
SYMBOL     ::= "@" SUFFIX_ID         ; symbol-table name, e.g. @main, @42
OP_NAME    ::= IDENT "." IDENT ( "." IDENT )*   ; dialect/op name
KEYWORD    ::= ":" IDENT ( "." IDENT )*         ; attr keys
ATTR_MARKER ::= "#" <opaque-text-until-delimiter>
               ; Dialect attribute, e.g. #arith.overflow<none>, #llvm.noalias
               ; Uses bracket-aware scanning: tracks depth of <> and () brackets
               ; Handles string literals "..." (ignores brackets inside strings)
               ; Stops at whitespace/delimiters {, }, [, ], ; only when bracket depth is 0
               ; This allows complex nested attributes with spaces:
               ;   #dlti.dl_spec<i1 = dense<8> : vector<2xi64>, i64 = dense<64>>
               ;   #attr<"key" = "value with spaces">

;----------------------------------------
; 1) Core S-expressions
;----------------------------------------
ATOM       ::= IDENT | NUMBER | STRING | VALUE_ID | BLOCK_ID | SYMBOL | ATTR_MARKER
TYPE       ::= "!" <opaque-text-until-delimiter> | IDENT
               ; Builtin types (i32, f64, index, etc.) are written as plain identifiers
               ; Dialect types (!llvm.ptr, !transform.any_op, etc.) require the ! prefix
               ; Uses bracket-aware scanning for complex types with spaces:
               ;   !llvm.array<10 x i8>, !llvm.struct<(i32, i64)>, !llvm.ptr<272>
               ; Function types use special syntax: (!function (inputs TYPE*) (results TYPE*))
ATTR       ::= NUMBER | STRING | true | false | SYMBOL | TYPED_LITERAL | ATTR_MARKER
TYPED_LITERAL ::= "(:" VALUE TYPE ")"
               ; Represents a typed literal value like (: 42 i32) or (: 3 i64)
SEXPR      ::= ATOM | "(" SEXPR* ")" | "[" SEXPR* "]" | "{" (SEXPR SEXPR)* "}"

;----------------------------------------
; 2) Top-level structure
;----------------------------------------
MLIR       ::= (mlir TOP_LEVEL_ITEM*)
TOP_LEVEL_ITEM ::= TYPE_ALIAS | OPERATION

TYPE_ALIAS ::= (type-alias TYPE_ID STRING)
TYPE_ID    ::= "!" IDENT  ; Type alias name, e.g. !my_vec, !my_tensor
               ; STRING contains the opaque type definition, e.g. "vector<4xf32>"

;----------------------------------------
; 3) Operation form
;----------------------------------------
OPERATION  ::= (operation
                 (name OP_NAME)
                 SECTION*)

SECTION    ::= RESULTS
             | RESULT_TYPES
             | OPERANDS
             | ATTRIBUTES
             | SUCCESSORS
             | REGIONS
             | LOCATION

;----------------------------------------
; 4) Explicit, self-descriptive section names
;----------------------------------------
RESULTS        ::= (result-bindings [ VALUE_ID* ])
RESULT_TYPES   ::= (result-types TYPE*)
OPERANDS       ::= (operand-uses VALUE_ID*)
ATTRIBUTES     ::= (attributes { KEYWORD ATTR* })
SUCCESSORS     ::= (successors SUCCESSOR*)
LOCATION       ::= (location ATTR)

;----------------------------------------
; 5) Regions and blocks
;----------------------------------------
REGIONS        ::= (regions REGION*)
REGION         ::= (region BLOCK+)

BLOCK          ::= (block
                     [ BLOCK_ID ]                     ; block label binding
                     (arguments [ [ VALUE_ID TYPE ]* ]) ; arg bindings
                     OPERATION*)                       ; block body

SUCCESSOR      ::= (successor BLOCK_ID (operand-bundle)?)
operand-bundle ::= ( VALUE_ID* )

;----------------------------------------
; 6) Well-formedness rules (non-grammatical)
;----------------------------------------
; • (operation) must have exactly one (name …).
; • Section order is flexible, all optional.
; • [ … ] always means "introduces new bindings" (values or labels).
; • { … } always means "key/value pairs" (maps).
; • Plain ( … ) is sequencing or grouping.
; • Dialect types use ! prefix (!llvm.ptr, !function, etc.).
; • Type aliases must be defined before use (at module level).
; • Type alias definitions are opaque strings (not parsed).
; • Typed literals use (: value type) syntax for integer attributes.
; • Symbol references (@func_name) are used for function names and callees.
; • Keywords (starting with :) are used for attribute keys and some enum values.
; • Round-trip guarantees: order preserved, no implied defaults.

;============================================================
; 7) Examples
;============================================================

;----------------------------------------
; Example 1 — type aliases
;----------------------------------------
(mlir
  (type-alias !my_vec "vector<4xf32>")
  (type-alias !my_tensor "tensor<10x20xf32>")

  (operation
    (name func.func)
    (attributes {
      :sym_name @test
      :function_type (!function (inputs !my_vec !my_tensor) (results !my_vec))
    })
    (regions
      (region
        (block [^entry]
          (arguments [ [%arg0 !my_vec] [%arg1 !my_tensor] ])
          (operation
            (name func.return)
            (operands %arg0)))))))

;----------------------------------------
; Example 2 — simple constant
;----------------------------------------
(mlir
  (operation
    (name arith.constant)
    (result-bindings [%c0])
    (result-types i32)
    (attributes { :value (: 42 i32) })))

;----------------------------------------
; Example 2 — add inside a function
;----------------------------------------
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
          (arguments [ [%x i32] [%y i32] ])
          (operation
            (name arith.addi)
            (result-bindings [%sum])
            (operands %x %y)
            (result-types i32))
          (operation
            (name func.return)
            (operands %sum)))))))

;----------------------------------------
; Example 3 — function call with two constants
;----------------------------------------
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @main
      :function_type (!function (inputs) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [])
          (operation
            (name arith.constant)
            (result-bindings [%a])
            (result-types i32)
            (attributes { :value (: 1 i32) }))
          (operation
            (name arith.constant)
            (result-bindings [%b])
            (result-types i32)
            (attributes { :value (: 2 i32) }))
          (operation
            (name func.call)
            (result-bindings [%r])
            (result-types i32)
            (operands %a %b)
            (attributes { :callee @add }))
          (operation
            (name func.return)
            (operands %r)))))))

;----------------------------------------
; Example 4 — control flow with multiple blocks
;----------------------------------------
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @branchy
      :function_type (!function (inputs i1 i32 i32) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [ [%cond i1] [%x i32] [%y i32] ])
          (operation
            (name cf.cond_br)
            (operands %cond)
            (successors
              (successor ^then (%x))
              (successor ^else (%y)))))
        (block [^then]
          (arguments [ [%t i32] ])
          (operation (name func.return) (operands %t)))
        (block [^else]
          (arguments [ [%e i32] ])
          (operation (name func.return) (operands %e)))))))

;----------------------------------------
; Example 5 — recursive fibonacci with scf.if
;----------------------------------------
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
          (arguments [ [%n i32] ])

          ;; Check if n <= 1 (base case)
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

          ;; scf.if with nested regions (then/else)
          (operation
            (name scf.if)
            (result-bindings [%result])
            (result-types i32)
            (operands %cond)
            (regions
              ;; Then region: base case, return n
              (region
                (block
                  (arguments [])
                  (operation
                    (name scf.yield)
                    (operands %n))))

              ;; Else region: recursive case, return fib(n-1) + fib(n-2)
              (region
                (block
                  (arguments [])

                  ;; Compute fib(n-1)
                  (operation
                    (name arith.constant)
                    (result-bindings [%c1_rec])
                    (result-types i32)
                    (attributes { :value (: 1 i32) }))

                  (operation
                    (name arith.subi)
                    (result-bindings [%n_minus_1])
                    (result-types i32)
                    (operands %n %c1_rec))

                  (operation
                    (name func.call)
                    (result-bindings [%fib_n_minus_1])
                    (result-types i32)
                    (operands %n_minus_1)
                    (attributes { :callee @fibonacci }))

                  ;; Compute fib(n-2)
                  (operation
                    (name arith.constant)
                    (result-bindings [%c2])
                    (result-types i32)
                    (attributes { :value (: 2 i32) }))

                  (operation
                    (name arith.subi)
                    (result-bindings [%n_minus_2])
                    (result-types i32)
                    (operands %n %c2))

                  (operation
                    (name func.call)
                    (result-bindings [%fib_n_minus_2])
                    (result-types i32)
                    (operands %n_minus_2)
                    (attributes { :callee @fibonacci }))

                  ;; Add fib(n-1) + fib(n-2) and yield
                  (operation
                    (name arith.addi)
                    (result-bindings [%sum])
                    (result-types i32)
                    (operands %fib_n_minus_1 %fib_n_minus_2))

                  (operation
                    (name scf.yield)
                    (operands %sum))))))

          ;; Return the result
          (operation
            (name func.return)
            (operands %result)))))))

;============================================================
; 8) Common Attribute Patterns
;============================================================

;----------------------------------------
; Integer Constants (typed literals)
;----------------------------------------
; Use (: value type) syntax:
; (attributes { :value (: 42 i32) })
; (attributes { :value (: 10 i64) })
; (attributes { :predicate (: 3 i64) })  ; Comparison predicate as integer

;----------------------------------------
; String Attributes
;----------------------------------------
; Plain strings for string values:
; (attributes { :predicate "eq" })
; (attributes { :predicate "sle" })

;----------------------------------------
; Symbol References
;----------------------------------------
; Use @ prefix for function names and symbol references:
; (attributes { :sym_name @main })
; (attributes { :callee @fibonacci })

;----------------------------------------
; Function Type Attributes
;----------------------------------------
; Use !function with inputs/results sections:
; (attributes { :function_type (!function (inputs i32 i32) (results i32)) })
; (attributes { :function_type (!function (inputs) (results i64)) })

;----------------------------------------
; Dialect Types
;----------------------------------------
; Simple dialect types use ! prefix:
; !llvm.ptr           ; LLVM pointer type
; !transform.any_op   ; Transform dialect type
; !function           ; Function type (special case)

; Complex dialect types with spaces and nested brackets:
; Uses bracket-aware parsing to handle spaces inside <> and () brackets
; !llvm.array<10 x i8>                  ; Array type with dimensions
; !llvm.array<20 x f32>                 ; Float array
; !llvm.struct<(i32, i64)>              ; Struct with multiple fields
; !llvm.struct<(i32, array<5 x f32>)>   ; Nested types
; !llvm.ptr<270>                        ; Pointer with address space
; !llvm.ptr<271>                        ; Different address space
; !llvm.func<ptr (ptr, ptr)>            ; Function type

; Multi-dimensional and complex nested types:
; !llvm.array<10 x array<20 x i32>>     ; 2D array
; !llvm.struct<(ptr, array<10 x i8>, i64)>  ; Complex struct

;----------------------------------------
; Builtin Types
;----------------------------------------
; Builtin types are plain identifiers:
; i32, i64, i1       ; Integer types
; f32, f64           ; Float types
; index              ; Index type

;----------------------------------------
; Dialect Attribute Markers
;----------------------------------------
; Simple dialect attributes (no spaces):
; (attributes { :linkage #llvm.linkage<internal> })
; (attributes { :frame_pointer #llvm.framePointerKind<none> })
; (attributes { :overflow #arith.overflow<none> })

; Complex nested attributes (with spaces):
; Uses bracket-aware parsing to handle spaces inside <> brackets
; (attributes {
;   :dlti.dl_spec #dlti.dl_spec<
;     i1 = dense<8> : vector<2xi64>,
;     i64 = dense<64> : vector<2xi64>,
;     "dlti.endianness" = "little"
;   >
; })

; Attributes with string literals containing special characters:
; (attributes { :meta #attr<"key" = "value with spaces"> })
; (attributes { :path #attr<"file.path" = "/usr/local/bin"> })

; Deeply nested attributes:
; (attributes { :nested #attr<outer<inner<deep<value>>>> })

; Real-world example - LLVM Data Layout Specification:
; #dlti.dl_spec<
;   i1 = dense<8> : vector<2xi64>,
;   !llvm.ptr = dense<64> : vector<4xi64>,
;   i128 = dense<128> : vector<2xi64>,
;   i64 = dense<64> : vector<2xi64>,
;   !llvm.ptr<272> = dense<64> : vector<4xi64>,
;   f64 = dense<64> : vector<2xi64>,
;   i32 = dense<32> : vector<2xi64>,
;   "dlti.stack_alignment" = 128 : i64,
;   "dlti.endianness" = "little"
; >
```