

```clojure
;============================================================
; Minimal S-expr MLIR grammar (explicit, round-trip-safe)
;============================================================

;----------------------------------------
; 0) Lexical tokens (opaque, dialect-agnostic)
;----------------------------------------
IDENT      ::= /[A-Za-z_][A-Za-z0-9_.$:-]*/
NUMBER     ::= integer | float | hex | binary
STRING     ::= "…"
VALUE_ID   ::= "%" IDENT             ; SSA value id (binding/ref)
BLOCK_ID   ::= "^" IDENT             ; block label (binding/ref)
SYMBOL     ::= "@" IDENT             ; symbol-table name
OP_NAME    ::= IDENT "." IDENT ( "." IDENT )*   ; dialect/op name
KEYWORD    ::= ":" IDENT ( "." IDENT )*         ; attr keys

;----------------------------------------
; 1) Core S-expressions
;----------------------------------------
ATOM       ::= IDENT | NUMBER | STRING | VALUE_ID | BLOCK_ID | SYMBOL
TYPE       ::= "!" SEXPR
ATTR       ::= NUMBER | STRING | true | false | "#" SEXPR
SEXPR      ::= ATOM | "(" SEXPR* ")" | "[" SEXPR* "]" | "{" (SEXPR SEXPR)* "}"

;----------------------------------------
; 2) Top-level structure
;----------------------------------------
MLIR       ::= (mlir OPERATION*)

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
; • [ … ] always means “introduces new bindings” (values or labels).
; • { … } always means “key/value pairs” (maps).
; • Plain ( … ) is sequencing or grouping.
; • Dialect types and attributes are opaque (# … and ! … payloads).
; • Round-trip guarantees: order preserved, no implied defaults.

;============================================================
; 7) Examples
;============================================================

;----------------------------------------
; Example 1 — simple constant
;----------------------------------------
(mlir
  (operation
    (name arith.constant)
    (result-bindings [%c0])
    (result-types !i32)
    (attributes { :value (#int 42) })
    (location (#unknown))))

;----------------------------------------
; Example 2 — add inside a function
;----------------------------------------
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym  (#sym @add)
      :type (!function (inputs !i32 !i32) (results !i32))
      :visibility :public
    })
    (regions
      (region
        (block [^entry]
          (arguments [ [%x !i32] [%y !i32] ])
          (operation
            (name arith.addi)
            (result-bindings [%sum])
            (operands %x %y)
            (result-types !i32))
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
      :sym (#sym @main)
      :type (!function (inputs) (results !i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [])
          (operation
            (name arith.constant)
            (result-bindings [%a])
            (result-types !i32)
            (attributes { :value (#int 1) }))
          (operation
            (name arith.constant)
            (result-bindings [%b])
            (result-types !i32)
            (attributes { :value (#int 2) }))
          (operation
            (name func.call)
            (result-bindings [%r])
            (result-types !i32)
            (operands %a %b)
            (attributes { :callee (#flat-symbol @add) }))
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
      :sym (#sym @branchy)
      :type (!function (inputs !i1 !i32 !i32) (results !i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [ [%cond !i1] [%x !i32] [%y !i32] ])
          (operation
            (name cf.cond_br)
            (operands %cond)
            (successors
              (successor ^then (%x))
              (successor ^else (%y)))))
        (block [^then]
          (arguments [ [%t !i32] ])
          (operation (name func.return) (operands %t)))
        (block [^else]
          (arguments [ [%e !i32] ])
          (operation (name func.return) (operands %e)))))))

;----------------------------------------
; Example 5 — recursive fibonacci with scf.if
;----------------------------------------
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym (#sym @fibonacci)
      :type (!function (inputs !i32) (results !i32))
      :visibility :public
    })
    (regions
      (region
        (block [^entry]
          (arguments [ [%n !i32] ])

          ;; Check if n <= 1 (base case)
          (operation
            (name arith.constant)
            (result-bindings [%c1])
            (result-types !i32)
            (attributes { :value (#int 1) }))

          (operation
            (name arith.cmpi)
            (result-bindings [%cond])
            (result-types !i1)
            (operands %n %c1)
            (attributes { :predicate (#string "sle") }))

          ;; scf.if with nested regions (then/else)
          (operation
            (name scf.if)
            (result-bindings [%result])
            (result-types !i32)
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
                    (result-types !i32)
                    (attributes { :value (#int 1) }))

                  (operation
                    (name arith.subi)
                    (result-bindings [%n_minus_1])
                    (result-types !i32)
                    (operands %n %c1_rec))

                  (operation
                    (name func.call)
                    (result-bindings [%fib_n_minus_1])
                    (result-types !i32)
                    (operands %n_minus_1)
                    (attributes { :callee (#flat-symbol @fibonacci) }))

                  ;; Compute fib(n-2)
                  (operation
                    (name arith.constant)
                    (result-bindings [%c2])
                    (result-types !i32)
                    (attributes { :value (#int 2) }))

                  (operation
                    (name arith.subi)
                    (result-bindings [%n_minus_2])
                    (result-types !i32)
                    (operands %n %c2))

                  (operation
                    (name func.call)
                    (result-bindings [%fib_n_minus_2])
                    (result-types !i32)
                    (operands %n_minus_2)
                    (attributes { :callee (#flat-symbol @fibonacci) }))

                  ;; Add fib(n-1) + fib(n-2) and yield
                  (operation
                    (name arith.addi)
                    (result-bindings [%sum])
                    (result-types !i32)
                    (operands %fib_n_minus_1 %fib_n_minus_2))

                  (operation
                    (name scf.yield)
                    (operands %sum))))))

          ;; Return the result
          (operation
            (name func.return)
            (operands %result)))))))
```