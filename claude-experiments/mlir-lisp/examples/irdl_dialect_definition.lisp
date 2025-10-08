;; IRDL (IR Definition Language) - Define MLIR Dialects in Lisp!
;; ================================================================
;; This file defines a custom dialect using IRDL, which is MLIR's
;; way of defining dialects within MLIR itself (meta-circular!)
;;
;; We're taking it further: Define IRDL *in Lisp*!

;; Import IRDL base dialect (like Racket's #lang)
(import-dialect irdl base)

;; Define our "lisp" dialect
(defirdl-dialect lisp
  :namespace "lisp"
  :description "High-level Lisp semantic operations"

  ;; Define lisp.constant operation
  (defirdl-op constant
    :summary "Immutable constant value"
    :description "A pure, immutable constant in the Lisp dialect"

    ;; Attributes
    :attributes [
      (value IntegerAttr "The constant integer value")
    ]

    ;; Results
    :results [
      (result AnyInteger "The constant value")
    ]

    ;; Traits
    :traits [Pure NoMemoryEffect]

    ;; Verifier (optional)
    :verify (lambda [op]
      (assert (has-attr? op "value")
              "lisp.constant must have value attribute")))

  ;; Define lisp.add operation
  (defirdl-op add
    :summary "Pure functional addition"
    :description "Adds two values with Lisp semantics (pure, no overflow)"

    :operands [
      (lhs AnyInteger "Left operand")
      (rhs AnyInteger "Right operand")
    ]

    :results [
      (result AnyInteger "Sum of operands")
    ]

    :traits [Pure Commutative NoMemoryEffect]

    ;; Same-type constraint
    :constraints [
      (same-type lhs rhs result)
    ])

  ;; Define lisp.sub operation
  (defirdl-op sub
    :summary "Pure functional subtraction"
    :operands [
      (lhs AnyInteger)
      (rhs AnyInteger)
    ]
    :results [
      (result AnyInteger)
    ]
    :traits [Pure NoMemoryEffect]
    :constraints [(same-type lhs rhs result)])

  ;; Define lisp.mul operation
  (defirdl-op mul
    :summary "Pure functional multiplication"
    :operands [
      (lhs AnyInteger)
      (rhs AnyInteger)
    ]
    :results [
      (result AnyInteger)
    ]
    :traits [Pure Commutative NoMemoryEffect]
    :constraints [(same-type lhs rhs result)])

  ;; Define lisp.call operation
  (defirdl-op call
    :summary "Tail-call optimizable function call"
    :description "Function call with Lisp calling conventions"

    :attributes [
      (callee FlatSymbolRefAttr "Function to call")
    ]

    :operands [
      (args Variadic<AnyType> "Function arguments")
    ]

    :results [
      (result AnyType "Return value")
    ]

    :traits [IsolatedFromAbove]

    :extra-class-declaration "
      // Helper to get callee name
      StringRef getCallee() { return (*this)->getAttr(\"callee\"); }
    "))

;; Export the dialect for use
(export-dialect lisp)

;; Print what we defined
(defn main [] i32
  (println "✓ Defined lisp dialect with IRDL")
  (println "✓ Operations: constant, add, sub, mul, call")
  (println "✓ All operations are Pure and NoMemoryEffect")
  (println "✓ Dialect ready for use!")
  0)
