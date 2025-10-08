;; ============================================================================
;; LISP CORE DIALECT
;; ============================================================================
;; Core operations for the Lisp dialect

(defirdl-dialect lisp
  :namespace "lisp"
  :description "High-level Lisp semantic operations"

  ;; Constants
  (defirdl-op constant
    :summary "Immutable constant value"
    :description "A pure, immutable constant in the Lisp dialect"
    :attributes [(value IntegerAttr "The constant integer value")]
    :results [(result AnyInteger "The constant value")]
    :traits [Pure NoMemoryEffect])

  ;; Arithmetic operations
  (defirdl-op add
    :summary "Pure functional addition"
    :description "Adds two values with Lisp semantics (pure, no overflow)"
    :operands [(lhs AnyInteger "Left operand")
               (rhs AnyInteger "Right operand")]
    :results [(result AnyInteger "Sum of operands")]
    :traits [Pure Commutative NoMemoryEffect]
    :constraints [(same-type lhs rhs result)])

  (defirdl-op sub
    :summary "Pure functional subtraction"
    :operands [(lhs AnyInteger) (rhs AnyInteger)]
    :results [(result AnyInteger)]
    :traits [Pure NoMemoryEffect]
    :constraints [(same-type lhs rhs result)])

  (defirdl-op mul
    :summary "Pure functional multiplication"
    :operands [(lhs AnyInteger) (rhs AnyInteger)]
    :results [(result AnyInteger)]
    :traits [Pure Commutative NoMemoryEffect]
    :constraints [(same-type lhs rhs result)])

  ;; Control flow
  (defirdl-op if
    :summary "Conditional expression"
    :operands [(condition I1 "Boolean condition")
               (true-value AnyType "Value if true")
               (false-value AnyType "Value if false")]
    :results [(result AnyType "Selected value")]
    :traits [Pure NoMemoryEffect]
    :constraints [(same-type true-value false-value result)])

  ;; Function calls
  (defirdl-op call
    :summary "Tail-call optimizable function call"
    :description "Function call with Lisp calling conventions"
    :attributes [(callee FlatSymbolRefAttr "Function to call")]
    :operands [(args Variadic<AnyType> "Function arguments")]
    :results [(result AnyType "Return value")]
    :traits [IsolatedFromAbove]))
