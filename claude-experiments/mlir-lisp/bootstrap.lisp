;; ============================================================================
;; BOOTSTRAP.LISP - Self-Contained Meta-Circular Compiler
;; ============================================================================
;; This file bootstraps the entire MLIR-Lisp compiler!
;; Everything is defined in Lisp - no Rust code needed by users.

;; ============================================================================
;; PHASE 1: Define the Core Lisp Dialect
;; ============================================================================

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

;; ============================================================================
;; PHASE 2: Define Optimization Patterns
;; ============================================================================

;; Constant folding for addition
(defpdl-pattern constant-fold-add
  :benefit 10
  :description "Fold addition of two constants at compile time"
  :match
  (let [val1 (pdl.attribute)
        val2 (pdl.attribute)
        type (pdl.type)
        const1 (pdl.operation "lisp.constant" :attributes {:value val1} :results [type])
        result1 (pdl.result 0 :of const1)
        const2 (pdl.operation "lisp.constant" :attributes {:value val2} :results [type])
        result2 (pdl.result 0 :of const2)
        add-op (pdl.operation "lisp.add" :operands [result1 result2] :results [type])]
    add-op)
  :rewrite
  (let [sum (pdl.apply-native "add-integers" [val1 val2])
        new-const (pdl.operation "lisp.constant" :attributes {:value sum} :results [type])]
    (pdl.replace add-op :with new-const)))

;; Constant folding for multiplication
(defpdl-pattern constant-fold-mul
  :benefit 10
  :description "Fold multiplication of two constants at compile time"
  :match
  (let [val1 (pdl.attribute)
        val2 (pdl.attribute)
        type (pdl.type)
        const1 (pdl.operation "lisp.constant" :attributes {:value val1} :results [type])
        result1 (pdl.result 0 :of const1)
        const2 (pdl.operation "lisp.constant" :attributes {:value val2} :results [type])
        result2 (pdl.result 0 :of const2)
        mul-op (pdl.operation "lisp.mul" :operands [result1 result2] :results [type])]
    mul-op)
  :rewrite
  (let [product (pdl.apply-native "mul-integers" [val1 val2])
        new-const (pdl.operation "lisp.constant" :attributes {:value product} :results [type])]
    (pdl.replace mul-op :with new-const)))

;; Dead code elimination
(defpdl-pattern eliminate-dead-code
  :benefit 5
  :description "Remove operations with no uses"
  :match
  (let [op (pdl.operation :any)
        result (pdl.result 0 :of op)]
    op)
  :constraint (pdl.no-uses? result)
  :constraint (pdl.has-trait? op "Pure")
  :rewrite
  (pdl.erase op))

;; ============================================================================
;; PHASE 3: Define Lowering Transformations
;; ============================================================================

;; Lower lisp dialect to arith dialect
(deftransform lower-to-arith
  :description "Lower lisp dialect operations to arith dialect"

  (transform.sequence failures(propagate)
    :args [module]

    ;; Stage 1: Lower lisp.constant -> arith.constant
    (let [constants (transform.match :ops ["lisp.constant"] :in module)]
      (transform.apply-patterns :to constants
        (use-pattern constant-lowering)))

    ;; Stage 2: Lower lisp.add -> arith.addi
    (let [adds (transform.match :ops ["lisp.add"] :in module)]
      (transform.apply-patterns :to adds
        (use-pattern add-lowering)))

    ;; Stage 3: Lower lisp.sub -> arith.subi
    (let [subs (transform.match :ops ["lisp.sub"] :in module)]
      (transform.apply-patterns :to subs
        (use-pattern sub-lowering)))

    ;; Stage 4: Lower lisp.mul -> arith.muli
    (let [muls (transform.match :ops ["lisp.mul"] :in module)]
      (transform.apply-patterns :to muls
        (use-pattern mul-lowering)))))

;; PDL patterns for lowering
(defpdl-pattern constant-lowering
  :benefit 1
  :description "Lower lisp.constant to arith.constant"
  :match
  (let [val (pdl.attribute)
        type (pdl.type)
        op (pdl.operation "lisp.constant" :attributes {:value val} :results [type])]
    op)
  :rewrite
  (pdl.operation "arith.constant" :attributes {:value val} :results [type]))

(defpdl-pattern add-lowering
  :benefit 1
  :description "Lower lisp.add to arith.addi"
  :match
  (let [lhs (pdl.operand)
        rhs (pdl.operand)
        type (pdl.type)
        op (pdl.operation "lisp.add" :operands [lhs rhs] :results [type])]
    op)
  :rewrite
  (pdl.operation "arith.addi" :operands [lhs rhs] :results [type]))

(defpdl-pattern sub-lowering
  :benefit 1
  :match
  (let [lhs (pdl.operand)
        rhs (pdl.operand)
        type (pdl.type)
        op (pdl.operation "lisp.sub" :operands [lhs rhs] :results [type])]
    op)
  :rewrite
  (pdl.operation "arith.subi" :operands [lhs rhs] :results [type]))

(defpdl-pattern mul-lowering
  :benefit 1
  :match
  (let [lhs (pdl.operand)
        rhs (pdl.operand)
        type (pdl.type)
        op (pdl.operation "lisp.mul" :operands [lhs rhs] :results [type])]
    op)
  :rewrite
  (pdl.operation "arith.muli" :operands [lhs rhs] :results [type]))

;; ============================================================================
;; PHASE 4: Define Compilation Pipeline
;; ============================================================================

;; Full optimization pipeline
(deftransform optimize
  :description "Apply all optimizations"

  (transform.sequence failures(propagate)
    :args [module]

    ;; Apply constant folding
    (transform.apply-patterns :to module
      (use-pattern constant-fold-add)
      (use-pattern constant-fold-mul))

    ;; Remove dead code
    (transform.apply-patterns :to module
      (use-pattern eliminate-dead-code))))

;; ============================================================================
;; Bootstrap Complete!
;; ============================================================================
;; The compiler is now fully defined in Lisp.
;; You can now write programs using the 'lisp' dialect and compile them!
