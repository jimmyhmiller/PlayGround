;; ============================================================================
;; Simple Demo: Dialect Registration
;; ============================================================================
;; Run: mlir-lisp examples/simple_demo.lisp

;; ============================================================================
;; STEP 1: Define a Dialect
;; ============================================================================

(defirdl-dialect calc
  :namespace "calc"
  :description "Simple calculator dialect"

  (defirdl-op constant
    :summary "Integer constant"
    :attributes [(value IntegerAttr)]
    :results [(result I32)])

  (defirdl-op add
    :summary "Addition"
    :operands [(lhs I32) (rhs I32)]
    :results [(result I32)])

  (defirdl-op mul
    :summary "Multiplication"
    :operands [(lhs I32) (rhs I32)]
    :results [(result I32)]))

;; ============================================================================
;; STEP 2: Define Transform Patterns
;; ============================================================================

(defpdl-pattern lower-calc-constant
  :benefit 1
  :description "Lower calc.constant to arith.constant"
  :match
  (let [val (pdl.attribute)]
    val)
  :rewrite
  (pdl.operation "arith.constant"))

(defpdl-pattern lower-calc-add
  :benefit 1
  :description "Lower calc.add to arith.addi"
  :match
  (pdl.operation "calc.add")
  :rewrite
  (pdl.operation "arith.addi"))

;; ============================================================================
;; STEP 3: Show What We Registered
;; ============================================================================

(println "✅ Registered dialects:" (list-dialects))
(println "✅ Registered patterns:" (list-patterns))

;; ============================================================================
;; Success!
;; ============================================================================
;; The dialect and patterns are now registered and can be used
;; to generate and transform MLIR IR.
;;
;; Next: Compile programs using calc.* operations and apply transforms!
