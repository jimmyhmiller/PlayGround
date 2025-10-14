;; ============================================================================
;; COMPLETE END-TO-END EXAMPLE
;; ============================================================================
;; Run: mlir-lisp examples/complete.lisp
;;
;; This demonstrates:
;; 1. Defining a dialect
;; 2. Defining transform patterns
;; 3. Writing a program using the dialect
;; 4. Compiling and showing the IR

;; ============================================================================
;; STEP 1: Define the calc dialect
;; ============================================================================

(defirdl-dialect calc
  :namespace "calc"
  :description "Calculator dialect"

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
;; STEP 2: Define transform patterns for lowering
;; ============================================================================

(defpdl-pattern lower-calc-constant
  :benefit 1
  :description "Lower calc.constant to arith.constant"
  :match (pdl.operation "calc.constant")
  :rewrite (pdl.operation "arith.constant"))

(defpdl-pattern lower-calc-add
  :benefit 1
  :description "Lower calc.add to arith.addi"
  :match (pdl.operation "calc.add")
  :rewrite (pdl.operation "arith.addi"))

(defpdl-pattern lower-calc-mul
  :benefit 1
  :description "Lower calc.mul to arith.muli"
  :match (pdl.operation "calc.mul")
  :rewrite (pdl.operation "arith.muli"))

;; ============================================================================
;; STEP 3: Write a program using our dialect
;; ============================================================================

;; This program computes: (10 * 20) + 30 = 230
;; It uses calc.* operations which will be in the generated IR!

(defn compute [] i32
  (calc.add
    (calc.mul
      (calc.constant 10)
      (calc.constant 20))
    (calc.constant 30)))

;; ============================================================================
;; STEP 4: Show what we have
;; ============================================================================

(println "\n✅ Compilation complete!")
(println "Dialects registered:" (list-dialects))
(println "Patterns registered:" (list-patterns))

;; ============================================================================
;; STEP 5: Execute (when ready)
;; ============================================================================

;; Once JIT is implemented, this will work:
;; (jit-execute "compute" "compute")
;; => 230

(println "\nTo see the generated IR with calc.* operations,")
(println "check the module compilation output above!")
