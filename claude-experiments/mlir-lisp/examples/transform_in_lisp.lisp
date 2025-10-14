;; ============================================================================
;; Transform Dialect Written Directly in Lisp
;; ============================================================================
;; No special Rust code needed - just write transform.* operations like
;; any other dialect!

;; ============================================================================
;; STEP 1: Define our source dialect (calc)
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
;; STEP 2: Write PDL patterns as Transform dialect operations
;; ============================================================================

;; PDL pattern to lower calc.constant -> arith.constant
(pdl.pattern
  :name "lower_calc_constant"
  :benefit 1
  :body
  (let [val (pdl.attribute)
        type (pdl.type)
        op (pdl.operation "calc.constant" :attrs {:value val} :results [type])
        result (pdl.result 0 op)]
    (pdl.rewrite op
      (let [new_op (pdl.operation "arith.constant" :attrs {:value val} :results [type])]
        (pdl.replace op new_op)))))

;; PDL pattern to lower calc.add -> arith.addi
(pdl.pattern
  :name "lower_calc_add"
  :benefit 1
  :body
  (let [lhs (pdl.operand)
        rhs (pdl.operand)
        type (pdl.type)
        op (pdl.operation "calc.add" :operands [lhs rhs] :results [type])
        result (pdl.result 0 op)]
    (pdl.rewrite op
      (let [new_op (pdl.operation "arith.addi" :operands [lhs rhs] :results [type])]
        (pdl.replace op new_op)))))

;; PDL pattern to lower calc.mul -> arith.muli
(pdl.pattern
  :name "lower_calc_mul"
  :benefit 1
  :body
  (let [lhs (pdl.operand)
        rhs (pdl.operand)
        type (pdl.type)
        op (pdl.operation "calc.mul" :operands [lhs rhs] :results [type])
        result (pdl.result 0 op)]
    (pdl.rewrite op
      (let [new_op (pdl.operation "arith.muli" :operands [lhs rhs] :results [type])]
        (pdl.replace op new_op)))))

;; ============================================================================
;; STEP 3: Write Transform sequence using Transform dialect operations
;; ============================================================================

(transform.sequence
  :attrs {:failure_propagation_mode "propagate"}
  :body
  (fn [module]
    ;; Match calc.constant operations
    (let [constants (transform.structured.match
                      :ops ["calc.constant"]
                      :in module)]
      ;; Apply lowering patterns
      (transform.apply_patterns :to constants
        :patterns [lower_calc_constant]))

    ;; Match calc.add operations
    (let [adds (transform.structured.match
                 :ops ["calc.add"]
                 :in module)]
      (transform.apply_patterns :to adds
        :patterns [lower_calc_add]))

    ;; Match calc.mul operations
    (let [muls (transform.structured.match
                 :ops ["calc.mul"]
                 :in module)]
      (transform.apply_patterns :to muls
        :patterns [lower_calc_mul]))))

;; ============================================================================
;; STEP 4: Write a program using calc dialect
;; ============================================================================

(defn compute [] i32
  (calc.add
    (calc.mul (calc.constant 10) (calc.constant 20))
    (calc.constant 30)))

;; ============================================================================
;; STEP 5: Apply the transform!
;; ============================================================================

;; This would:
;; 1. Emit the Transform dialect IR we wrote above
;; 2. Call MLIR's transform interpreter to execute it
;; 3. Get back lowered module with arith.* operations
;; 4. Continue lowering to LLVM
;; 5. JIT and execute

;; (apply-transform lower-calc-to-arith compute)
;; => 230
