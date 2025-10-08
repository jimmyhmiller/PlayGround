;; ============================================================================
;; LOWERING TRANSFORMATIONS
;; ============================================================================
;; Transform from lisp dialect to arith dialect

;; Main lowering transformation
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
