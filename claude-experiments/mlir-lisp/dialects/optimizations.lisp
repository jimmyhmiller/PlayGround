;; ============================================================================
;; OPTIMIZATION PATTERNS
;; ============================================================================
;; Common optimization patterns for the Lisp dialect

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

;; Optimization pipeline
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
