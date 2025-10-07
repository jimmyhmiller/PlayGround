;; Test nested expressions with a do-style macro

;; Basic operators
(defmacro const [value]
  (op arith.constant :attrs {:value value} :results [i32]))

(defmacro + [a b]
  (op arith.addi :operands [a b] :results [i32]))

;; Test: (+ (const 5) (const 3))
;; We need to emit the nested consts first, then use their results
(defn main [] i32
  ;; Manual nesting: emit subexpressions first
  (const 5 :as %five)
  (const 3 :as %three)
  (+ %five %three :as %result)
  (op func.return :operands [%result]))
