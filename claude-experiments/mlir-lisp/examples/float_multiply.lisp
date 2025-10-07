;; Multiply two floating point numbers
;; Returns 15.0 (3.0 * 5.0)

(op arith.constant
    :attrs {:value 3.0}
    :results [f64]
    :as %three)

(op arith.constant
    :attrs {:value 5.0}
    :results [f64]
    :as %five)

(op arith.mulf
    :operands [%three %five]
    :results [f64]
    :as %result)

(op func.return
    :operands [%result])
