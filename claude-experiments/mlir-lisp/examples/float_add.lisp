;; Add two floating point numbers
;; Returns 5.7 (3.14 + 2.56)

(op arith.constant
    :attrs {:value 3.14}
    :results [f64]
    :as %pi_ish)

(op arith.constant
    :attrs {:value 2.56}
    :results [f64]
    :as %other)

(op arith.addf
    :operands [%pi_ish %other]
    :results [f64]
    :as %result)

(op func.return
    :operands [%result])
