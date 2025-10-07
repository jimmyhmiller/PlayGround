;; Divide two numbers (signed)
;; Returns 7 (84 / 12)

(op arith.constant
    :attrs {:value 84}
    :results [i32]
    :as %dividend)

(op arith.constant
    :attrs {:value 12}
    :results [i32]
    :as %divisor)

(op arith.divsi
    :operands [%dividend %divisor]
    :results [i32]
    :as %result)

(op func.return
    :operands [%result])
