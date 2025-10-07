;; Add two numbers together
;; Returns 42 (10 + 32)

(op arith.constant
    :attrs {:value 10}
    :results [i32]
    :as %ten)

(op arith.constant
    :attrs {:value 32}
    :results [i32]
    :as %thirty_two)

(op arith.addi
    :operands [%ten %thirty_two]
    :results [i32]
    :as %result)

(op func.return
    :operands [%result])
