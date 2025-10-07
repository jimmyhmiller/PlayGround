;; Multiply two numbers
;; Returns 150 (10 * 15)

(op arith.constant
    :attrs {:value 10}
    :results [i32]
    :as %ten)

(op arith.constant
    :attrs {:value 15}
    :results [i32]
    :as %fifteen)

(op arith.muli
    :operands [%ten %fifteen]
    :results [i32]
    :as %result)

(op func.return
    :operands [%result])
