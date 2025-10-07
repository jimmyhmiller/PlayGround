;; Subtract two numbers
;; Returns 8 (50 - 42)

(op arith.constant
    :attrs {:value 50}
    :results [i32]
    :as %fifty)

(op arith.constant
    :attrs {:value 42}
    :results [i32]
    :as %fortytwo)

(op arith.subi
    :operands [%fifty %fortytwo]
    :results [i32]
    :as %result)

(op func.return
    :operands [%result])
