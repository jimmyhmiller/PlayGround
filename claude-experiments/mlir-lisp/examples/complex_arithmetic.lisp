;; Complex arithmetic expression
;; Computes: (10 + 5) * 2 - 3 = 27

(op arith.constant
    :attrs {:value 10}
    :results [i32]
    :as %ten)

(op arith.constant
    :attrs {:value 5}
    :results [i32]
    :as %five)

(op arith.constant
    :attrs {:value 2}
    :results [i32]
    :as %two)

(op arith.constant
    :attrs {:value 3}
    :results [i32]
    :as %three)

;; 10 + 5 = 15
(op arith.addi
    :operands [%ten %five]
    :results [i32]
    :as %sum)

;; 15 * 2 = 30
(op arith.muli
    :operands [%sum %two]
    :results [i32]
    :as %product)

;; 30 - 3 = 27
(op arith.subi
    :operands [%product %three]
    :results [i32]
    :as %result)

(op func.return
    :operands [%result])
