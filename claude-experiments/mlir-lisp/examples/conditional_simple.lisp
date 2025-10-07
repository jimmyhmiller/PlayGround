;; Simple conditional branch example
;; Uses a pre-computed boolean value instead of cmpi
;; Returns 100 from true branch, 200 from false branch

(block entry []
  ;; Create a boolean constant (1 = true, 0 = false)
  (op arith.constant
      :attrs {:value 1}
      :results [i1]
      :as %condition)

  ;; Conditional branch based on the boolean
  (op cf.cond_br
      :operands [%condition]
      :true then_block
      :false else_block))

(block then_block []
  ;; True case: return 100
  (op arith.constant
      :attrs {:value 100}
      :results [i32]
      :as %result_true)

  (op cf.br
      :dest exit_block
      :args [%result_true]))

(block else_block []
  ;; False case: return 200
  (op arith.constant
      :attrs {:value 200}
      :results [i32]
      :as %result_false)

  (op cf.br
      :dest exit_block
      :args [%result_false]))

(block exit_block [i32]
  ;; Block argument ^0 contains the result
  (op func.return
      :operands [^0]))
