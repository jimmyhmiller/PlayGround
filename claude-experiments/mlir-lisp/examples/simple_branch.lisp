;; Simple unconditional branch example
;; Just creates a value and branches to return it

(block entry []
  ;; Create constant
  (op arith.constant
      :attrs {:value 42}
      :results [i32]
      :as %result)

  ;; Branch to exit with the value
  (op cf.br
      :dest exit_block
      :args [%result]))

(block exit_block [i32]
  ;; Block argument ^0 contains the result
  (op func.return
      :operands [^0]))
