;; Complete if-then-else using arith.cmpi and cf.cond_br
;; Compares 10 vs 5, returns 42 if true, 0 if false

(block entry []
  ;; Create constants
  (op arith.constant
      :attrs {:value 10}
      :results [i32]
      :as %ten)

  (op arith.constant
      :attrs {:value 5}
      :results [i32]
      :as %five)

  ;; Compare: is 10 > 5?
  (op arith.cmpi
      :attrs {:predicate "sgt"}
      :operands [%ten %five]
      :results [i1]
      :as %cond)

  ;; Conditional branch
  (op cf.cond_br
      :operands [%cond]
      :true then_block
      :false else_block))

(block then_block []
  ;; True case: return 42
  (op arith.constant
      :attrs {:value 42}
      :results [i32]
      :as %result_true)

  (op cf.br
      :dest exit_block
      :args [%result_true]))

(block else_block []
  ;; False case: return 0
  (op arith.constant
      :attrs {:value 0}
      :results [i32]
      :as %result_false)

  (op cf.br
      :dest exit_block
      :args [%result_false]))

(block exit_block [i32]
  ;; Block argument ^0 contains the result
  (op func.return
      :operands [^0]))
