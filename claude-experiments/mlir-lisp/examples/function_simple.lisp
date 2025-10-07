;; Simple function that adds two numbers

(defn add [x:i32 y:i32] i32
  (op arith.addi
      :operands [x y]
      :results [i32]
      :as %result)
  (op func.return
      :operands [%result]))

(defn main [] i32
  (op arith.constant
      :attrs {:value 5}
      :results [i32]
      :as %five)

  (op arith.constant
      :attrs {:value 10}
      :results [i32]
      :as %ten)

  (op func.call
      :attrs {:callee "add"}
      :operands [%five %ten]
      :results [i32]
      :as %result)

  (op func.return
      :operands [%result]))
