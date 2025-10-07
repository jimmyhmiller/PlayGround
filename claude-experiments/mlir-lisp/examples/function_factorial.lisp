;; Factorial using recursion (conceptual - need proper conditionals)
;; For now, a simpler function that multiplies and calls helper

(defn square [x:i32] i32
  (op arith.muli
      :operands [x x]
      :results [i32]
      :as %result)
  (op func.return
      :operands [%result]))

(defn add_one [x:i32] i32
  (op arith.constant
      :attrs {:value 1}
      :results [i32]
      :as %one)

  (op arith.addi
      :operands [x %one]
      :results [i32]
      :as %result)

  (op func.return
      :operands [%result]))

(defn main [] i32
  ;; Compute square(5) + 1
  (op arith.constant
      :attrs {:value 5}
      :results [i32]
      :as %five)

  (op func.call
      :attrs {:callee "square"}
      :operands [%five]
      :results [i32]
      :as %squared)

  (op func.call
      :attrs {:callee "add_one"}
      :operands [%squared]
      :results [i32]
      :as %result)

  (op func.return
      :operands [%result]))
