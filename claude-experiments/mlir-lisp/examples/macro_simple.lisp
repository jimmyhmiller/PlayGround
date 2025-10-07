;; Simple macro test
;; Define a macro that adds 1 to a value

(defmacro inc [x]
  (op arith.addi
    :operands [x %one]
    :results [i32]
    :as %result))

;; Use the macro
(op arith.constant :attrs {:value 1} :results [i32] :as %one)
(op arith.constant :attrs {:value 41} :results [i32] :as %forty_one)

(inc %forty_one)

(op func.return :operands [%result])
