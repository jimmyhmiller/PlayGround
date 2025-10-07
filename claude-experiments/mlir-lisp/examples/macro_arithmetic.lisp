;; Macro for defining arithmetic operations more concisely

(defmacro const [value result_name]
  (op arith.constant
    :attrs {:value value}
    :results [i32]
    :as result_name))

(defmacro add [a b result_name]
  (op arith.addi
    :operands [a b]
    :results [i32]
    :as result_name))

(defmacro mul [a b result_name]
  (op arith.muli
    :operands [a b]
    :results [i32]
    :as result_name))

;; Use macros to compute (5 + 3) * 2 = 16
(const 5 %five)
(const 3 %three)
(const 2 %two)

(add %five %three %sum)
(mul %sum %two %result)

(op func.return :operands [%result])
