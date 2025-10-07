;; A simple MLIR-Lisp program
;; This demonstrates the basic operation syntax
;; Returns 42

(op arith.constant
    :attrs {:value 42}
    :results [i32]
    :as %result)

(op func.return
    :operands [%result])
