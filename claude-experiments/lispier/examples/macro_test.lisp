; Test macro expansion
; Expected output: 15

(require-dialect arith)
(require-dialect func)

(defn main [] -> i32
  (def x (: 10 i32))
  (def y (: 5 i32))
  (def result (arith.addi x y))
  (func.return result))
