; Test defn macro with parameters
; Expected output: 42

(require-dialect arith)
(require-dialect func)

(defn add [(: x i32) (: y i32)] -> i32
  (func.return (arith.addi x y)))

(defn main [] -> i32
  (def result (func.call {:result i32} "add" (: 21 i32) (: 21 i32)))
  (func.return result))
