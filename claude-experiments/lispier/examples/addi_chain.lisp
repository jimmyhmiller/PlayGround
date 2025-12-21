;; Example: Chain of arith operations
(require-dialect arith)

(module
  (let [a (: 10 i32)
        b (: 20 i32)]
    (arith.addi a b)))
