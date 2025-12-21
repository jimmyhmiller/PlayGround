;; Math library module
(require-dialect arith)
(require-dialect func)

(module
  (do
    (func.func {:sym_name "add"
                :function_type (-> [i64 i64] [i64])}
      (do
        (block [(: x i64) (: y i64)]
          (def result (arith.addi x y))
          (func.return result))))))