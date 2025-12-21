;; Test file that requires the math library
(require ["./lib/math.lisp" :as math])
(require-dialect arith)
(require-dialect func)

(module
  (do
    (func.func {:sym_name "main"
                :function_type (-> [] [i64])}
      (do
        (block []
          (def x (func.call {:result i64} "add" (: 10 i64) (: 20 i64)))
          (func.return x))))))