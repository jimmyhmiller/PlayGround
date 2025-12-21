;; Test file that uses project-relative path
(require ["@examples/lib/math.lisp" :as math])
(require-dialect func)

(module
  (do
    (func.func {:sym_name "main"
                :function_type (-> [] [i64])}
      (do
        (block []
          (def x (func.call {:result i64} "add" (: 5 i64) (: 7 i64)))
          (func.return x))))))