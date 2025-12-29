;; Test without llvm.func

(require-dialect func)
(require-dialect arith)

(defn main [] -> i64
  (func.return (: 0 i64)))
