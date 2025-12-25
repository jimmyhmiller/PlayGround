; Example of using the quasiquote-based double macro
(require-dialect func)
(require-dialect arith)
(require-macros "./macro_double_qq.lisp")

; (double (: 21 i64)) should expand to (arith.addi (: 21 i64) (: 21 i64)) = 42

(defn main [] -> i64
  (func.return (double (: 21 i64))))
