; Example of using JIT-compiled macros via require-macros
; This file imports the double macro from macro_double.lisp

(require-dialect func)
(require-dialect arith)
(require-macros "./macro_double.lisp")

; The double macro transforms (double x) into (arith.addi x x)
; So (double (: 21 i64)) becomes (arith.addi (: 21 i64) (: 21 i64)) = 42

(defn main [] -> i64
  ; Use the JIT-compiled double macro
  ; 21 + 21 = 42
  (func.return (double (: 21 i64))))
