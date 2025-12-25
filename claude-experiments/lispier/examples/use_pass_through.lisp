; Example of using pass_through macro
; This file imports the pass_through macro from macro_double_value.lisp

(require-dialect func)
(require-dialect arith)
(require-macros "./macro_double_value.lisp")

; The pass_through macro should be available now
; (pass_through x) expands to x

(defn main [] -> i64
  ; Use the JIT-compiled pass_through macro
  (func.return (pass_through (: 99 i64))))
