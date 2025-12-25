; A macro that doubles a value using quasiquote: (double x) -> (arith.addi x x)
; This demonstrates the new quasiquote syntax

(require-dialect func)
(require-dialect llvm)
(require-dialect arith)

(extern :value-ffi)

; Declare external FFI functions needed by quasiquote expansion
(func.func {:sym_name "value_list_new" :function_type (-> [] [!llvm.ptr]) :sym_visibility "private"})
(func.func {:sym_name "value_list_push" :function_type (-> [!llvm.ptr !llvm.ptr] []) :sym_visibility "private"})
(func.func {:sym_name "value_list_first" :function_type (-> [!llvm.ptr] [!llvm.ptr]) :sym_visibility "private"})
(func.func {:sym_name "value_symbol_arith_addi" :function_type (-> [] [!llvm.ptr]) :sym_visibility "private"})

; The double macro using quasiquote: (double x) -> (arith.addi x x)
(func.func {:sym_name "double"
            :function_type (-> [!llvm.ptr] [!llvm.ptr])}
  (do
    (block [(: form !llvm.ptr)]
      (def x (func.call {:callee @value_list_first :result !llvm.ptr} form))
      ; Use quasiquote! ~x unquotes the variable x
      ; Note: quasiquote expands to a (do ...) block, so we bind it first
      (def result `(arith.addi ~x ~x))
      (func.return result))))

(defmacro double)

; Test main
(defn main [] -> i64
  (func.return (: 0 i64)))
