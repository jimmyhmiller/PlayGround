; Example: Inline macro definition with dynamic compilation
; This demonstrates the dynamic macro compilation feature
;
; When the file is loaded, the const42 macro is compiled and registered,
; then it can be used in the main function.

(require-dialect func)
(require-dialect llvm)
(require-dialect arith)

(extern :value-ffi)

; FFI declarations needed for macro implementation
(func.func {:sym_name "value_list_new" :function_type (-> [] [!llvm.ptr]) :sym_visibility "private"})
(func.func {:sym_name "value_list_push" :function_type (-> [!llvm.ptr !llvm.ptr] []) :sym_visibility "private"})
(func.func {:sym_name "value_list_first" :function_type (-> [!llvm.ptr] [!llvm.ptr]) :sym_visibility "private"})
(func.func {:sym_name "value_symbol_arith_addi" :function_type (-> [] [!llvm.ptr]) :sym_visibility "private"})
(func.func {:sym_name "value_number_new" :function_type (-> [f64] [!llvm.ptr]) :sym_visibility "private"})

; A simple macro that adds 21 to its argument: (add21 x) -> (arith.addi x 21)
(func.func {:sym_name "add21"
            :function_type (-> [!llvm.ptr] [!llvm.ptr])}
  (do
    (block [(: form !llvm.ptr)]
      ; Get the argument
      (def x (func.call {:callee @value_list_first :result !llvm.ptr} form))
      ; Build (arith.addi x 21)
      (def result (func.call {:callee @value_list_new :result !llvm.ptr}))
      (def sym (func.call {:callee @value_symbol_arith_addi :result !llvm.ptr}))
      (func.call {:callee @value_list_push} result sym)
      (func.call {:callee @value_list_push} result x)
      (def twenty_one (func.call {:callee @value_number_new :result !llvm.ptr} (: 21.0 f64)))
      (func.call {:callee @value_list_push} result twenty_one)
      (func.return result))))

; Register add21 as a macro - this triggers dynamic compilation!
(defmacro add21)

; Now add21 is available as a macro. Use it in main:
(defn main [] -> i64
  (func.return (add21 (: 21 i64))))
