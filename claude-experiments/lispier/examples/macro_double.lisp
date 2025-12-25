; A macro that doubles a value: (double x) -> (arith.addi x x)
; This demonstrates a macro that actually constructs new code

(require-dialect func)
(require-dialect llvm)

(extern :value-ffi)

; Declare external FFI functions
(func.func {:sym_name "value_list_new" :function_type (-> [] [!llvm.ptr]) :sym_visibility "private"})
(func.func {:sym_name "value_list_push" :function_type (-> [!llvm.ptr !llvm.ptr] []) :sym_visibility "private"})
(func.func {:sym_name "value_list_first" :function_type (-> [!llvm.ptr] [!llvm.ptr]) :sym_visibility "private"})
(func.func {:sym_name "value_symbol_arith_addi" :function_type (-> [] [!llvm.ptr]) :sym_visibility "private"})

; The double macro: (double x) -> (arith.addi x x)
; Input: form = [x] (the arguments after the macro name)
; Output: (arith.addi x x)
(func.func {:sym_name "double"
            :function_type (-> [!llvm.ptr] [!llvm.ptr])}
  (do
    (block [(: form !llvm.ptr)]
      ; Get the argument x
      (def x (func.call {:callee @value_list_first :result !llvm.ptr} form))

      ; Build: (arith.addi x x)
      ; 1. Create a new list
      (def result (func.call {:callee @value_list_new :result !llvm.ptr}))

      ; 2. Create the symbol "arith.addi"
      (def addi-sym (func.call {:callee @value_symbol_arith_addi :result !llvm.ptr}))

      ; 3. Push symbol onto result list
      (func.call {:callee @value_list_push} result addi-sym)

      ; 4. Push x twice (for x + x)
      (func.call {:callee @value_list_push} result x)
      (func.call {:callee @value_list_push} result x)

      (func.return result))))

; Register this function as a macro
(defmacro double)

; Test main
(defn main [] -> i64
  (func.return (: 0 i64)))
