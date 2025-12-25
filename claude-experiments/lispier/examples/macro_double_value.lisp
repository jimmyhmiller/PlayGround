; A more practical macro: pass-through that proves the FFI works
; (pass-through x) -> x (just returns the first argument)

(require-dialect func)
(require-dialect llvm)

(extern :value-ffi)

; Declare external FFI functions
(func.func {:sym_name "value_list_first" :function_type (-> [!llvm.ptr] [!llvm.ptr]) :sym_visibility "private"})

; The macro function: (pass-through x) -> x
(func.func {:sym_name "pass_through"
            :function_type (-> [!llvm.ptr] [!llvm.ptr])}
  (do
    (block [(: form !llvm.ptr)]
      ; Get the first argument and return it
      (def result (func.call {:callee @value_list_first :result !llvm.ptr} form))
      (func.return result))))

; Register this function as a macro
(defmacro pass_through)

; Test - verify it compiles
(defn main [] -> i64
  (func.return (: 0 i64)))
