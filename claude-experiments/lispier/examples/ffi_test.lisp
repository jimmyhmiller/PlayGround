; Test FFI integration - call Value manipulation functions from compiled code
; This proves that the extern :value-ffi declaration works and FFI symbols are registered

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)

(extern :value-ffi)

; Declare external FFI functions using func.func with sym_visibility private
; value_list_new() -> !llvm.ptr
(func.func {:sym_name "value_list_new"
            :function_type (-> [] [!llvm.ptr])
            :sym_visibility "private"})

; value_number_new(f64) -> !llvm.ptr
(func.func {:sym_name "value_number_new"
            :function_type (-> [f64] [!llvm.ptr])
            :sym_visibility "private"})

; value_list_push(!llvm.ptr, !llvm.ptr) -> void
(func.func {:sym_name "value_list_push"
            :function_type (-> [!llvm.ptr !llvm.ptr] [])
            :sym_visibility "private"})

; value_list_len(!llvm.ptr) -> i64
(func.func {:sym_name "value_list_len"
            :function_type (-> [!llvm.ptr] [i64])
            :sym_visibility "private"})

; value_free(!llvm.ptr) -> void
(func.func {:sym_name "value_free"
            :function_type (-> [!llvm.ptr] [])
            :sym_visibility "private"})

(defn main [] -> i64
  ; Create a new list
  (def list (func.call {:callee @value_list_new :result !llvm.ptr}))

  ; Create numbers and add them to the list
  (def n1 (func.call {:callee @value_number_new :result !llvm.ptr}
            (: 1.0 f64)))
  (func.call {:callee @value_list_push} list n1)

  (def n2 (func.call {:callee @value_number_new :result !llvm.ptr}
            (: 2.0 f64)))
  (func.call {:callee @value_list_push} list n2)

  (def n3 (func.call {:callee @value_number_new :result !llvm.ptr}
            (: 3.0 f64)))
  (func.call {:callee @value_list_push} list n3)

  ; Get the length - should be 3
  (def len (func.call {:callee @value_list_len :result i64} list))

  ; Clean up
  (func.call {:callee @value_free} n1)
  (func.call {:callee @value_free} n2)
  (func.call {:callee @value_free} n3)
  (func.call {:callee @value_free} list)

  (func.return len))
