; Using variables with def
; Expected output: 15

(require-dialect arith)
(require-dialect func)

(func.func {:sym_name "main"
            :function_type (-> [] [i32])
            :llvm.emit_c_interface true}
  (do
    (block []
      (def x (: 10 i32))
      (def y (: 5 i32))
      (def result (arith.addi x y))
      (func.return result))))