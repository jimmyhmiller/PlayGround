; Type inference - bare numbers infer type from context
; Expected output: 30

(require-dialect arith)
(require-dialect func)

(func.func {:sym_name "main"
            :function_type (-> [] [i32])
            :llvm.emit_c_interface true}
  (do
    (block []
      (def x (: 10 i32))
      ; The 20 infers i32 from x
      (def y (arith.addi x 20))
      (func.return y))))