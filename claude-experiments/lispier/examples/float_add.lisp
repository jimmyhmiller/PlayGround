; Floating point addition
; Expected output: 5.140000

(require-dialect arith)
(require-dialect func)

(func.func {:sym_name "main"
            :function_type (-> [] [f32])
            :llvm.emit_c_interface true}
  (do
    (block []
      (func.return (arith.addf (: 3.14 f32) (: 2.0 f32))))))