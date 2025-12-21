; Double precision floating point
; Expected output: 5.859873

(require-dialect arith)
(require-dialect func)

(func.func {:sym_name "main"
            :function_type (-> [] [f64])
            :llvm.emit_c_interface true}
  (do
    (block []
      (func.return
        (arith.addf (: 3.14159265 f64) (: 2.71828 f64))))))