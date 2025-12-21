; Large i64 numbers
; Expected output: 3000000000000

(require-dialect arith)
(require-dialect func)

(func.func {:sym_name "main"
            :function_type (-> [] [i64])
            :llvm.emit_c_interface true}
  (do
    (block []
      (func.return
        (arith.addi (: 1000000000000 i64) (: 2000000000000 i64))))))