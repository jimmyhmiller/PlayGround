; Multiplication example
; Expected output: 60

(require-dialect arith)
(require-dialect func)

(func.func {:sym_name "main"
            :function_type (-> [] [i32])
            :llvm.emit_c_interface true}
  (do
    (block []
      (func.return (arith.muli (: 10 i32) (: 6 i32))))))