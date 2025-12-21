; Simple addition example
; Expected output: 42

(require-dialect arith)
(require-dialect func)

(func.func {:sym_name "main"
            :function_type (-> [] [i32])
            :llvm.emit_c_interface true}
  (do
    (block []
      (func.return (arith.addi (: 21 i32) (: 21 i32))))))