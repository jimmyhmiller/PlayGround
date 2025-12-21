; Nested operations: (10 + 20) * 3
; Expected output: 90

(require-dialect arith)
(require-dialect func)

(func.func {:sym_name "main"
            :function_type (-> [] [i32])
            :llvm.emit_c_interface true}
  (do
    (block []
      (func.return
        (arith.muli
          (arith.addi (: 10 i32) (: 20 i32))
          (: 3 i32))))))