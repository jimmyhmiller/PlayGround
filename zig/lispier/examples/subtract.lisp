; Subtraction example
; Expected output: 7

(require-dialect arith)
(require-dialect func)

(func.func {:sym_name "main"
            :function_type (-> [] [i32])
            :llvm.emit_c_interface true}
  (do
    (block []
      (func.return (arith.subi (: 10 i32) (: 3 i32))))))