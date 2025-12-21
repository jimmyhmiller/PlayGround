; Simple hello world example
; For full LLVM support, use llvm dialect with proper linkage

(require-dialect [func :as f] [arith :as a])

(module
  (do
    ; Main function that returns hello world length (13)
    (f/func {:sym_name "main"
             :function_type (-> [] [i32])
             :llvm.emit_c_interface true}
      (region
        (block []
          ; Return length of "Hello, World!"
          (f/return (: 13 i32)))))))
