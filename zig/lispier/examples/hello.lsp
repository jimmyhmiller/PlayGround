; Simple hello world example using LLVM dialect

(require-dialect [llvm :as l] [func :as f])

(module
  (do
    ; Define a global string constant
    (l/mlir.global {:sym_name "hello_str"
                    :type !llvm.array<14 x i8>
                    :value "Hello, World!\00"})

    ; Main function
    (f/func {:sym_name "main"
             :function_type (-> [] [i32])}
      (do
        (block []
          ; Get address of string
          (def str_addr (l/mlir.addressof {:global_name "hello_str"}))

          ; Print the string (assuming printf is available)
          ; For now, just return 0
          (f/return (: 0 i32)))))))
