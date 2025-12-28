;; Minimal FFI test - with checkpoint loading

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)

(extern :gpt2-ffi)
(link-library :c)

(module
  (do
    (func.func {:sym_name "gpt2_load_checkpoint"
                :function_type (-> [!llvm.ptr] [i32])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_get_config"
                :function_type (-> [i32] [i64])
                :sym_visibility "private"})

    (func.func {:sym_name "main"
                :function_type (-> [] [i64])}
      (region
        (block []
          ;; Get address of the checkpoint path global
          (def checkpoint_path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))

          ;; Load checkpoint
          (def load_result (func.call {:result i32} "gpt2_load_checkpoint" checkpoint_path))

          ;; Get channels (index 5)
          (def idx (: 5 i32))
          (def result (func.call {:result i64} "gpt2_get_config" idx))
          (func.return result))))

    ;; Global string constant with proper region/block structure
    (llvm.mlir.global {:sym_name "checkpoint_path"
                       :linkage 0
                       :global_type !llvm.array<40 x i8>
                       :constant true}
      (region
        (block []
          (def _str_val (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_124M.bin\00" :result !llvm.array<40 x i8>}))
          (llvm.return _str_val))))))