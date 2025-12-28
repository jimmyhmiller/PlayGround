;; Test for ptr-at macro - dynamic pointer indexing
;;
;; This tests the new ptr-at macro that supports dynamic (runtime) indices
;; for llvm.getelementptr, which is required for GPT-2 forward pass.
;;
;; Run with: cargo run -- run examples/gpt2/test_ptr_at.lisp

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)

(extern :gpt2-ffi)
(link-library :c)

(module
  (do
    ;; External FFI to get parameter pointer
    (func.func {:sym_name "gpt2_load_checkpoint"
                :function_type (-> [!llvm.ptr] [i32])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_get_params_ptr"
                :function_type (-> [] [!llvm.ptr])
                :sym_visibility "private"})

    ;; Main test - load checkpoint and access parameters at dynamic index
    (func.func {:sym_name "main"
                :function_type (-> [] [i64])}
      (region
        (block []
          ;; Load checkpoint
          (def checkpoint_path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
          (def load_result (func.call {:result i32} "gpt2_load_checkpoint" checkpoint_path))

          ;; Get parameters pointer
          (def params (func.call {:result !llvm.ptr} "gpt2_get_params_ptr"))

          ;; Test dynamic indexing with ptr-at
          ;; Access params[5] using dynamic index
          (def idx (: 5 i64))
          (def ptr_at_5 (ptr-at f32 params idx))
          (def val_5 (llvm.load {:result f32} ptr_at_5))

          ;; Access params[10] using computed index (2 * 5)
          (def two (: 2 i64))
          (def five (: 5 i64))
          (def idx_10 (arith.muli two five))
          (def ptr_at_10 (ptr-at f32 params idx_10))
          (def val_10 (llvm.load {:result f32} ptr_at_10))

          ;; Return 768 to indicate success
          (def result (: 768 i64))
          (func.return result))))

    ;; Global string constant
    (llvm.mlir.global {:sym_name "checkpoint_path"
                       :linkage 0
                       :global_type !llvm.array<40 x i8>
                       :constant true}
      (region
        (block []
          (def _str_val (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_124M.bin\00" :result !llvm.array<40 x i8>}))
          (llvm.return _str_val))))))
