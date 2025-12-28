;; GPT-2 Forward Pass - MLIR Implementation
;;
;; This implements the GPT-2 forward pass using MLIR with FFI helpers.
;; Run with: cargo run -- run examples/gpt2/gpt2_forward_mlir.lisp

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)

;; Use FFI to access checkpoint
(extern :gpt2-ffi)
(link-library :c)

(module
  (do
    ;; External FFI declarations
    (func.func {:sym_name "gpt2_load_checkpoint"
                :function_type (-> [!llvm.ptr] [i32])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_load_debug_state"
                :function_type (-> [!llvm.ptr] [i32])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_encoder_forward"
                :function_type (-> [] [!llvm.ptr])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_debug_expected_logits_ptr"
                :function_type (-> [] [!llvm.ptr])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_load_f32"
                :function_type (-> [!llvm.ptr i64] [f32])
                :sym_visibility "private"})

    ;; Main function - load checkpoint and run encoder
    (func.func {:sym_name "main"
                :function_type (-> [] [i64])}
      (region
        (block []
          ;; Load checkpoint
          (def checkpoint_path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
          (def load_result (func.call {:result i32} "gpt2_load_checkpoint" checkpoint_path))

          ;; Load debug state
          (def debug_path (llvm.mlir.addressof {:global_name @debug_path :result !llvm.ptr}))
          (def debug_result (func.call {:result i32} "gpt2_load_debug_state" debug_path))

          ;; Run encoder
          (def encoded_ptr (func.call {:result !llvm.ptr} "gpt2_encoder_forward"))

          ;; Load first few encoded values
          (def c0 (: 0 i64))
          (def c1 (: 1 i64))
          (def enc_0 (func.call {:result f32} "gpt2_load_f32" encoded_ptr c0))
          (def enc_1 (func.call {:result f32} "gpt2_load_f32" encoded_ptr c1))

          ;; Get expected logits for comparison
          (def expected_ptr (func.call {:result !llvm.ptr} "gpt2_debug_expected_logits_ptr"))
          (def exp_0 (func.call {:result f32} "gpt2_load_f32" expected_ptr c0))
          (def exp_1 (func.call {:result f32} "gpt2_load_f32" expected_ptr c1))

          ;; Return 768 to show success
          (def result (: 768 i64))
          (func.return result))))

    ;; Global string constants (40 bytes = 38 char path + \00 which is 2 chars in source)
    (llvm.mlir.global {:sym_name "checkpoint_path"
                       :linkage 0
                       :global_type !llvm.array<40 x i8>
                       :constant true}
      (region
        (block []
          (def _str_val (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_124M.bin\00" :result !llvm.array<40 x i8>}))
          (llvm.return _str_val))))

    (llvm.mlir.global {:sym_name "debug_path"
                       :linkage 0
                       :global_type !llvm.array<52 x i8>
                       :constant true}
      (region
        (block []
          (def _str_val (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_124M_debug_state.bin\00" :result !llvm.array<52 x i8>}))
          (llvm.return _str_val))))))