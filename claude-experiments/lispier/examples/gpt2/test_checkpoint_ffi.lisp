;; Test checkpoint loading via FFI
;;
;; This test verifies that:
;; 1. The GPT-2 checkpoint can be loaded via FFI
;; 2. The debug state can be loaded
;; 3. Parameters are accessible via pointers
;;
;; Run with: cargo run -- run examples/gpt2/test_checkpoint_ffi.lisp

(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect llvm)
(require-dialect scf)

;; Use the extern declaration to register the GPT-2 FFI
(extern :gpt2-ffi)
(link-library :c)

(module
  (do
    ;; External function declarations for GPT-2 FFI
    (func.func {:sym_name "gpt2_load_checkpoint"
                :function_type (-> [!llvm.ptr] [i32])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_load_debug_state"
                :function_type (-> [!llvm.ptr] [i32])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_get_params_ptr"
                :function_type (-> [] [!llvm.ptr])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_get_config"
                :function_type (-> [i32] [i64])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_debug_batch_size"
                :function_type (-> [] [i64])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_debug_seq_len"
                :function_type (-> [] [i64])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_debug_x_ptr"
                :function_type (-> [] [!llvm.ptr])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_debug_expected_logits_ptr"
                :function_type (-> [] [!llvm.ptr])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_debug_expected_loss"
                :function_type (-> [] [f32])
                :sym_visibility "private"})

    ;; printf for output
    (func.func {:sym_name "printf"
                :function_type (-> [!llvm.ptr "..."] [i32])
                :sym_visibility "private"})

    ;; Main test function
    (func.func {:sym_name "main"
                :function_type (-> [] [i32])}
      (region
        (block []
          ;; String constants for checkpoint paths
          (def checkpoint_path (llvm.mlir.addressof {:global_type !llvm.array<35xi8>} "checkpoint_path"))
          (def debug_path (llvm.mlir.addressof {:global_type !llvm.array<42xi8>} "debug_path"))

          ;; Load checkpoint
          (def load_result (func.call {:result i32} "gpt2_load_checkpoint" checkpoint_path))

          ;; Check if load succeeded
          (def zero_i32 (: 0 i32))
          (def load_ok (arith.cmpi {:predicate "eq"} load_result zero_i32))
          (scf.if load_ok
            (region
              (block []
                ;; Load debug state
                (def debug_result (func.call {:result i32} "gpt2_load_debug_state" debug_path))

                ;; Get config values
                (def c0 (: 0 i32))
                (def c1 (: 1 i32))
                (def c2 (: 2 i32))
                (def c3 (: 3 i32))
                (def c4 (: 4 i32))
                (def c5 (: 5 i32))

                (def max_seq_len (func.call {:result i64} "gpt2_get_config" c0))
                (def vocab_size (func.call {:result i64} "gpt2_get_config" c1))
                (def padded_vocab (func.call {:result i64} "gpt2_get_config" c2))
                (def num_layers (func.call {:result i64} "gpt2_get_config" c3))
                (def num_heads (func.call {:result i64} "gpt2_get_config" c4))
                (def channels (func.call {:result i64} "gpt2_get_config" c5))

                ;; Get debug state info
                (def batch_size (func.call {:result i64} "gpt2_debug_batch_size"))
                (def seq_len (func.call {:result i64} "gpt2_debug_seq_len"))
                (def expected_loss (func.call {:result f32} "gpt2_debug_expected_loss"))

                ;; Get parameter pointer
                (def params_ptr (func.call {:result !llvm.ptr} "gpt2_get_params_ptr"))

                ;; Load first few values from wte (token embeddings)
                (def wte_val0 (llvm.load {:result f32} params_ptr))
                (def one_i64 (: 1 i64))
                (def params_ptr1 (llvm.getelementptr {:result !llvm.ptr :elem_type f32} params_ptr one_i64))
                (def wte_val1 (llvm.load {:result f32} params_ptr1))

                ;; Print results using printf format strings
                (def fmt_config (llvm.mlir.addressof {:global_type !llvm.array<61xi8>} "fmt_config"))
                (func.call "printf" fmt_config max_seq_len vocab_size num_layers channels)

                (def fmt_debug (llvm.mlir.addressof {:global_type !llvm.array<43xi8>} "fmt_debug"))
                (def expected_loss_f64 (arith.extf {:result f64} expected_loss))
                (func.call "printf" fmt_debug batch_size seq_len expected_loss_f64)

                (def fmt_wte (llvm.mlir.addressof {:global_type !llvm.array<27xi8>} "fmt_wte"))
                (def wte_val0_f64 (arith.extf {:result f64} wte_val0))
                (def wte_val1_f64 (arith.extf {:result f64} wte_val1))
                (func.call "printf" fmt_wte wte_val0_f64 wte_val1_f64)

                (scf.yield)))
            (region
              (block []
                ;; Print error
                (def fmt_err (llvm.mlir.addressof {:global_type !llvm.array<26xi8>} "fmt_err"))
                (func.call "printf" fmt_err)
                (scf.yield))))

          (func.return zero_i32))))

    ;; Global string constants
    (llvm.mlir.global {:sym_name "checkpoint_path"
                       :linkage "internal"
                       :value "/home/jimmyhmiller/llm.c/gpt2_124M.bin\00"
                       :global_type !llvm.array<35xi8>})

    (llvm.mlir.global {:sym_name "debug_path"
                       :linkage "internal"
                       :value "/home/jimmyhmiller/llm.c/gpt2_124M_debug_state.bin\00"
                       :global_type !llvm.array<42xi8>})

    (llvm.mlir.global {:sym_name "fmt_config"
                       :linkage "internal"
                       :value "Config: max_seq=%ld, vocab=%ld, layers=%ld, channels=%ld\n\00"
                       :global_type !llvm.array<61xi8>})

    (llvm.mlir.global {:sym_name "fmt_debug"
                       :linkage "internal"
                       :value "Debug: batch=%ld, seq_len=%ld, loss=%f\n\00"
                       :global_type !llvm.array<43xi8>})

    (llvm.mlir.global {:sym_name "fmt_wte"
                       :linkage "internal"
                       :value "wte[0]=%f, wte[1]=%f\n\00"
                       :global_type !llvm.array<27xi8>})

    (llvm.mlir.global {:sym_name "fmt_err"
                       :linkage "internal"
                       :value "Failed to load checkpoint\n\00"
                       :global_type !llvm.array<26xi8>})))
