;; Simple test for checkpoint loading via FFI
;;
;; This test just loads the checkpoint and returns success/failure.
;; Run with: cargo run -- run examples/gpt2/test_checkpoint_simple.lisp

(require-dialect func)
(require-dialect arith)

;; Use the extern declaration to register the GPT-2 FFI
(extern :gpt2-ffi)
(link-library :c)

(module
  (do
    ;; External function declarations for GPT-2 FFI
    ;; Note: We use !llvm.ptr for C strings
    (func.func {:sym_name "gpt2_load_checkpoint"
                :function_type (-> [!llvm.ptr] [i32])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_get_config"
                :function_type (-> [i32] [i64])
                :sym_visibility "private"})

    (func.func {:sym_name "gpt2_get_params_ptr"
                :function_type (-> [] [!llvm.ptr])
                :sym_visibility "private"})

    ;; We'll test by calling the Rust functions that were already loaded
    ;; during testing. For a real test, we'd need to pass the path.

    ;; For now, just test that config returns correct values after loading
    (func.func {:sym_name "main"
                :function_type (-> [] [i64])}
      (region
        (block []
          ;; Get channels config - should be 768 if checkpoint was loaded
          (def config_idx (: 5 i32))  ; index 5 = channels
          (def channels (func.call {:result i64} "gpt2_get_config" config_idx))

          ;; Return the channels value (768 if loaded, -1 if not)
          (func.return channels))))))
