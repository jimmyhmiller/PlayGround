;; Test: Load weights from checkpoint directly into memref
;; Strategy: Allocate memref, extract its pointer, use fread

(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect llvm)
(require-dialect scf)
(require-dialect linalg)
(require-dialect gpu)

(link-library :c)

;; Libc functions
(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fread (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64]))
(extern-fn fseek (-> [!llvm.ptr i64 i32] [i32]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

;; No compilation spec - use default JIT for CPU

(module
  (do
    ;; Checkpoint path
    (llvm.mlir.global {:sym_name "checkpoint_path"
                       :linkage 0
                       :global_type !llvm.array<39 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_124M.bin\0" :result !llvm.array<39 x i8>}))
          (llvm.return s))))

    (llvm.mlir.global {:sym_name "read_mode"
                       :linkage 0
                       :global_type !llvm.array<3 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "rb\0" :result !llvm.array<3 x i8>}))
          (llvm.return s))))

))

;; Small test - load first 4 floats of wte and print
(defn main [] -> i64
  ;; Open file
  (def path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
  (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
  (def file (call !llvm.ptr fopen path mode))

  ;; Skip 256-int header (1024 bytes)
  (def header_size (: 1024 i64))
  (def seek_set (: 0 i32))
  (def _skip (call i32 fseek file header_size seek_set))

  ;; Allocate raw buffer for first row of wte (768 floats)
  (def row_size (: 3072 i64))  ; 768 * 4 bytes
  (def wte_row_ptr (call !llvm.ptr malloc row_size))

  ;; Read first row
  (def num_floats (: 768 i64))
  (def sizeof_f32 (: 4 i64))
  (def read_count (call i64 fread wte_row_ptr sizeof_f32 num_floats file))

  ;; Print first value
  (def first_val (llvm.load {:result f32} wte_row_ptr))
  (def first_val_f64 (arith.extf {:result f64} first_val))
  (print "wte[0][0] = %f\n" first_val_f64)

  ;; Close file
  (def _close (call i32 fclose file))

  (func.return (: 0 i64)))
