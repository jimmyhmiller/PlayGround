;; Test: Load weights from checkpoint and run matmul
;; Strategy: fread to buffer, copy to memref, run linalg.matmul

(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect llvm)
(require-dialect scf)
(require-dialect linalg)

(link-library :c)

;; Libc functions
(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fread (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64]))
(extern-fn fseek (-> [!llvm.ptr i64 i32] [i32]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

;; CPU pipeline for now - verify weight loading works

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

    ;; Simple matmul kernel for embedding lookup simulation
    ;; Takes first row of wte (768 values) and computes dot product
    (func.func {:sym_name "dot_test"
                :function_type (-> [memref<768xf32> memref<768xf32>] [f32])}
      (region
        (block [(: a memref<768xf32>) (: b memref<768xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c768 (: 768 index))
          (def zero (: 0.0 f32))

          (def result (scf.for {:result f32} c0 c768 c1 zero
            (region
              (block [(: i index) (: acc f32)]
                (def va (memref.load a i))
                (def vb (memref.load b i))
                (def prod (arith.mulf va vb))
                (def new_acc (arith.addf acc prod))
                (scf.yield new_acc)))))

          (func.return result))))))

;; Main function
(defn main [] -> i64
  ;; Open file
  (def path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
  (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
  (def file (call !llvm.ptr fopen path mode))

  ;; Skip 256-int header (1024 bytes)
  (def header_size (: 1024 i64))
  (def seek_set (: 0 i32))
  (def _skip (call i32 fseek file header_size seek_set))

  ;; Allocate raw buffer for first row of wte (768 floats = 3072 bytes)
  (def row_bytes (: 3072 i64))
  (def wte_row_ptr (call !llvm.ptr malloc row_bytes))

  ;; Read first row
  (def num_floats (: 768 i64))
  (def sizeof_f32 (: 4 i64))
  (def read_count (call i64 fread wte_row_ptr sizeof_f32 num_floats file))

  ;; Print first value
  (def first_val (llvm.load {:result f32} wte_row_ptr))
  (def first_val_f64 (arith.extf {:result f64} first_val))
  (print "wte[0][0] = %f\n" first_val_f64)

  ;; Allocate memref and copy data from raw buffer
  (def wte_row (memref.alloc {:result memref<768xf32>}))
  (def c0 (: 0 index))
  (def c1 (: 1 index))
  (def c768 (: 768 index))

  (scf.for c0 c768 c1
    (region
      (block [(: i index)]
        (def i_i64 (arith.index_cast {:result i64} i))
        (def val_ptr (ptr-at f32 wte_row_ptr i_i64))
        (def val (llvm.load {:result f32} val_ptr))
        (memref.store val wte_row i)
        (scf.yield))))

  ;; Verify first value in memref
  (def check_val (memref.load wte_row c0))
  (def check_val_f64 (arith.extf {:result f64} check_val))
  (print "memref[0] = %f\n" check_val_f64)

  ;; Create all-ones vector for dot product test
  (def ones (memref.alloc {:result memref<768xf32>}))
  (def one (: 1.0 f32))
  (scf.for c0 c768 c1
    (region
      (block [(: i index)]
        (memref.store one ones i)
        (scf.yield))))

  ;; Compute dot product (sum of wte[0][:])
  (def dot_result (func.call {:result f32} "dot_test" wte_row ones))
  (def dot_f64 (arith.extf {:result f64} dot_result))
  (print "dot(wte[0], ones) = %f\n" dot_f64)

  ;; Cleanup
  (memref.dealloc wte_row)
  (memref.dealloc ones)

  ;; Close file
  (def _close (call i32 fclose file))

  (func.return (: 0 i64)))
