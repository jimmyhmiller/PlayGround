;; Test: Load layer 0 QKV weights and run matmul
;; Weight layout from gpt2.rs:
;; - wte: 50257 * 768 = 38597376
;; - wpe: 1024 * 768 = 786432
;; - ln1w: 12 * 768 = 9216
;; - ln1b: 12 * 768 = 9216
;; - qkvw: 12 * 768 * 2304 = 21233664
;; Layer 0 qkvw starts at: 38597376 + 786432 + 9216 + 9216 = 39402240

(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect llvm)
(require-dialect scf)
(require-dialect linalg)

(link-library :c)

(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fread (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64]))
(extern-fn fseek (-> [!llvm.ptr i64 i32] [i32]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

(module
  (do
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

    ;; Simple matmul kernel: (1,768) @ (768,2304) -> (1,2304)
    ;; Just compute first row to test
    (func.func {:sym_name "qkv_matmul_row"
                :function_type (-> [memref<2304xf32>     ; out (1 row of 2304)
                                    memref<768xf32>       ; inp (1 row of 768)
                                    memref<768x2304xf32>] [])} ; weight
      (region
        (block [(: out memref<2304xf32>)
                (: inp memref<768xf32>)
                (: weight memref<768x2304xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c768 (: 768 index))
          (def c2304 (: 2304 index))
          (def zero (: 0.0 f32))

          ;; out[j] = sum_i inp[i] * weight[i,j]
          (scf.for c0 c2304 c1
            (region
              (block [(: j index)]
                (def sum (scf.for {:result f32} c0 c768 c1 zero
                  (region
                    (block [(: i index) (: acc f32)]
                      (def inp_val (memref.load inp i))
                      (def w_val (memref.load weight i j))
                      (def prod (arith.mulf inp_val w_val))
                      (def new_acc (arith.addf acc prod))
                      (scf.yield new_acc)))))
                (memref.store sum out j)
                (scf.yield))))
          (func.return))))))

;; Offset constants (in floats, not bytes)
;; Header: 256 ints = 1024 bytes (but we skip this with fseek)
;; wte: 50257 * 768 = 38597376
;; wpe: 1024 * 768 = 786432
;; ln1w: 12 * 768 = 9216
;; ln1b: 12 * 768 = 9216
;; qkvw starts at: 38597376 + 786432 + 9216 + 9216 = 39402240
;; Layer 0 qkvw: offset 0 within qkvw block, size 768*2304 = 1769472

(defn main [] -> i64
  ;; Open file
  (def path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
  (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
  (def file (call !llvm.ptr fopen path mode))

  ;; Skip header (256 ints = 1024 bytes)
  (def header_offset (: 1024 i64))
  (def seek_set (: 0 i32))
  (def _skip1 (call i32 fseek file header_offset seek_set))

  ;; Skip to qkvw: 39402240 floats = 157608960 bytes
  (def qkvw_offset_bytes (: 157608960 i64))
  (def seek_cur (: 1 i32))
  (def _skip2 (call i32 fseek file qkvw_offset_bytes seek_cur))

  ;; Allocate buffer for layer 0 qkvw (768 * 2304 = 1769472 floats)
  (def qkvw_size (: 1769472 i64))
  (def sizeof_f32 (: 4 i64))
  (def qkvw_bytes (arith.muli qkvw_size sizeof_f32))
  (def qkvw_ptr (call !llvm.ptr malloc qkvw_bytes))

  ;; Read layer 0 qkvw
  (def read_count (call i64 fread qkvw_ptr sizeof_f32 qkvw_size file))
  (print "Read %ld floats for qkvw\n" read_count)

  ;; Print first value
  (def first_w (llvm.load {:result f32} qkvw_ptr))
  (def first_w_f64 (arith.extf {:result f64} first_w))
  (print "qkvw[0][0] = %f\n" first_w_f64)

  ;; Copy to memref (768x2304)
  (def qkvw (memref.alloc {:result memref<768x2304xf32>}))
  (def c0 (: 0 index))
  (def c1 (: 1 index))
  (def c768 (: 768 index))
  (def c2304 (: 2304 index))

  (scf.for c0 c768 c1
    (region
      (block [(: i index)]
        (scf.for c0 c2304 c1
          (region
            (block [(: j index)]
              ;; Linear index: i * 2304 + j
              (def i_i64 (arith.index_cast {:result i64} i))
              (def j_i64 (arith.index_cast {:result i64} j))
              (def c2304_i64 (: 2304 i64))
              (def idx (arith.addi (arith.muli i_i64 c2304_i64) j_i64))
              (def val_ptr (ptr-at f32 qkvw_ptr idx))
              (def val (llvm.load {:result f32} val_ptr))
              (memref.store val qkvw i j)
              (scf.yield))))
        (scf.yield))))

  ;; Verify first weight in memref
  (def check_w (memref.load qkvw c0 c0))
  (def check_w_f64 (arith.extf {:result f64} check_w))
  (print "memref qkvw[0][0] = %f\n" check_w_f64)

  ;; Create test input (all ones)
  (def inp (memref.alloc {:result memref<768xf32>}))
  (def one (: 1.0 f32))
  (scf.for c0 c768 c1
    (region
      (block [(: i index)]
        (memref.store one inp i)
        (scf.yield))))

  ;; Allocate output
  (def out (memref.alloc {:result memref<2304xf32>}))

  ;; Run QKV matmul
  (func.call "qkv_matmul_row" out inp qkvw)

  ;; Print first output (should be sum of qkvw column 0)
  (def out0 (memref.load out c0))
  (def out0_f64 (arith.extf {:result f64} out0))
  (print "qkv_out[0] = %f\n" out0_f64)

  ;; Cleanup
  (memref.dealloc qkvw)
  (memref.dealloc inp)
  (memref.dealloc out)
  (def _close (call i32 fclose file))

  (func.return (: 0 i64)))
