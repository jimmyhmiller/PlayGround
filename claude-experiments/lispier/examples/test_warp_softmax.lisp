;; Test fused softmax with warp-level reductions
;; Softmax: exp(x - max) / sum(exp(x - max))
;;
;; One warp (32 threads) per row, each thread handles 2 elements (64/32)
;; Uses warp shuffle for max and sum reductions

(require-dialect memref)
(require-dialect arith)
(require-dialect func)
(require-dialect gpu)
(require-dialect scf)
(require-dialect math)

(compilation
  (target rocm
    (pass lower-affine)
    (pass convert-scf-to-cf)
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
    (pass gpu-to-llvm {:use-bare-pointers-for-kernels true})
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-math-to-llvm)
    (pass convert-index-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; External function declarations
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"}
      (region))

    (defn main []
      ;; Test: 4 rows of 64 elements each
      ;; Input: row i has value i+1 for all elements (constant per row)
      ;; Expected softmax output: 1/64 for all elements in each row
      (def num_rows (: 4 index))
      (def row_size (: 64 index))
      (def c0 (: 0 index))
      (def c1 (: 1 index))
      (def c32 (: 32 index))

      ;; Allocate input and output
      (def inp (memref.alloc {:result memref<4x64xf32>}))
      (def out (memref.alloc {:result memref<4x64xf32>}))

      ;; Initialize input: each row has constant value = row_idx + 1
      (scf.for c0 num_rows c1
        (region
          (block [(: row index)]
            (def row_val_i64 (arith.index_cast {:result i64} row))
            (def row_val_f32 (arith.sitofp {:result f32} row_val_i64))
            (def base_val (arith.addf row_val_f32 (: 1.0 f32)))
            (scf.for c0 row_size c1
              (region
                (block [(: col index)]
                  (memref.store base_val inp row col)
                  (scf.yield))))
            (scf.yield))))

      ;; Register memory for GPU
      (def inp_unranked (memref.cast {:result "memref<*xf32>"} inp))
      (def out_unranked (memref.cast {:result "memref<*xf32>"} out))
      (gpu.host_register inp_unranked)
      (gpu.host_register out_unranked)

      ;; Shuffle constants
      (def c16_i32 (: 16 i32))
      (def c8_i32 (: 8 i32))
      (def c4_i32 (: 4 i32))
      (def c2_i32 (: 2 i32))
      (def c1_i32 (: 1 i32))
      (def c32_i32 (: 32 i32))

      ;; Fused softmax kernel: one warp (32 threads) per row
      ;; Each thread handles 64/32 = 2 elements
      (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
        num_rows c1 c1 c32 c1 c1
        (region
          (block [(: row index) (: _by index) (: _bz index)
                  (: lane index) (: _ty index) (: _tz index)
                  (: _gridDimX index) (: _gridDimY index) (: _gridDimZ index)
                  (: _blockDimX index) (: _blockDimY index) (: _blockDimZ index)]

            (def neg_inf (: -1e30 f32))
            (def zero (: 0.0 f32))
            (def c2 (: 2 index))

            ;; Step 1: Each thread finds local max of its 2 elements
            (def local_max (scf.for {:result f32} c0 c2 c1 neg_inf
              (region
                (block [(: i index) (: m f32)]
                  (def offset (arith.muli i c32))
                  (def col (arith.addi lane offset))
                  (def x (memref.load {:result f32} inp row col))
                  (def new_m (arith.maximumf m x))
                  (scf.yield new_m)))))

            ;; Step 2: Warp reduce max
            (def m16 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} local_max c16_i32 c32_i32))
            (def max16 (arith.maximumf local_max m16))
            (def m8 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} max16 c8_i32 c32_i32))
            (def max8 (arith.maximumf max16 m8))
            (def m4 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} max8 c4_i32 c32_i32))
            (def max4 (arith.maximumf max8 m4))
            (def m2 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} max4 c2_i32 c32_i32))
            (def max2 (arith.maximumf max4 m2))
            (def m1 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} max2 c1_i32 c32_i32))
            (def global_max (arith.maximumf max2 m1))

            ;; Step 3: Each thread computes local sum of exp(x - max)
            (def local_sum (scf.for {:result f32} c0 c2 c1 zero
              (region
                (block [(: i index) (: s f32)]
                  (def offset (arith.muli i c32))
                  (def col (arith.addi lane offset))
                  (def x (memref.load {:result f32} inp row col))
                  (def shifted (arith.subf x global_max))
                  (def exp_x (math.exp shifted))
                  (def new_s (arith.addf s exp_x))
                  (scf.yield new_s)))))

            ;; Step 4: Warp reduce sum
            (def s16 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} local_sum c16_i32 c32_i32))
            (def sum16 (arith.addf local_sum s16))
            (def s8 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum16 c8_i32 c32_i32))
            (def sum8 (arith.addf sum16 s8))
            (def s4 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum8 c4_i32 c32_i32))
            (def sum4 (arith.addf sum8 s4))
            (def s2 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum4 c2_i32 c32_i32))
            (def sum2 (arith.addf sum4 s2))
            (def s1 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum2 c1_i32 c32_i32))
            (def total_sum (arith.addf sum2 s1))

            ;; Step 5: Normalize and write output
            (def scale (arith.divf (: 1.0 f32) total_sum))
            (scf.for c0 c2 c1
              (region
                (block [(: i index)]
                  (def offset (arith.muli i c32))
                  (def col (arith.addi lane offset))
                  (def x (memref.load {:result f32} inp row col))
                  (def shifted (arith.subf x global_max))
                  (def exp_x (math.exp shifted))
                  (def softmax_val (arith.mulf exp_x scale))
                  (memref.store softmax_val out row col)
                  (scf.yield))))

            (gpu.terminator))))

      ;; Print first few values from each row
      ;; Expected: 1/64 = 0.015625 for all
      (def result0 (memref.load out c0 c0))
      (func.call {:callee "@printF32"} result0)
      (func.call {:callee "@printNewline"})

      (def result1 (memref.load out c1 c0))
      (func.call {:callee "@printF32"} result1)
      (func.call {:callee "@printNewline"})

      ;; Cleanup
      (gpu.host_unregister inp_unranked)
      (gpu.host_unregister out_unranked)
      (memref.dealloc inp)
      (memref.dealloc out)
      (func.return))))
