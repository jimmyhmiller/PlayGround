;; Test the exact GPT-2 pattern: batch_matmul -> warp_softmax -> batch_matmul
;; This tests if there's an issue with linalg-derived kernel followed by gpu.launch

(require-dialect memref)
(require-dialect arith)
(require-dialect func)
(require-dialect gpu)
(require-dialect scf)
(require-dialect math)
(require-dialect linalg)

;; Use GPT-2's compilation pipeline
(compilation
  (target rocm
    (pass convert-linalg-to-parallel-loops)
    (pass scf-parallel-loop-tiling {:parallel-loop-tile-sizes "16,16"})
    (pass gpu-map-parallel-loops)
    (pass convert-parallel-loops-to-gpu)
    (pass lower-affine)
    (pass convert-scf-to-cf)
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl)
    (pass gpu-module-to-binary)
    (pass gpu-to-llvm)
    (pass expand-strided-metadata)
    (pass lower-affine)
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-index-to-llvm)
    (pass convert-math-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"}
      (region))

    ;; batched_qk_matmul like GPT-2: scores = Q @ K_t
    (func.func {:sym_name "batched_qk_matmul"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: out memref<12x64x64xf32>)
                (: Q memref<12x64x64xf32>)
                (: K_t memref<12x64x64xf32>)]

          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))

          (linalg.batch_matmul {:ins 2 :outs 1} Q K_t out
            (region
              (block [(: q_val f32) (: k_val f32) (: accum f32)]
                (def prod (arith.mulf q_val k_val))
                (def sum (arith.addf accum prod))
                (linalg.yield sum))))

          (func.return))))

    ;; warp_softmax like we want to integrate into GPT-2
    (func.func {:sym_name "warp_softmax"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: weights memref<12x64x64xf32>)
                (: scores memref<12x64x64xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c2 (: 2 index))
          (def c32 (: 32 index))
          (def c64 (: 64 index))
          (def num_blocks (: 768 index))
          (def zero (: 0.0 f32))
          (def one (: 1.0 f32))
          (def neg_inf (: -1e30 f32))

          (def c16_i32 (: 16 i32))
          (def c8_i32 (: 8 i32))
          (def c4_i32 (: 4 i32))
          (def c2_i32 (: 2 i32))
          (def c1_i32 (: 1 i32))
          (def c32_i32 (: 32 i32))

          (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
            num_blocks c1 c1 c32 c1 c1
            (region
              (block [(: block_id index) (: _by index) (: _bz index)
                      (: lane index) (: _ty index) (: _tz index)
                      (: _gridDimX index) (: _gridDimY index) (: _gridDimZ index)
                      (: _blockDimX index) (: _blockDimY index) (: _blockDimZ index)]

                (def head (arith.divui block_id c64))
                (def query (arith.remui block_id c64))

                (def local_max (scf.for {:result f32} c0 c2 c1 neg_inf
                  (region
                    (block [(: i index) (: m f32)]
                      (def offset (arith.muli i c32))
                      (def key_pos (arith.addi lane offset))
                      (def score (memref.load {:result f32} scores head query key_pos))
                      (def is_valid (arith.cmpi {:predicate 7} key_pos query))
                      (def masked_score (arith.select is_valid score neg_inf))
                      (def new_m (arith.maximumf m masked_score))
                      (scf.yield new_m)))))

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

                (def local_sum (scf.for {:result f32} c0 c2 c1 zero
                  (region
                    (block [(: i index) (: s f32)]
                      (def offset (arith.muli i c32))
                      (def key_pos (arith.addi lane offset))
                      (def score (memref.load {:result f32} scores head query key_pos))
                      (def is_valid (arith.cmpi {:predicate 7} key_pos query))
                      (def masked_score (arith.select is_valid score neg_inf))
                      (def shifted (arith.subf masked_score global_max))
                      (def exp_val (math.exp shifted))
                      (def new_s (arith.addf s exp_val))
                      (scf.yield new_s)))))

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

                (def scale (arith.divf one total_sum))
                (scf.for c0 c2 c1
                  (region
                    (block [(: i index)]
                      (def offset (arith.muli i c32))
                      (def key_pos (arith.addi lane offset))
                      (def score (memref.load {:result f32} scores head query key_pos))
                      (def is_valid (arith.cmpi {:predicate 7} key_pos query))
                      (def masked_score (arith.select is_valid score neg_inf))
                      (def shifted (arith.subf masked_score global_max))
                      (def exp_val (math.exp shifted))
                      (def softmax_val (arith.mulf exp_val scale))
                      (memref.store softmax_val weights head query key_pos)
                      (scf.yield))))

                (gpu.terminator))))

          (func.return))))

    (defn main []
      (def c0 (: 0 index))
      (def c1 (: 1 index))
      (def c2 (: 2 index))
      (def c63 (: 63 index))

      ;; Allocate like GPT-2
      (def Q (memref.alloc {:result memref<12x64x64xf32>}))
      (def K_t (memref.alloc {:result memref<12x64x64xf32>}))
      (def scores (memref.alloc {:result memref<12x64x64xf32>}))
      (def weights (memref.alloc {:result memref<12x64x64xf32>}))

      ;; Initialize Q and K_t with 1.0 and 0.015625
      ;; So scores = Q @ K_t will have all elements = 64 * 1.0 * 0.015625 = 1.0
      (def one (: 1.0 f32))
      (def scale (: 0.015625 f32))  ;; 1/64
      (linalg.fill {:ins 1 :outs 1} one Q
        (region
          (block [(: in f32) (: _out f32)]
            (linalg.yield in))))
      (linalg.fill {:ins 1 :outs 1} scale K_t
        (region
          (block [(: in f32) (: _out f32)]
            (linalg.yield in))))

      ;; Register for GPU
      (def Q_unranked (memref.cast {:result "memref<*xf32>"} Q))
      (def K_t_unranked (memref.cast {:result "memref<*xf32>"} K_t))
      (def scores_unranked (memref.cast {:result "memref<*xf32>"} scores))
      (def weights_unranked (memref.cast {:result "memref<*xf32>"} weights))
      (gpu.host_register Q_unranked)
      (gpu.host_register K_t_unranked)
      (gpu.host_register scores_unranked)
      (gpu.host_register weights_unranked)

      ;; Step 1: batched_qk_matmul (linalg-based)
      (func.call {:callee "@batched_qk_matmul"} scores Q K_t)

      ;; Debug: print scores value to verify matmul worked
      (func.call {:callee "@printF32"} (memref.load scores c0 c0 c0))
      (func.call {:callee "@printNewline"})

      ;; Step 2: warp_softmax (gpu.launch-based)
      (func.call {:callee "@warp_softmax"} weights scores)

      ;; Print results
      ;; Row 0: only position 0 valid -> 1.0
      (func.call {:callee "@printF32"} (memref.load weights c0 c0 c0))
      (func.call {:callee "@printNewline"})

      ;; Row 1: positions 0,1 valid, equal scores -> 0.5, 0.5
      (func.call {:callee "@printF32"} (memref.load weights c0 c1 c0))
      (func.call {:callee "@printNewline"})
      (func.call {:callee "@printF32"} (memref.load weights c0 c1 c1))
      (func.call {:callee "@printNewline"})

      ;; Row 63: all 64 positions valid -> 1/64 = 0.015625
      (func.call {:callee "@printF32"} (memref.load weights c0 c63 c0))
      (func.call {:callee "@printNewline"})

      ;; Cleanup
      (gpu.host_unregister Q_unranked)
      (gpu.host_unregister K_t_unranked)
      (gpu.host_unregister scores_unranked)
      (gpu.host_unregister weights_unranked)
      (memref.dealloc Q)
      (memref.dealloc K_t)
      (memref.dealloc scores)
      (memref.dealloc weights)
      (func.return))))