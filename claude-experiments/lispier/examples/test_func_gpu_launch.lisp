;; Test gpu.launch inside a function (not main)
;; This mimics the GPT-2 pattern where causal_softmax is a separate function

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

    ;; Separate function with gpu.launch inside (like causal_softmax)
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

          ;; Fused causal softmax kernel
          (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
            num_blocks c1 c1 c32 c1 c1
            (region
              (block [(: block_id index) (: _by index) (: _bz index)
                      (: lane index) (: _ty index) (: _tz index)
                      (: _gridDimX index) (: _gridDimY index) (: _gridDimZ index)
                      (: _blockDimX index) (: _blockDimY index) (: _blockDimZ index)]

                (def head (arith.divui block_id c64))
                (def query (arith.remui block_id c64))

                ;; Step 1: Find local max (with causal mask)
                (def local_max (scf.for {:result f32} c0 c2 c1 neg_inf
                  (region
                    (block [(: i index) (: m f32)]
                      (def offset (arith.muli i c32))
                      (def key_pos (arith.addi lane offset))
                      (def score (memref.load {:result f32} scores head query key_pos))
                      ;; key_pos <= query (ule = predicate 7)
                      (def is_valid (arith.cmpi {:predicate 7} key_pos query))
                      (def masked_score (arith.select is_valid score neg_inf))
                      (def new_m (arith.maximumf m masked_score))
                      (scf.yield new_m)))))

                ;; Warp reduce max
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

                ;; Step 2: Compute local sum of exp(score - max)
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

                ;; Warp reduce sum
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

                ;; Step 3: Normalize and write output
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
      (def num_heads (: 12 index))
      (def seq_len (: 64 index))
      (def c0 (: 0 index))
      (def c1 (: 1 index))
      (def c2 (: 2 index))

      ;; Allocate
      (def scores (memref.alloc {:result memref<12x64x64xf32>}))
      (def weights (memref.alloc {:result memref<12x64x64xf32>}))

      ;; Initialize scores with zeros (so all positions have same value)
      (def zero (: 0.0 f32))
      (linalg.fill {:ins 1 :outs 1} zero scores
        (region
          (block [(: in f32) (: _out f32)]
            (linalg.yield in))))

      ;; Register for GPU
      (def scores_unranked (memref.cast {:result "memref<*xf32>"} scores))
      (def weights_unranked (memref.cast {:result "memref<*xf32>"} weights))
      (gpu.host_register scores_unranked)
      (gpu.host_register weights_unranked)

      ;; Call the warp_softmax function
      (func.call {:callee "@warp_softmax"} weights scores)

      ;; Print results
      ;; Row 0: only position 0 valid -> 1.0
      (func.call {:callee "@printF32"} (memref.load weights c0 c0 c0))
      (func.call {:callee "@printNewline"})

      ;; Row 1: positions 0,1 valid -> 0.5, 0.5
      (func.call {:callee "@printF32"} (memref.load weights c0 c1 c0))
      (func.call {:callee "@printNewline"})
      (func.call {:callee "@printF32"} (memref.load weights c0 c1 c1))
      (func.call {:callee "@printNewline"})

      ;; Row 63: all 64 positions valid -> 1/64 = 0.015625
      (def c63 (: 63 index))
      (func.call {:callee "@printF32"} (memref.load weights c0 c63 c0))
      (func.call {:callee "@printNewline"})

      ;; Cleanup
      (gpu.host_unregister scores_unranked)
      (gpu.host_unregister weights_unranked)
      (memref.dealloc scores)
      (memref.dealloc weights)
      (func.return))))