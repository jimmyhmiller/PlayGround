;; GPT-2 Text Generation with Linalg Operations for ROCm GPU
;;
;; This file combines:
;; - Linalg operations from gpt2_forward_linalg.lisp for GPU acceleration
;; - Weight loading and generation loop from gpt2_generate.lisp
;;
;; Architecture: GPT-2 Small (124M)
;; - T=64 (sequence length), C=768 (channels)
;; - V=50257 (vocab), L=12 (layers), NH=12 (heads)

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect linalg)
(require-dialect scf)
(require-dialect math)
(require-dialect llvm)

;; NOTE: Using func.func external declarations instead of extern-fn for non-vararg functions
;; because extern-fn + GPU pipeline causes llvm-request-c-wrappers to crash
;; For printf (vararg), we use extern-fn which generates llvm.func (not affected by the pass)
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

;; Compilation pipeline for AMD GPU via linalg -> parallel loops -> GPU
;;
;; IMPORTANT: The order of sections matters! Module passes are split around
;; gpu-passes because convert-gpu-to-rocdl must run BEFORE gpu-module-to-binary.
;; Func-scoped passes must run AFTER convert-linalg-to-parallel-loops creates the loops.
(compilation-gpu rocm
  ;; First create parallel loops from linalg ops
  (passes
    (pass convert-linalg-to-parallel-loops)
    (pass linalg-fuse-elementwise-ops))

  ;; Function-scoped passes (run inside func.func on the parallel loops)
  (func-passes
    ;; Tile parallel loops to create block/thread structure (16x16 = 256 threads per block)
    (pass scf-parallel-loop-tiling {:parallel-loop-tile-sizes "16,16"})
    (pass gpu-map-parallel-loops))

  ;; Module-level passes BEFORE GPU module lowering
  (passes
    ;; Convert to GPU ops
    (pass convert-parallel-loops-to-gpu)
    ;; Lower affine constructs
    (pass lower-affine)
    ;; CRITICAL: SCF to CF BEFORE GPU outlining
    (pass convert-scf-to-cf)
    ;; GPU lowering
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target))

  ;; GPU module passes (run inside gpu.module)
  ;; MUST come before gpu-module-to-binary
  (gpu-passes
    (pass convert-gpu-to-rocdl))

  ;; Module-level passes AFTER GPU module lowering
  (passes
    (pass gpu-module-to-binary)
    ;; Host-side LLVM lowering (no bare pointers for strided memref support)
    (pass gpu-to-llvm)
    ;; expand-strided-metadata BEFORE LLVM conversions (generates arith/affine ops)
    (pass expand-strided-metadata)
    ;; lower-affine again because expand-strided-metadata generates affine.apply
    (pass lower-affine)
    ;; Now convert everything to LLVM
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-index-to-llvm)
    (pass convert-math-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; =========================================================================
    ;; External function declarations (using func.func instead of extern-fn)
    ;; =========================================================================
    (func.func {:sym_name "malloc"
                :function_type (-> [i64] [!llvm.ptr])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "free"
                :function_type (-> [!llvm.ptr] [])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "fopen"
                :function_type (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "fread"
                :function_type (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "fseek"
                :function_type (-> [!llvm.ptr i64 i32] [i32])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "fclose"
                :function_type (-> [!llvm.ptr] [i32])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "clock_ms"
                :function_type (-> [] [i64])
                :sym_visibility "private"}
      (region))

    ;; printf is variadic - use extern-fn which generates llvm.func (not affected by llvm-request-c-wrappers)
    ;; This is declared outside module using extern-fn macro

    ;; =========================================================================
    ;; Global paths and state
    ;; =========================================================================
    (llvm.mlir.global {:sym_name "checkpoint_path"
                       :linkage "#llvm.linkage<external>"
                       :global_type !llvm.array<39 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_124M.bin\0" :result !llvm.array<39 x i8>}))
          (llvm.return s))))

    (llvm.mlir.global {:sym_name "tokenizer_path"
                       :linkage "#llvm.linkage<external>"
                       :global_type !llvm.array<44 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_tokenizer.bin\0" :result !llvm.array<44 x i8>}))
          (llvm.return s))))

    (llvm.mlir.global {:sym_name "read_mode"
                       :linkage "#llvm.linkage<external>"
                       :global_type !llvm.array<3 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "rb\0" :result !llvm.array<3 x i8>}))
          (llvm.return s))))

    (llvm.mlir.global {:sym_name "g_params"
                       :linkage "#llvm.linkage<external>"
                       :global_type !llvm.ptr
                       :constant false}
      (region
        (block []
          (def null (llvm.mlir.zero {:result !llvm.ptr}))
          (llvm.return null))))

    (llvm.mlir.global {:sym_name "token_table"
                       :linkage "#llvm.linkage<external>"
                       :global_type !llvm.ptr
                       :constant false}
      (region
        (block []
          (def null (llvm.mlir.zero {:result !llvm.ptr}))
          (llvm.return null))))

    (llvm.mlir.global {:sym_name "eot_token"
                       :linkage "#llvm.linkage<external>"
                       :global_type i32
                       :constant false}
      (region
        (block []
          (def v (llvm.mlir.constant {:value 50256 :result i32}))
          (llvm.return v))))

    ;; =========================================================================
    ;; QKV Projection: (T,C) @ (C,3C) + bias -> (T,3C)
    ;; BF16 weights with F32 accumulation for reduced memory bandwidth
    ;; =========================================================================
    (func.func {:sym_name "matmul_qkv"
                :function_type (-> [memref<64x2304xf32>
                                    memref<64x768xf32>
                                    "memref<768x2304xbf16, strided<[2304, 1], offset: ?>>"
                                    "memref<2304xf32, strided<[1], offset: ?>>"] [])}
      (region
        (block [(: out memref<64x2304xf32>)
                (: inp memref<64x768xf32>)
                (: weight "memref<768x2304xbf16, strided<[2304, 1], offset: ?>>")
                (: bias "memref<2304xf32, strided<[1], offset: ?>>")]

          ;; Zero output for accumulation
          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))

          ;; Core matmul with bf16 weights: (64,768) @ (768,2304) -> (64,2304)
          ;; Load bf16, extend to f32, multiply-accumulate in f32
          (linalg.generic
            {:ins 2 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2)->(d0,d2)>
                             affine_map<(d0,d1,d2)->(d2,d1)>
                             affine_map<(d0,d1,d2)->(d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<reduction>"]}
            inp weight out
            (region
              (block [(: a f32) (: b bf16) (: c f32)]
                (def b_f32 (arith.extf {:result f32} b))
                (def mul (arith.mulf a b_f32))
                (def sum (arith.addf c mul))
                (linalg.yield sum))))

          ;; Add bias using linalg.generic (GPU-parallelized)
          (linalg.generic
            {:ins 2 :outs 1
             :indexing_maps [affine_map<(d0,d1)->(d0,d1)>
                             affine_map<(d0,d1)->(d1)>
                             affine_map<(d0,d1)->(d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            out bias out
            (region
              (block [(: val f32) (: b f32) (: _acc f32)]
                (def result (arith.addf val b))
                (linalg.yield result))))

          (func.return))))

    ;; =========================================================================
    ;; Attention Output Projection: (T,C) @ (C,C) + bias
    ;; BF16 weights with F32 accumulation
    ;; =========================================================================
    (func.func {:sym_name "matmul_attn_proj"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x768xf32>
                                    "memref<768x768xbf16, strided<[768, 1], offset: ?>>"
                                    "memref<768xf32, strided<[1], offset: ?>>"] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x768xf32>)
                (: weight "memref<768x768xbf16, strided<[768, 1], offset: ?>>")
                (: bias "memref<768xf32, strided<[1], offset: ?>>")]

          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))

          ;; Core matmul with bf16 weights
          (linalg.generic
            {:ins 2 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2)->(d0,d2)>
                             affine_map<(d0,d1,d2)->(d2,d1)>
                             affine_map<(d0,d1,d2)->(d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<reduction>"]}
            inp weight out
            (region
              (block [(: a f32) (: b bf16) (: c f32)]
                (def b_f32 (arith.extf {:result f32} b))
                (def mul (arith.mulf a b_f32))
                (def sum (arith.addf c mul))
                (linalg.yield sum))))

          ;; Add bias using linalg.generic (GPU-parallelized)
          (linalg.generic
            {:ins 2 :outs 1
             :indexing_maps [affine_map<(d0,d1)->(d0,d1)>
                             affine_map<(d0,d1)->(d1)>
                             affine_map<(d0,d1)->(d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            out bias out
            (region
              (block [(: val f32) (: b f32) (: _acc f32)]
                (def result (arith.addf val b))
                (linalg.yield result))))

          (func.return))))

    ;; =========================================================================
    ;; MLP FC1: (T,C) @ (C,4C) + bias -> (T,4C)
    ;; BF16 weights with F32 accumulation
    ;; =========================================================================
    (func.func {:sym_name "matmul_fc"
                :function_type (-> [memref<64x3072xf32>
                                    memref<64x768xf32>
                                    "memref<768x3072xbf16, strided<[3072, 1], offset: ?>>"
                                    "memref<3072xf32, strided<[1], offset: ?>>"] [])}
      (region
        (block [(: out memref<64x3072xf32>)
                (: inp memref<64x768xf32>)
                (: weight "memref<768x3072xbf16, strided<[3072, 1], offset: ?>>")
                (: bias "memref<3072xf32, strided<[1], offset: ?>>")]

          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))

          ;; Core matmul with bf16 weights
          (linalg.generic
            {:ins 2 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2)->(d0,d2)>
                             affine_map<(d0,d1,d2)->(d2,d1)>
                             affine_map<(d0,d1,d2)->(d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<reduction>"]}
            inp weight out
            (region
              (block [(: a f32) (: b bf16) (: c f32)]
                (def b_f32 (arith.extf {:result f32} b))
                (def mul (arith.mulf a b_f32))
                (def sum (arith.addf c mul))
                (linalg.yield sum))))

          ;; Add bias using linalg.generic (GPU-parallelized)
          (linalg.generic
            {:ins 2 :outs 1
             :indexing_maps [affine_map<(d0,d1)->(d0,d1)>
                             affine_map<(d0,d1)->(d1)>
                             affine_map<(d0,d1)->(d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            out bias out
            (region
              (block [(: val f32) (: b f32) (: _acc f32)]
                (def result (arith.addf val b))
                (linalg.yield result))))

          (func.return))))

    ;; =========================================================================
    ;; MLP Projection: (T,4C) @ (4C,C) + bias -> (T,C)
    ;; BF16 weights with F32 accumulation
    ;; =========================================================================
    (func.func {:sym_name "matmul_fc_proj"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x3072xf32>
                                    "memref<3072x768xbf16, strided<[768, 1], offset: ?>>"
                                    "memref<768xf32, strided<[1], offset: ?>>"] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x3072xf32>)
                (: weight "memref<3072x768xbf16, strided<[768, 1], offset: ?>>")
                (: bias "memref<768xf32, strided<[1], offset: ?>>")]

          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))

          ;; Core matmul with bf16 weights
          (linalg.generic
            {:ins 2 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2)->(d0,d2)>
                             affine_map<(d0,d1,d2)->(d2,d1)>
                             affine_map<(d0,d1,d2)->(d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<reduction>"]}
            inp weight out
            (region
              (block [(: a f32) (: b bf16) (: c f32)]
                (def b_f32 (arith.extf {:result f32} b))
                (def mul (arith.mulf a b_f32))
                (def sum (arith.addf c mul))
                (linalg.yield sum))))

          ;; Add bias using linalg.generic (GPU-parallelized)
          (linalg.generic
            {:ins 2 :outs 1
             :indexing_maps [affine_map<(d0,d1)->(d0,d1)>
                             affine_map<(d0,d1)->(d1)>
                             affine_map<(d0,d1)->(d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            out bias out
            (region
              (block [(: val f32) (: b f32) (: _acc f32)]
                (def result (arith.addf val b))
                (linalg.yield result))))

          (func.return))))

    ;; =========================================================================
    ;; Residual Add using linalg.add
    ;; =========================================================================
    (func.func {:sym_name "residual_forward"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x768xf32>
                                    memref<64x768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: inp1 memref<64x768xf32>)
                (: inp2 memref<64x768xf32>)]
          (linalg.add {:ins 2 :outs 1} inp1 inp2 out
            (region
              (block [(: x f32) (: y f32) (: _out f32)]
                (def sum (arith.addf x y))
                (linalg.yield sum))))
          (func.return))))

    ;; =========================================================================
    ;; Copy buffer using linalg.copy
    ;; =========================================================================
    (func.func {:sym_name "copy_buffer"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x768xf32>)]
          (linalg.copy {:ins 1 :outs 1} inp out
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))
          (func.return))))

    ;; =========================================================================
    ;; Token Embedding Lookup (keep as scf.for - index-based gather)
    ;; =========================================================================
    (func.func {:sym_name "embedding_lookup"
                :function_type (-> [memref<64x768xf32>
                                    memref<50257x768xf32>
                                    memref<1024x768xf32>
                                    memref<64xi32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: wte memref<50257x768xf32>)
                (: wpe memref<1024x768xf32>)
                (: tokens memref<64xi32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))

          ;; GPU parallel over all (t, c) pairs: 64*768 = 49,152 threads
          ;; operandSegmentSizes: [2 lowerBounds, 2 upperBounds, 2 steps, 0 initVals]
          (scf.parallel {:operandSegmentSizes "array<i32: 2, 2, 2, 0>"} [c0 c0] [T C] [c1 c1]
            (region
              (block [(: t index) (: c index)]
                ;; Load token_id for this position (redundant loads, but GPU handles this well)
                (def token_id_i32 (memref.load {:result i32} tokens t))
                (def token_idx (arith.index_cast {:result index} token_id_i32))

                ;; Get token embedding: wte[token_id, c]
                (def wte_val (memref.load {:result f32} wte token_idx c))

                ;; Get position embedding: wpe[t, c]
                (def wpe_val (memref.load {:result f32} wpe t c))

                ;; Sum and store
                (def sum (arith.addf wte_val wpe_val))
                (memref.store sum out t c)
                (scf.reduce))))
          (func.return))))

    ;; =========================================================================
    ;; LayerNorm with Fused GPU Warp Reductions
    ;; Single kernel: warp reduction for mean/variance + apply normalization
    ;; =========================================================================
    (func.func {:sym_name "layernorm_forward"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x768xf32>
                                    "memref<768xf32, strided<[1], offset: ?>>"
                                    "memref<768xf32, strided<[1], offset: ?>>"
                                    memref<64xf32>
                                    memref<64xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x768xf32>)
                (: weight "memref<768xf32, strided<[1], offset: ?>>")
                (: bias "memref<768xf32, strided<[1], offset: ?>>")
                (: mean_buf memref<64xf32>)
                (: rs_buf memref<64xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c24 (: 24 index))
          (def T (: 64 index))
          (def c32 (: 32 index))
          (def C_f32 (: 768.0 f32))
          (def zero (: 0.0 f32))
          (def eps (: 1e-5 f32))
          (def one (: 1.0 f32))

          ;; Shuffle constants
          (def c16_i32 (: 16 i32))
          (def c8_i32 (: 8 i32))
          (def c4_i32 (: 4 i32))
          (def c2_i32 (: 2 i32))
          (def c1_i32 (: 1 i32))
          (def c32_i32 (: 32 i32))

          ;; Fused GPU kernel: reduction + normalization in one pass
          ;; Launch 64 blocks (one per row) x 32 threads (one warp)
          ;; Each thread handles 768/32 = 24 elements
          (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
            T c1 c1 c32 c1 c1
            (region
              (block [(: row index) (: _by index) (: _bz index)
                      (: lane index) (: _ty index) (: _tz index)
                      (: _gridDimX index) (: _gridDimY index) (: _gridDimZ index)
                      (: _blockDimX index) (: _blockDimY index) (: _blockDimZ index)]

                ;; Step 1: Each thread sums its 24 elements
                (def partial_sum (scf.for {:result f32} c0 c24 c1 zero
                  (region
                    (block [(: i index) (: acc f32)]
                      (def offset (arith.muli i c32))
                      (def col (arith.addi lane offset))
                      (def x (memref.load {:result f32} inp row col))
                      (def new_acc (arith.addf acc x))
                      (scf.yield new_acc)))))

                ;; Step 2: Warp reduce sum (5 XOR shuffles)
                (def s16 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} partial_sum c16_i32 c32_i32))
                (def sum16 (arith.addf partial_sum s16))
                (def s8 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum16 c8_i32 c32_i32))
                (def sum8 (arith.addf sum16 s8))
                (def s4 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum8 c4_i32 c32_i32))
                (def sum4 (arith.addf sum8 s4))
                (def s2 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum4 c2_i32 c32_i32))
                (def sum2 (arith.addf sum4 s2))
                (def s1 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum2 c1_i32 c32_i32))
                (def row_sum (arith.addf sum2 s1))
                (def mean (arith.divf row_sum C_f32))

                ;; Step 3: Each thread computes variance for its 24 elements
                (def partial_var (scf.for {:result f32} c0 c24 c1 zero
                  (region
                    (block [(: i index) (: vacc f32)]
                      (def offset (arith.muli i c32))
                      (def col (arith.addi lane offset))
                      (def x (memref.load {:result f32} inp row col))
                      (def diff (arith.subf x mean))
                      (def diff_sq (arith.mulf diff diff))
                      (def new_vacc (arith.addf vacc diff_sq))
                      (scf.yield new_vacc)))))

                ;; Step 4: Warp reduce variance
                (def v16 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} partial_var c16_i32 c32_i32))
                (def vsum16 (arith.addf partial_var v16))
                (def v8 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} vsum16 c8_i32 c32_i32))
                (def vsum8 (arith.addf vsum16 v8))
                (def v4 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} vsum8 c4_i32 c32_i32))
                (def vsum4 (arith.addf vsum8 v4))
                (def v2 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} vsum4 c2_i32 c32_i32))
                (def vsum2 (arith.addf vsum4 v2))
                (def v1 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} vsum2 c1_i32 c32_i32))
                (def var_sum (arith.addf vsum2 v1))

                ;; Step 5: Compute reciprocal std
                (def variance (arith.divf var_sum C_f32))
                (def var_eps (arith.addf variance eps))
                (def std (math.sqrt var_eps))
                (def rs (arith.divf one std))

                ;; Step 6: Apply normalization and write output
                ;; Each thread normalizes its 24 elements: out = (x - mean) * rs * w + b
                (scf.for c0 c24 c1
                  (region
                    (block [(: i index)]
                      (def offset (arith.muli i c32))
                      (def col (arith.addi lane offset))
                      (def x (memref.load {:result f32} inp row col))
                      (def w (memref.load {:result f32} weight col))
                      (def b (memref.load {:result f32} bias col))
                      (def x_centered (arith.subf x mean))
                      (def x_norm (arith.mulf x_centered rs))
                      (def scaled (arith.mulf x_norm w))
                      (def result (arith.addf scaled b))
                      (memref.store result out row col)
                      (scf.yield))))

                ;; Store mean/rs for debugging (optional, can remove later)
                (def lane_i32 (arith.index_cast {:result i32} lane))
                (def zero_i32 (: 0 i32))
                (def is_lane0 (arith.cmpi {:predicate 0} lane_i32 zero_i32))
                (scf.if is_lane0
                  (region
                    (block []
                      (memref.store mean mean_buf row)
                      (memref.store rs rs_buf row)
                      (scf.yield)))
                  (region
                    (block []
                      (scf.yield))))

                (gpu.terminator))))

          (func.return))))

    ;; =========================================================================
    ;; Reshape QKV from (T=64, 3C=2304) to separate Q/K/V as (NH=12, T=64, hs=64)
    ;; GPU-parallel via linalg.generic with affine indexing
    ;; =========================================================================
    (func.func {:sym_name "reshape_qkv_to_batched"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<64x2304xf32>] [])}
      (region
        (block [(: Q_out memref<12x64x64xf32>)
                (: K_out memref<12x64x64xf32>)
                (: V_out memref<12x64x64xf32>)
                (: qkv memref<64x2304xf32>)]

          ;; Q[h,t,i] = qkv[t, h*64 + i] (offset 0-767)
          ;; Input reads from (d1, d0*64 + d2), output writes to (d0,d1,d2)
          (linalg.generic
            {:ins 1 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2) -> (d1, d0*64 + d2)>
                             affine_map<(d0,d1,d2) -> (d0,d1,d2)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            qkv Q_out
            (region
              (block [(: val f32) (: _out f32)]
                (linalg.yield val))))

          ;; K[h,t,i] = qkv[t, 768 + h*64 + i] (offset 768-1535)
          (linalg.generic
            {:ins 1 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2) -> (d1, 768 + d0*64 + d2)>
                             affine_map<(d0,d1,d2) -> (d0,d1,d2)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            qkv K_out
            (region
              (block [(: val f32) (: _out f32)]
                (linalg.yield val))))

          ;; V[h,t,i] = qkv[t, 1536 + h*64 + i] (offset 1536-2303)
          (linalg.generic
            {:ins 1 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2) -> (d1, 1536 + d0*64 + d2)>
                             affine_map<(d0,d1,d2) -> (d0,d1,d2)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            qkv V_out
            (region
              (block [(: val f32) (: _out f32)]
                (linalg.yield val))))

          (func.return))))

    ;; =========================================================================
    ;; Transpose K from (NH, T, hs) to (NH, hs, T) for K^T operation
    ;; GPU-parallel via linalg.generic with swapped indexing maps
    ;; =========================================================================
    (func.func {:sym_name "transpose_k_for_attention"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: K_t memref<12x64x64xf32>)
                (: K memref<12x64x64xf32>)]

          ;; GPU-parallel transpose: K_t[h, i, t] = K[h, t, i]
          ;; Input indexing swaps d1 and d2 to read from transposed position
          (linalg.generic
            {:ins 1 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2) -> (d0,d2,d1)>
                             affine_map<(d0,d1,d2) -> (d0,d1,d2)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            K K_t
            (region
              (block [(: val f32) (: _out f32)]
                (linalg.yield val))))

          (func.return))))

    ;; =========================================================================
    ;; Batched Q @ K^T using linalg.batch_matmul + scaling
    ;; Q: (12, 64, 64), K_t: (12, 64, 64) -> scores: (12, 64, 64)
    ;; =========================================================================
    (func.func {:sym_name "batched_qk_matmul"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: scores memref<12x64x64xf32>)
                (: Q memref<12x64x64xf32>)
                (: K_t memref<12x64x64xf32>)]

          ;; Zero output for accumulation
          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero scores
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))

          ;; Batched matmul: for each batch h, compute scores[h] = Q[h] @ K_t[h]
          (linalg.batch_matmul {:ins 2 :outs 1} Q K_t scores
            (region
              (block [(: a f32) (: b f32) (: c f32)]
                (def mul (arith.mulf a b))
                (def sum (arith.addf c mul))
                (linalg.yield sum))))

          ;; Apply scaling (1/sqrt(64) = 0.125) using GPU-parallel linalg.generic
          (def scale (: 0.125 f32))

          (linalg.generic
            {:ins 1 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2) -> (d0,d1,d2)>
                             affine_map<(d0,d1,d2) -> (d0,d1,d2)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            scores scores
            (region
              (block [(: val f32) (: _out f32)]
                (def scaled (arith.mulf val scale))
                (linalg.yield scaled))))

          (func.return))))

    ;; =========================================================================
    ;; Causal Softmax: Apply causal mask and softmax to attention scores
    ;; scores: (12, 64, 64), weights_out: (12, 64, 64)
    ;; FUSED WARP-BASED VERSION using gpu.launch with warp shuffles
    ;; Each warp (32 threads) processes one (head, query) pair
    ;; Each thread handles 2 key positions (64/32 = 2)
    ;; =========================================================================
    (func.func {:sym_name "causal_softmax"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64xf32>
                                    memref<12x64xf32>] [])}
      (region
        (block [(: weights memref<12x64x64xf32>)
                (: scores memref<12x64x64xf32>)
                (: _masked_scores memref<12x64x64xf32>)
                (: _max_buf memref<12x64xf32>)
                (: _sum_buf memref<12x64xf32>)]

          ;; Constants
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c2 (: 2 index))
          (def c32 (: 32 index))
          (def c64 (: 64 index))
          (def num_blocks (: 768 index))
          (def zero (: 0.0 f32))
          (def one (: 1.0 f32))
          (def neg_inf (: -1e30 f32))

          ;; Shuffle constants
          (def c16_i32 (: 16 i32))
          (def c8_i32 (: 8 i32))
          (def c4_i32 (: 4 i32))
          (def c2_i32 (: 2 i32))
          (def c1_i32 (: 1 i32))
          (def c32_i32 (: 32 i32))

          ;; Launch 768 blocks (12 heads x 64 queries), 32 threads each
          (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
            num_blocks c1 c1 c32 c1 c1
            (region
              (block [(: block_id index) (: _by index) (: _bz index)
                      (: lane index) (: _ty index) (: _tz index)
                      (: _gridDimX index) (: _gridDimY index) (: _gridDimZ index)
                      (: _blockDimX index) (: _blockDimY index) (: _blockDimZ index)]

                ;; Decompose block_id into head and query indices
                (def head (arith.divui block_id c64))
                (def query (arith.remui block_id c64))

                ;; Step 1: Find local max (with causal mask applied inline)
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

    ;; =========================================================================
    ;; Batched Attention @ V using linalg.batch_matmul
    ;; weights: (12, 64, 64), V: (12, 64, 64) -> out: (12, 64, 64)
    ;; =========================================================================
    (func.func {:sym_name "batched_attn_v_matmul"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: out memref<12x64x64xf32>)
                (: weights memref<12x64x64xf32>)
                (: V memref<12x64x64xf32>)]

          ;; Zero output for accumulation
          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))

          ;; Batched matmul: out[h] = weights[h] @ V[h]
          (linalg.batch_matmul {:ins 2 :outs 1} weights V out
            (region
              (block [(: a f32) (: b f32) (: c f32)]
                (def mul (arith.mulf a b))
                (def sum (arith.addf c mul))
                (linalg.yield sum))))

          (func.return))))

    ;; =========================================================================
    ;; Reshape attention output from (NH=12, T=64, hs=64) back to (T=64, C=768)
    ;; GPU-parallel via linalg.generic with affine indexing
    ;; =========================================================================
    (func.func {:sym_name "reshape_attn_output"
                :function_type (-> [memref<64x768xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: attn_values memref<12x64x64xf32>)]

          ;; out[t, h*64 + i] = attn_values[h, t, i]
          ;; Iterating over (h, t, i) = (d0, d1, d2)
          ;; Input reads from (d0, d1, d2), output writes to (d1, d0*64 + d2)
          (linalg.generic
            {:ins 1 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2) -> (d0,d1,d2)>
                             affine_map<(d0,d1,d2) -> (d1, d0*64 + d2)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            attn_values out
            (region
              (block [(: val f32) (: _out f32)]
                (linalg.yield val))))

          (func.return))))

    ;; =========================================================================
    ;; Attention Forward - Orchestrates batched GPU attention
    ;; =========================================================================
    (func.func {:sym_name "attention_forward"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x2304xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64xf32>
                                    memref<12x64xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: qkv memref<64x2304xf32>)
                (: Q memref<12x64x64xf32>)
                (: K memref<12x64x64xf32>)
                (: V memref<12x64x64xf32>)
                (: K_t memref<12x64x64xf32>)
                (: scores memref<12x64x64xf32>)
                (: weights memref<12x64x64xf32>)
                (: values memref<12x64x64xf32>)
                (: masked_scores memref<12x64x64xf32>)
                (: max_buf memref<12x64xf32>)
                (: sum_buf memref<12x64xf32>)]

          ;; Step 1: Reshape QKV to batched format
          (func.call {:callee "@reshape_qkv_to_batched"} Q K V qkv)

          ;; Step 2: Transpose K for K^T
          (func.call {:callee "@transpose_k_for_attention"} K_t K)

          ;; Step 3: Q @ K^T -> scores (GPU via linalg.batch_matmul)
          (func.call {:callee "@batched_qk_matmul"} scores Q K_t)

          ;; Step 4: Causal softmax on scores -> weights
          (func.call {:callee "@causal_softmax"} weights scores masked_scores max_buf sum_buf)

          ;; Step 5: weights @ V -> values (GPU via linalg.batch_matmul)
          (func.call {:callee "@batched_attn_v_matmul"} values weights V)

          ;; Step 6: Reshape output back to (T, C) format
          (func.call {:callee "@reshape_attn_output"} out values)

          (func.return))))

    ;; =========================================================================
    ;; GELU Activation (GPU-parallel via linalg.generic)
    ;; GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ;; =========================================================================
    (func.func {:sym_name "gelu_forward"
                :function_type (-> [memref<64x3072xf32>
                                    memref<64x3072xf32>] [])}
      (region
        (block [(: out memref<64x3072xf32>)
                (: inp memref<64x3072xf32>)]
          ;; Constants for GELU computation
          (def half (: 0.5 f32))
          (def one (: 1.0 f32))
          (def sqrt_2_over_pi (: 0.7978845608 f32))
          (def coeff (: 0.044715 f32))

          ;; GPU-parallel GELU using linalg.generic
          ;; Use numerically stable tanh via exp(-2|x|) to avoid overflow
          (def two (: 2.0 f32))
          (def neg_two (: -2.0 f32))
          (def neg_one (: -1.0 f32))

          (linalg.generic
            {:ins 1 :outs 1
             :indexing_maps [affine_map<(d0,d1) -> (d0,d1)>
                             affine_map<(d0,d1) -> (d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>" "#linalg.iterator_type<parallel>"]}
            inp out
            (region
              (block [(: x f32) (: _out f32)]
                ;; GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                (def x2 (arith.mulf x x))
                (def x3 (arith.mulf x2 x))
                (def inner1 (arith.mulf coeff x3))
                (def inner2 (arith.addf x inner1))
                (def y (arith.mulf sqrt_2_over_pi inner2))
                ;; Numerically stable tanh:
                ;; tanh(y) = sign(y) * (1 - exp(-2|y|)) / (1 + exp(-2|y|))
                ;; For positive y: (1 - exp(-2y)) / (1 + exp(-2y))
                ;; For negative y: -(1 - exp(2y)) / (1 + exp(2y)) = (exp(2y) - 1) / (exp(2y) + 1)
                (def abs_y (math.absf y))
                (def neg_2_abs_y (arith.mulf neg_two abs_y))
                (def exp_neg (math.exp neg_2_abs_y))
                (def one_minus_exp (arith.subf one exp_neg))
                (def one_plus_exp (arith.addf one exp_neg))
                (def abs_tanh (arith.divf one_minus_exp one_plus_exp))
                ;; Apply sign: copysign(abs_tanh, y)
                (def tanh_val (math.copysign abs_tanh y))
                (def one_plus_tanh (arith.addf one tanh_val))
                (def half_x (arith.mulf half x))
                (def result (arith.mulf half_x one_plus_tanh))
                (linalg.yield result))))
          (func.return))))

    ;; =========================================================================
    ;; Logits Projection using linalg.matmul (only last position for efficiency)
    ;; =========================================================================
    (func.func {:sym_name "logits_forward"
                :function_type (-> [memref<50257xf32>
                                    memref<64x768xf32>
                                    memref<50257x768xf32>
                                    index] [])}
      (region
        (block [(: out memref<50257xf32>)
                (: inp memref<64x768xf32>)
                (: wte memref<50257x768xf32>)
                (: pos index)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def V (: 50257 index))
          (def C (: 768 index))
          (def zero (: 0.0 f32))

          ;; For each vocab token v, compute dot(inp[pos, :], wte[v, :])
          (scf.for c0 V c1
            (region
              (block [(: v index)]
                (def dot (scf.for {:result f32} c0 C c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def inp_val (memref.load {:result f32} inp pos c))
                      (def w_val (memref.load {:result f32} wte v c))
                      (def prod (arith.mulf inp_val w_val))
                      (def new_acc (arith.addf acc prod))
                      (scf.yield new_acc)))))
                (memref.store dot out v)
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Softmax for logits using math.exp
    ;; =========================================================================
    (func.func {:sym_name "softmax_logits"
                :function_type (-> [memref<50257xf32>
                                    memref<50257xf32>] [])}
      (region
        (block [(: probs memref<50257xf32>)
                (: logits memref<50257xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def V (: 50257 index))
          (def zero (: 0.0 f32))
          (def neg_inf (: -10000.0 f32))

          ;; Find max for numerical stability
          (def max_val (scf.for {:result f32} c0 V c1 neg_inf
            (region
              (block [(: v index) (: curr_max f32)]
                (def val (memref.load {:result f32} logits v))
                (def is_greater (arith.cmpf {:predicate 2} val curr_max))
                (def new_max (arith.select is_greater val curr_max))
                (scf.yield new_max)))))

          ;; Exp and sum using math.exp
          (def exp_sum (scf.for {:result f32} c0 V c1 zero
            (region
              (block [(: v index) (: sum_acc f32)]
                (def val (memref.load {:result f32} logits v))
                (def shifted (arith.subf val max_val))
                (def exp_val (math.exp shifted))
                (memref.store exp_val probs v)
                (def new_sum (arith.addf sum_acc exp_val))
                (scf.yield new_sum)))))

          ;; Normalize
          (scf.for c0 V c1
            (region
              (block [(: v index)]
                (def exp_val (memref.load {:result f32} probs v))
                (def normalized (arith.divf exp_val exp_sum))
                (memref.store normalized probs v)
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Argmax
    ;; =========================================================================
    (func.func {:sym_name "argmax"
                :function_type (-> [memref<50257xf32>] [i32])}
      (region
        (block [(: probs memref<50257xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def V (: 50257 index))
          (def neg_inf (: -10000.0 f32))

          ;; Find max value
          (def max_val (scf.for {:result f32} c0 V c1 neg_inf
            (region
              (block [(: v index) (: curr_max f32)]
                (def val (memref.load {:result f32} probs v))
                (def is_greater (arith.cmpf {:predicate 2} val curr_max))
                (def new_max (arith.select is_greater val curr_max))
                (scf.yield new_max)))))

          ;; Find index of max
          (def max_idx (scf.for {:result index} c0 V c1 c0
            (region
              (block [(: v index) (: curr_idx index)]
                (def val (memref.load {:result f32} probs v))
                (def is_max (arith.cmpf {:predicate 1} val max_val))
                (def new_idx (arith.select is_max v curr_idx))
                (scf.yield new_idx)))))

          (def result (arith.index_cast {:result i32} max_idx))
          (func.return result))))

    ;; =========================================================================
    ;; Print token
    ;; =========================================================================
    (func.func {:sym_name "print_token"
                :function_type (-> [i32] [])}
      (region
        (block [(: token_id i32)]
          (def token_table_addr (llvm.mlir.addressof {:global_name @token_table :result !llvm.ptr}))
          (def table_ptr (llvm.load {:result !llvm.ptr} token_table_addr))
          (def token_i64 (arith.extsi {:result i64} token_id))
          (def str_ptr_ptr (ptr-at !llvm.ptr table_ptr token_i64))
          (def str_ptr (llvm.load {:result !llvm.ptr} str_ptr_ptr))
          ;; Use llvm.call directly for variadic printf (can't use func.call with llvm.func)
          ;; operandSegmentSizes is auto-computed by ir_gen.rs
          (def _p (llvm.call {:callee @printf
                              :var_callee_type !llvm.func<i32 (ptr, ...)>
                              :op_bundle_sizes "array<i32>"
                              :result i32}
                    str_ptr))
          (func.return))))

    ;; =========================================================================
    ;; Tokenizer init
    ;; =========================================================================
    (func.func {:sym_name "tokenizer_init"
                :function_type (-> [!llvm.ptr] [i32])}
      (region
        (block [(: path !llvm.ptr)]
          (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
          (def file (func.call {:callee "@fopen" :result !llvm.ptr} path mode))

          ;; Read header (256 * 4 = 1024 bytes)
          (def header_size (: 1024 i64))
          (def header_ptr (func.call {:callee "@malloc" :result !llvm.ptr} header_size))
          (def _read_h (func.call {:callee "@fread" :result i64} header_ptr (: 4 i64) (: 256 i64) file))

          ;; Check magic (should be 20240328)
          (def magic (llvm.load {:result i32} header_ptr))

          ;; Get vocab size from header[2] (header[1] is version)
          (def vocab_ptr (ptr-at i32 header_ptr (: 2 i64)))
          (def vocab_size (llvm.load {:result i32} vocab_ptr))

          ;; Get EOT from header[3]
          (def eot_ptr (ptr-at i32 header_ptr (: 3 i64)))
          (def eot (llvm.load {:result i32} eot_ptr))

          ;; Store EOT globally
          (def eot_global (llvm.mlir.addressof {:global_name @eot_token :result !llvm.ptr}))
          (llvm.store eot eot_global)

          ;; Allocate token table (array of string pointers)
          (def vocab_i64 (arith.extsi {:result i64} vocab_size))
          (def ptr_size (: 8 i64))
          (def table_bytes (arith.muli vocab_i64 ptr_size))
          (def table_ptr (func.call {:callee "@malloc" :result !llvm.ptr} table_bytes))

          ;; Store globally
          (def token_table_addr (llvm.mlir.addressof {:global_name @token_table :result !llvm.ptr}))
          (llvm.store table_ptr token_table_addr)

          ;; Read each token string
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def vocab_idx (arith.index_cast {:result index} vocab_size))
          (def one_byte (: 1 i64))

          (scf.for c0 vocab_idx c1
            (region
              (block [(: i index)]
                ;; Read length (1 byte)
                (def len_buf (func.call {:callee "@malloc" :result !llvm.ptr} one_byte))
                (def _r1 (func.call {:callee "@fread" :result i64} len_buf one_byte one_byte file))
                (def len_u8 (llvm.load {:result i8} len_buf))
                (def len (arith.extui {:result i64} len_u8))
                (func.call {:callee "@free"} len_buf)

                ;; Allocate string buffer (+1 for null)
                (def str_len_plus1 (arith.addi len one_byte))
                (def str_ptr (func.call {:callee "@malloc" :result !llvm.ptr} str_len_plus1))

                ;; Read string
                (def _r2 (func.call {:callee "@fread" :result i64} str_ptr one_byte len file))

                ;; Add null terminator
                (def null_pos (ptr-at i8 str_ptr len))
                (def null_byte (: 0 i8))
                (llvm.store null_byte null_pos)

                ;; Store in table
                (def i_i64 (arith.index_cast {:result i64} i))
                (def slot_ptr (ptr-at !llvm.ptr table_ptr i_i64))
                (llvm.store str_ptr slot_ptr)
                (scf.yield))))

          (def _fc (func.call {:callee "@fclose" :result i32} file))
          (func.call {:callee "@free"} header_ptr)

          (func.return (: 0 i32)))))

    ;; =========================================================================
    ;; External function declarations
    ;; =========================================================================
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"}
      (region))

    ;; =========================================================================
    ;; Main function with weight loading and generation loop
    ;; =========================================================================
    (defn main []
      ;; Load checkpoint
      (def path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
      (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
      (def file (func.call {:callee "@fopen" :result !llvm.ptr} path mode))
    
      ;; Read checkpoint header (256 ints) to get model config
      (def header_buf (func.call {:callee "@malloc" :result !llvm.ptr} (: 1024 i64)))
      (def _read_header (func.call {:callee "@fread" :result i64} header_buf (: 4 i64) (: 256 i64) file))
    
      ;; Extract model config from header
      (def magic_ptr header_buf)
      (def _magic (llvm.load {:result i32} magic_ptr))
    
      (def maxT_ptr (ptr-at i32 header_buf (: 2 i64)))
      (def maxT_i32 (llvm.load {:result i32} maxT_ptr))
      (def maxT (arith.extsi {:result i64} maxT_i32))
    
      (def V_ptr (ptr-at i32 header_buf (: 3 i64)))
      (def V_i32 (llvm.load {:result i32} V_ptr))
      (def V (arith.extsi {:result i64} V_i32))
    
      (def L_ptr (ptr-at i32 header_buf (: 4 i64)))
      (def L_i32 (llvm.load {:result i32} L_ptr))
      (def L (arith.extsi {:result i64} L_i32))
    
      (def NH_ptr (ptr-at i32 header_buf (: 5 i64)))
      (def NH_i32 (llvm.load {:result i32} NH_ptr))
    
      (def C_ptr (ptr-at i32 header_buf (: 6 i64)))
      (def C_i32 (llvm.load {:result i32} C_ptr))
      (def C (arith.extsi {:result i64} C_i32))
    
      (def Vp_ptr (ptr-at i32 header_buf (: 7 i64)))
      (def Vp_i32 (llvm.load {:result i32} Vp_ptr))
      (def Vp (arith.extsi {:result i64} Vp_i32))
    
      ;; Print config
      (print "[GPT-2 Config]\n")
      (print "  max_seq_len: %d\n" maxT_i32)
      (print "  vocab_size: %d\n" V_i32)
      (print "  num_layers: %d\n" L_i32)
      (print "  num_heads: %d\n" NH_i32)
      (print "  channels: %d\n" C_i32)
    
      ;; Compute derived dimensions
      (def C3 (arith.muli C (: 3 i64)))
      (def C4 (arith.muli C (: 4 i64)))
    
      ;; Compute parameter sizes
      (def size_wte (arith.muli Vp C))
      (def size_wpe (arith.muli maxT C))
      (def size_ln1w (arith.muli L C))
      (def size_ln1b (arith.muli L C))
      (def size_qkvw (arith.muli L (arith.muli C3 C)))
      (def size_qkvb (arith.muli L C3))
      (def size_attprojw (arith.muli L (arith.muli C C)))
      (def size_attprojb (arith.muli L C))
      (def size_ln2w (arith.muli L C))
      (def size_ln2b (arith.muli L C))
      (def size_fcw (arith.muli L (arith.muli C4 C)))
      (def size_fcb (arith.muli L C4))
      (def size_fcprojw (arith.muli L (arith.muli C C4)))
      (def size_fcprojb (arith.muli L C))
      (def size_lnfw C)
      (def size_lnfb C)
    
      ;; Compute total params
      (def total_params
        (arith.addi size_wte
          (arith.addi size_wpe
            (arith.addi size_ln1w
              (arith.addi size_ln1b
                (arith.addi size_qkvw
                  (arith.addi size_qkvb
                    (arith.addi size_attprojw
                      (arith.addi size_attprojb
                        (arith.addi size_ln2w
                          (arith.addi size_ln2b
                            (arith.addi size_fcw
                              (arith.addi size_fcb
                                (arith.addi size_fcprojw
                                  (arith.addi size_fcprojb
                                    (arith.addi size_lnfw size_lnfb))))))))))))))))
    
      (print "  total_params: %ld\n" total_params)
    
      ;; Compute base offsets
      (def wte_base (: 0 i64))
      (def wpe_base (arith.addi wte_base size_wte))
      (def ln1w_base (arith.addi wpe_base size_wpe))
      (def ln1b_base (arith.addi ln1w_base size_ln1w))
      (def qkvw_base (arith.addi ln1b_base size_ln1b))
      (def qkvb_base (arith.addi qkvw_base size_qkvw))
      (def attprojw_base (arith.addi qkvb_base size_qkvb))
      (def attprojb_base (arith.addi attprojw_base size_attprojw))
      (def ln2w_base (arith.addi attprojb_base size_attprojb))
      (def ln2b_base (arith.addi ln2w_base size_ln2w))
      (def fcw_base (arith.addi ln2b_base size_ln2b))
      (def fcb_base (arith.addi fcw_base size_fcw))
      (def fcprojw_base (arith.addi fcb_base size_fcb))
      (def fcprojb_base (arith.addi fcprojw_base size_fcprojw))
      (def lnfw_base (arith.addi fcprojb_base size_fcprojb))
      (def lnfb_base (arith.addi lnfw_base size_lnfw))
    
      ;; Compute per-layer strides
      (def ln_stride C)
      (def qkvw_stride (arith.muli C3 C))
      (def qkvb_stride C3)
      (def attprojw_stride (arith.muli C C))
      (def attprojb_stride C)
      (def fcw_stride (arith.muli C4 C))
      (def fcb_stride C4)
      (def fcprojw_stride (arith.muli C C4))
      (def fcprojb_stride C)
    
      ;; Load parameters
      (def sizeof_f32 (: 4 i64))
      (def total_bytes (arith.muli total_params sizeof_f32))
      (def params_ptr (func.call {:callee "@malloc" :result !llvm.ptr} total_bytes))
      (print "Loading weights...\n")
      (def read_count (func.call {:callee "@fread" :result i64} params_ptr sizeof_f32 total_params file))
      (print "Loaded %ld floats\n" read_count)
      (def _close (func.call {:callee "@fclose" :result i32} file))
    
      ;; Store params globally
      (def g_params_addr (llvm.mlir.addressof {:global_name @g_params :result !llvm.ptr}))
      (llvm.store params_ptr g_params_addr)
    
      ;; Verify wte[0][0]
      (def wte_val (llvm.load {:result f32} params_ptr))
      (def wte_val_f64 (arith.extf {:result f64} wte_val))
      (print "wte[0][0] = %f\n" wte_val_f64)
    
      ;; Load tokenizer
      (def tok_path (llvm.mlir.addressof {:global_name @tokenizer_path :result !llvm.ptr}))
      (def _tok_ok (func.call {:callee "@tokenizer_init" :result i32} tok_path))
    
      ;; Get wte and wpe pointers
      (def wte_ptr params_ptr)
      (def wpe_ptr (ptr-at f32 params_ptr wpe_base))
    
      ;; Create token_ids buffer with EOT token
      (def token_ids (memref.alloc {:result memref<64xi32>}))
      (gpu.host_register (memref.cast {:result "memref<*xi32>"} token_ids))
      (def eot_global (llvm.mlir.addressof {:global_name @eot_token :result !llvm.ptr}))
      (def eot_token (llvm.load {:result i32} eot_global))
      (def c0 (: 0 index))
      (def c1 (: 1 index))
      (def c64 (: 64 index))
    
      ;; Fill with EOT
      (scf.for c0 c64 c1
        (region
          (block [(: i index)]
            (memref.store eot_token token_ids i)
            (scf.yield))))
    
      (print "Prompt: <|endoftext|>\nGenerated:")
    
      ;; Allocate activation buffers
      (def x (memref.alloc {:result memref<64x768xf32>}))
      (def x2 (memref.alloc {:result memref<64x768xf32>}))
      (def ln_out (memref.alloc {:result memref<64x768xf32>}))
      (def qkv_out (memref.alloc {:result memref<64x2304xf32>}))
      (def attn_out (memref.alloc {:result memref<64x768xf32>}))
      (def attn_proj_out (memref.alloc {:result memref<64x768xf32>}))
      (def fc_out (memref.alloc {:result memref<64x3072xf32>}))
      (def gelu_out (memref.alloc {:result memref<64x3072xf32>}))
      (def fc_proj_out (memref.alloc {:result memref<64x768xf32>}))
      (def logits (memref.alloc {:result memref<50257xf32>}))

      ;; Attention workspace buffers (for batched attention)
      (def Q_batched (memref.alloc {:result memref<12x64x64xf32>}))
      (def K_batched (memref.alloc {:result memref<12x64x64xf32>}))
      (def V_batched (memref.alloc {:result memref<12x64x64xf32>}))
      (def K_transposed (memref.alloc {:result memref<12x64x64xf32>}))
      (def attn_scores (memref.alloc {:result memref<12x64x64xf32>}))
      (def attn_weights (memref.alloc {:result memref<12x64x64xf32>}))
      (def attn_values (memref.alloc {:result memref<12x64x64xf32>}))

      ;; Causal softmax workspace buffers
      (def softmax_masked (memref.alloc {:result memref<12x64x64xf32>}))
      (def softmax_max_buf (memref.alloc {:result memref<12x64xf32>}))
      (def softmax_sum_buf (memref.alloc {:result memref<12x64xf32>}))

      ;; LayerNorm temporary buffers for mean and reciprocal std
      (def ln_mean_buf (memref.alloc {:result memref<64xf32>}))
      (def ln_rs_buf (memref.alloc {:result memref<64xf32>}))

      ;; Layer weight buffers - ALL 12 LAYERS cached (eliminates per-token ptr loading)
      ;; BF16 for weight matrices to reduce memory bandwidth (~170MB instead of 340MB)
      (def all_ln1_w (memref.alloc {:result memref<12x768xf32>}))
      (def all_ln1_b (memref.alloc {:result memref<12x768xf32>}))
      (def all_qkv_w (memref.alloc {:result memref<12x768x2304xbf16>}))
      (def all_qkv_b (memref.alloc {:result memref<12x2304xf32>}))
      (def all_attn_w (memref.alloc {:result memref<12x768x768xbf16>}))
      (def all_attn_b (memref.alloc {:result memref<12x768xf32>}))
      (def all_ln2_w (memref.alloc {:result memref<12x768xf32>}))
      (def all_ln2_b (memref.alloc {:result memref<12x768xf32>}))
      (def all_fc_w (memref.alloc {:result memref<12x768x3072xbf16>}))
      (def all_fc_b (memref.alloc {:result memref<12x3072xf32>}))
      (def all_fcproj_w (memref.alloc {:result memref<12x3072x768xbf16>}))
      (def all_fcproj_b (memref.alloc {:result memref<12x768xf32>}))
      ;; Final layer norm (not per-layer, just one)
      (def lnf_w (memref.alloc {:result memref<768xf32>}))
      (def lnf_b (memref.alloc {:result memref<768xf32>}))

      ;; Token and position embeddings as memrefs (for GPU access)
      ;; wte: 50257 x 768 = ~147MB, wpe: 1024 x 768 = ~3MB
      (def wte_memref (memref.alloc {:result memref<50257x768xf32>}))
      (def wpe_memref (memref.alloc {:result memref<1024x768xf32>}))

      ;; Register all buffers with GPU runtime
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} x))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} x2))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} ln_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} ln_mean_buf))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} ln_rs_buf))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} qkv_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_proj_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} fc_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} gelu_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} fc_proj_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} logits))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} all_ln1_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} all_ln1_b))
      (gpu.host_register (memref.cast {:result "memref<*xbf16>"} all_qkv_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} all_qkv_b))
      (gpu.host_register (memref.cast {:result "memref<*xbf16>"} all_attn_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} all_attn_b))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} all_ln2_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} all_ln2_b))
      (gpu.host_register (memref.cast {:result "memref<*xbf16>"} all_fc_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} all_fc_b))
      (gpu.host_register (memref.cast {:result "memref<*xbf16>"} all_fcproj_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} all_fcproj_b))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} lnf_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} lnf_b))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} wte_memref))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} wpe_memref))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} Q_batched))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} K_batched))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} V_batched))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} K_transposed))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_scores))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_weights))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_values))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} softmax_masked))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} softmax_max_buf))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} softmax_sum_buf))

      (def c768 (: 768 index))
      (def c2304 (: 2304 index))
      (def c3072 (: 3072 index))
      (def c12 (: 12 index))
      (def C (: 768 i64))
      (def C4 (: 3072 i64))

      ;; =====================================================================
      ;; PRE-LOAD WTE AND WPE EMBEDDINGS TO MEMREFS
      ;; wte: 50257 x 768 floats, wpe: 1024 x 768 floats
      ;; =====================================================================
      (def c50257 (: 50257 index))
      (def c1024 (: 1024 index))
      (def C768_i64 (: 768 i64))

      ;; Copy wte (token embeddings) from params_ptr to wte_memref
      (scf.for c0 c50257 c1
        (region
          (block [(: v index)]
            (def v_i64 (arith.index_cast {:result i64} v))
            (def row_offset (arith.muli v_i64 C768_i64))
            (scf.for c0 c768 c1
              (region
                (block [(: c index)]
                  (def c_i64 (arith.index_cast {:result i64} c))
                  (def idx (arith.addi row_offset c_i64))
                  (def val_ptr (ptr-at f32 wte_ptr idx))
                  (def val (llvm.load {:result f32} val_ptr))
                  (memref.store val wte_memref v c)
                  (scf.yield))))
            (scf.yield))))

      ;; Copy wpe (position embeddings) from params_ptr to wpe_memref
      (scf.for c0 c1024 c1
        (region
          (block [(: t index)]
            (def t_i64 (arith.index_cast {:result i64} t))
            (def row_offset (arith.muli t_i64 C768_i64))
            (scf.for c0 c768 c1
              (region
                (block [(: c index)]
                  (def c_i64 (arith.index_cast {:result i64} c))
                  (def idx (arith.addi row_offset c_i64))
                  (def val_ptr (ptr-at f32 wpe_ptr idx))
                  (def val (llvm.load {:result f32} val_ptr))
                  (memref.store val wpe_memref t c)
                  (scf.yield))))
            (scf.yield))))

      ;; =====================================================================
      ;; PRE-LOAD ALL 12 LAYERS OF WEIGHTS (eliminates per-token loading!)
      ;; This is done ONCE before generation, not per token
      ;; =====================================================================
      (print "Pre-loading all layer weights...\n")
      (scf.for c0 c12 c1
        (region
          (block [(: layer index)]
            (def layer_i64 (arith.index_cast {:result i64} layer))

            ;; Load ln1 weights for this layer
            (def ln1w_offset (arith.addi ln1w_base (arith.muli layer_i64 ln_stride)))
            (def ln1b_offset (arith.addi ln1b_base (arith.muli layer_i64 ln_stride)))
            (scf.for c0 c768 c1
              (region
                (block [(: i index)]
                  (def i_i64 (arith.index_cast {:result i64} i))
                  (def w_ptr (ptr-at f32 params_ptr (arith.addi ln1w_offset i_i64)))
                  (def b_ptr (ptr-at f32 params_ptr (arith.addi ln1b_offset i_i64)))
                  (memref.store (llvm.load {:result f32} w_ptr) all_ln1_w layer i)
                  (memref.store (llvm.load {:result f32} b_ptr) all_ln1_b layer i)
                  (scf.yield))))

            ;; Load qkv weights (transpose: checkpoint[oc*C+c] -> qkv_w[layer][c][oc])
            (def qkvw_offset (arith.addi qkvw_base (arith.muli layer_i64 qkvw_stride)))
            (def qkvb_offset (arith.addi qkvb_base (arith.muli layer_i64 qkvb_stride)))
            (scf.for c0 c2304 c1
              (region
                (block [(: oc index)]
                  (def oc_i64 (arith.index_cast {:result i64} oc))
                  (def row_offset (arith.muli oc_i64 C))
                  (scf.for c0 c768 c1
                    (region
                      (block [(: c index)]
                        (def c_i64 (arith.index_cast {:result i64} c))
                        (def w_ptr (ptr-at f32 params_ptr (arith.addi qkvw_offset (arith.addi row_offset c_i64))))
                        (def w_f32 (llvm.load {:result f32} w_ptr))
                        (def w_bf16 (arith.truncf {:result bf16} w_f32))
                        (memref.store w_bf16 all_qkv_w layer c oc)
                        (scf.yield))))
                  (scf.yield))))
            (scf.for c0 c2304 c1
              (region
                (block [(: i index)]
                  (def i_i64 (arith.index_cast {:result i64} i))
                  (def b_ptr (ptr-at f32 params_ptr (arith.addi qkvb_offset i_i64)))
                  (memref.store (llvm.load {:result f32} b_ptr) all_qkv_b layer i)
                  (scf.yield))))

            ;; Load attention projection weights (transpose)
            (def attprojw_offset (arith.addi attprojw_base (arith.muli layer_i64 attprojw_stride)))
            (def attprojb_offset (arith.addi attprojb_base (arith.muli layer_i64 attprojb_stride)))
            (scf.for c0 c768 c1
              (region
                (block [(: oc index)]
                  (def oc_i64 (arith.index_cast {:result i64} oc))
                  (def row_offset (arith.muli oc_i64 C))
                  (scf.for c0 c768 c1
                    (region
                      (block [(: c index)]
                        (def c_i64 (arith.index_cast {:result i64} c))
                        (def w_ptr (ptr-at f32 params_ptr (arith.addi attprojw_offset (arith.addi row_offset c_i64))))
                        (def w_f32 (llvm.load {:result f32} w_ptr))
                        (def w_bf16 (arith.truncf {:result bf16} w_f32))
                        (memref.store w_bf16 all_attn_w layer c oc)
                        (scf.yield))))
                  (scf.yield))))
            (scf.for c0 c768 c1
              (region
                (block [(: i index)]
                  (def i_i64 (arith.index_cast {:result i64} i))
                  (def ab_ptr (ptr-at f32 params_ptr (arith.addi attprojb_offset i_i64)))
                  (memref.store (llvm.load {:result f32} ab_ptr) all_attn_b layer i)
                  (scf.yield))))

            ;; Load ln2 weights
            (def ln2w_offset (arith.addi ln2w_base (arith.muli layer_i64 ln_stride)))
            (def ln2b_offset (arith.addi ln2b_base (arith.muli layer_i64 ln_stride)))
            (scf.for c0 c768 c1
              (region
                (block [(: i index)]
                  (def i_i64 (arith.index_cast {:result i64} i))
                  (def w_ptr (ptr-at f32 params_ptr (arith.addi ln2w_offset i_i64)))
                  (def b_ptr (ptr-at f32 params_ptr (arith.addi ln2b_offset i_i64)))
                  (memref.store (llvm.load {:result f32} w_ptr) all_ln2_w layer i)
                  (memref.store (llvm.load {:result f32} b_ptr) all_ln2_b layer i)
                  (scf.yield))))

            ;; Load fc weights (transpose)
            (def fcw_offset (arith.addi fcw_base (arith.muli layer_i64 fcw_stride)))
            (def fcb_offset (arith.addi fcb_base (arith.muli layer_i64 fcb_stride)))
            (scf.for c0 c3072 c1
              (region
                (block [(: oc index)]
                  (def oc_i64 (arith.index_cast {:result i64} oc))
                  (def row_offset (arith.muli oc_i64 C))
                  (scf.for c0 c768 c1
                    (region
                      (block [(: c index)]
                        (def c_i64 (arith.index_cast {:result i64} c))
                        (def w_ptr (ptr-at f32 params_ptr (arith.addi fcw_offset (arith.addi row_offset c_i64))))
                        (def w_f32 (llvm.load {:result f32} w_ptr))
                        (def w_bf16 (arith.truncf {:result bf16} w_f32))
                        (memref.store w_bf16 all_fc_w layer c oc)
                        (scf.yield))))
                  (scf.yield))))
            (scf.for c0 c3072 c1
              (region
                (block [(: i index)]
                  (def i_i64 (arith.index_cast {:result i64} i))
                  (def b_ptr (ptr-at f32 params_ptr (arith.addi fcb_offset i_i64)))
                  (memref.store (llvm.load {:result f32} b_ptr) all_fc_b layer i)
                  (scf.yield))))

            ;; Load fc projection weights (transpose)
            (def fcprojw_offset (arith.addi fcprojw_base (arith.muli layer_i64 fcprojw_stride)))
            (def fcprojb_offset (arith.addi fcprojb_base (arith.muli layer_i64 fcprojb_stride)))
            (scf.for c0 c768 c1
              (region
                (block [(: oc index)]
                  (def oc_i64 (arith.index_cast {:result i64} oc))
                  (def row_offset (arith.muli oc_i64 C4))
                  (scf.for c0 c3072 c1
                    (region
                      (block [(: ic index)]
                        (def ic_i64 (arith.index_cast {:result i64} ic))
                        (def w_ptr (ptr-at f32 params_ptr (arith.addi fcprojw_offset (arith.addi row_offset ic_i64))))
                        (def w_f32 (llvm.load {:result f32} w_ptr))
                        (def w_bf16 (arith.truncf {:result bf16} w_f32))
                        (memref.store w_bf16 all_fcproj_w layer ic oc)
                        (scf.yield))))
                  (scf.yield))))
            (scf.for c0 c768 c1
              (region
                (block [(: i index)]
                  (def i_i64 (arith.index_cast {:result i64} i))
                  (def b_ptr (ptr-at f32 params_ptr (arith.addi fcprojb_offset i_i64)))
                  (memref.store (llvm.load {:result f32} b_ptr) all_fcproj_b layer i)
                  (scf.yield))))

            (scf.yield))))

      ;; Load final layernorm weights (only one, not per-layer)
      (scf.for c0 c768 c1
        (region
          (block [(: i index)]
            (def i_i64 (arith.index_cast {:result i64} i))
            (def w_ptr (ptr-at f32 params_ptr (arith.addi lnfw_base i_i64)))
            (def b_ptr (ptr-at f32 params_ptr (arith.addi lnfb_base i_i64)))
            (memref.store (llvm.load {:result f32} w_ptr) lnf_w i)
            (memref.store (llvm.load {:result f32} b_ptr) lnf_b i)
            (scf.yield))))

      (print "Weights pre-loaded!\n")

      ;; Generation loop: Generate 20 tokens
      (def prompt_len (: 1 index))
      (def gen_steps (: 20 index))
      (def gen_end (arith.addi prompt_len gen_steps))

      (print "\nGenerating tokens:\n")

      (def start_time (func.call {:callee "@clock_ms" :result i64}))

      (scf.for prompt_len gen_end c1
        (region
          (block [(: step index)]
            ;; 1. Embedding lookup
            (func.call {:callee "@embedding_lookup"} x wte_memref wpe_memref token_ids)
    
            ;; 2. Run 12 transformer layers (copy from cached weights - no transpose!)
            (scf.for c0 c12 c1
              (region
                (block [(: layer index)]
                  ;; Create subviews into cached weight buffers (zero-copy!)
                  ;; 2D→1D: memref<12x768xf32>[layer,:] → memref<768xf32, strided<[1], offset: ?>>
                  (def ln1_w_view (memref.subview {:result "memref<768xf32, strided<[1], offset: ?>>"
                                                   :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                   :static_offsets "array<i64: -9223372036854775808, 0>"
                                                   :static_sizes "array<i64: 1, 768>"
                                                   :static_strides "array<i64: 1, 1>"}
                                    all_ln1_w layer))
                  (def ln1_b_view (memref.subview {:result "memref<768xf32, strided<[1], offset: ?>>"
                                                   :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                   :static_offsets "array<i64: -9223372036854775808, 0>"
                                                   :static_sizes "array<i64: 1, 768>"
                                                   :static_strides "array<i64: 1, 1>"}
                                    all_ln1_b layer))

                  ;; 3D→2D: memref<12x768x2304xbf16>[layer,:,:] → memref<768x2304xbf16, strided<[2304, 1], offset: ?>>
                  (def qkv_w_view (memref.subview {:result "memref<768x2304xbf16, strided<[2304, 1], offset: ?>>"
                                                   :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                   :static_offsets "array<i64: -9223372036854775808, 0, 0>"
                                                   :static_sizes "array<i64: 1, 768, 2304>"
                                                   :static_strides "array<i64: 1, 1, 1>"}
                                    all_qkv_w layer))
                  (def qkv_b_view (memref.subview {:result "memref<2304xf32, strided<[1], offset: ?>>"
                                                   :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                   :static_offsets "array<i64: -9223372036854775808, 0>"
                                                   :static_sizes "array<i64: 1, 2304>"
                                                   :static_strides "array<i64: 1, 1>"}
                                    all_qkv_b layer))

                  ;; 3D→2D: memref<12x768x768xbf16>[layer,:,:] → memref<768x768xbf16, strided<[768, 1], offset: ?>>
                  (def attn_w_view (memref.subview {:result "memref<768x768xbf16, strided<[768, 1], offset: ?>>"
                                                    :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                    :static_offsets "array<i64: -9223372036854775808, 0, 0>"
                                                    :static_sizes "array<i64: 1, 768, 768>"
                                                    :static_strides "array<i64: 1, 1, 1>"}
                                     all_attn_w layer))
                  (def attn_b_view (memref.subview {:result "memref<768xf32, strided<[1], offset: ?>>"
                                                    :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                    :static_offsets "array<i64: -9223372036854775808, 0>"
                                                    :static_sizes "array<i64: 1, 768>"
                                                    :static_strides "array<i64: 1, 1>"}
                                     all_attn_b layer))

                  ;; 2D→1D: memref<12x768xf32>[layer,:] → memref<768xf32, strided<[1], offset: ?>>
                  (def ln2_w_view (memref.subview {:result "memref<768xf32, strided<[1], offset: ?>>"
                                                   :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                   :static_offsets "array<i64: -9223372036854775808, 0>"
                                                   :static_sizes "array<i64: 1, 768>"
                                                   :static_strides "array<i64: 1, 1>"}
                                    all_ln2_w layer))
                  (def ln2_b_view (memref.subview {:result "memref<768xf32, strided<[1], offset: ?>>"
                                                   :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                   :static_offsets "array<i64: -9223372036854775808, 0>"
                                                   :static_sizes "array<i64: 1, 768>"
                                                   :static_strides "array<i64: 1, 1>"}
                                    all_ln2_b layer))

                  ;; 3D→2D: memref<12x768x3072xbf16>[layer,:,:] → memref<768x3072xbf16, strided<[3072, 1], offset: ?>>
                  (def fc_w_view (memref.subview {:result "memref<768x3072xbf16, strided<[3072, 1], offset: ?>>"
                                                  :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                  :static_offsets "array<i64: -9223372036854775808, 0, 0>"
                                                  :static_sizes "array<i64: 1, 768, 3072>"
                                                  :static_strides "array<i64: 1, 1, 1>"}
                                   all_fc_w layer))
                  (def fc_b_view (memref.subview {:result "memref<3072xf32, strided<[1], offset: ?>>"
                                                  :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                  :static_offsets "array<i64: -9223372036854775808, 0>"
                                                  :static_sizes "array<i64: 1, 3072>"
                                                  :static_strides "array<i64: 1, 1>"}
                                   all_fc_b layer))

                  ;; 3D→2D: memref<12x3072x768xbf16>[layer,:,:] → memref<3072x768xbf16, strided<[768, 1], offset: ?>>
                  (def fcproj_w_view (memref.subview {:result "memref<3072x768xbf16, strided<[768, 1], offset: ?>>"
                                                      :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                      :static_offsets "array<i64: -9223372036854775808, 0, 0>"
                                                      :static_sizes "array<i64: 1, 3072, 768>"
                                                      :static_strides "array<i64: 1, 1, 1>"}
                                       all_fcproj_w layer))
                  (def fcproj_b_view (memref.subview {:result "memref<768xf32, strided<[1], offset: ?>>"
                                                      :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                                      :static_offsets "array<i64: -9223372036854775808, 0>"
                                                      :static_sizes "array<i64: 1, 768>"
                                                      :static_strides "array<i64: 1, 1>"}
                                       all_fcproj_b layer))

                  ;; === Transformer block operations ===
                  ;; 1. LayerNorm1
                  (func.call {:callee "@layernorm_forward"} ln_out x ln1_w_view ln1_b_view ln_mean_buf ln_rs_buf)

                  ;; 2. QKV Projection (using linalg.matmul)
                  (func.call {:callee "@matmul_qkv"} qkv_out ln_out qkv_w_view qkv_b_view)

                  ;; 3. Attention (GPU batched matmul + softmax)
                  (func.call {:callee "@attention_forward"} attn_out qkv_out
                             Q_batched K_batched V_batched
                             K_transposed attn_scores attn_weights attn_values
                             softmax_masked softmax_max_buf softmax_sum_buf)

                  ;; 4. Attention Output Projection (using linalg.matmul)
                  (func.call {:callee "@matmul_attn_proj"} attn_proj_out attn_out attn_w_view attn_b_view)

                  ;; 5. Residual 1 (using linalg.add)
                  (func.call {:callee "@residual_forward"} x2 x attn_proj_out)

                  ;; 6. LayerNorm2
                  (func.call {:callee "@layernorm_forward"} ln_out x2 ln2_w_view ln2_b_view ln_mean_buf ln_rs_buf)

                  ;; 7. MLP FC (using linalg.matmul)
                  (func.call {:callee "@matmul_fc"} fc_out ln_out fc_w_view fc_b_view)

                  ;; 8. GELU (using math.tanh)
                  (func.call {:callee "@gelu_forward"} gelu_out fc_out)

                  ;; 9. MLP Projection (using linalg.matmul)
                  (func.call {:callee "@matmul_fc_proj"} fc_proj_out gelu_out fcproj_w_view fcproj_b_view)

                  ;; 10. Residual 2 (using linalg.add)
                  (func.call {:callee "@residual_forward"} x x2 fc_proj_out)

                  (scf.yield))))

            ;; 3. Final LayerNorm (uses pre-loaded lnf_w, lnf_b)
            ;; Cast to strided memref for compatibility with layer-based function signature
            (def lnf_w_cast (memref.cast {:result "memref<768xf32, strided<[1], offset: ?>>"} lnf_w))
            (def lnf_b_cast (memref.cast {:result "memref<768xf32, strided<[1], offset: ?>>"} lnf_b))
            (func.call {:callee "@layernorm_forward"} x2 x lnf_w_cast lnf_b_cast ln_mean_buf ln_rs_buf)
    
            ;; 4. Compute logits from position step-1 (the last token's position)
            (def logit_pos (arith.subi step c1))
            (func.call {:callee "@logits_forward"} logits x2 wte_memref logit_pos)
    
            ;; 5. Argmax to get next token
            (def next_token (func.call {:callee "@argmax" :result i32} logits))
    
            ;; 6. Print token
            (func.call {:callee "@print_token"} next_token)
    
            ;; 7. Store token at current step position
            (memref.store next_token token_ids step)
    
            (scf.yield))))

      (def end_time (func.call {:callee "@clock_ms" :result i64}))
      (def elapsed_ms (arith.subi end_time start_time))
      (print "\n\nGeneration complete!\n")
      (print "Time for 20 tokens: %ld ms\n" elapsed_ms)
      (print "Per token: %ld ms\n" (arith.divsi elapsed_ms (: 20 i64)))
    
      ;; Cleanup
      (memref.dealloc token_ids)
      (memref.dealloc x)
      (memref.dealloc x2)
      (memref.dealloc ln_out)
      (memref.dealloc qkv_out)
      (memref.dealloc attn_out)
      (memref.dealloc attn_proj_out)
      (memref.dealloc fc_out)
      (memref.dealloc gelu_out)
      (memref.dealloc fc_proj_out)
      (memref.dealloc logits)
      (memref.dealloc Q_batched)
      (memref.dealloc K_batched)
      (memref.dealloc V_batched)
      (memref.dealloc K_transposed)
      (memref.dealloc attn_scores)
      (memref.dealloc attn_weights)
      (memref.dealloc attn_values)
      (memref.dealloc lnf_w)
      (memref.dealloc lnf_b)
      (memref.dealloc wte_memref)
      (memref.dealloc wpe_memref)
    
      (func.call {:callee "@free"} params_ptr)

      (func.return))))