;; Profile individual operations to find bottlenecks
;; Tests each operation type with GPT-2 dimensions

(require-dialect memref)
(require-dialect arith)
(require-dialect func)
(require-dialect gpu)
(require-dialect scf)
(require-dialect math)
(require-dialect linalg)

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

;; printf is variadic - use extern-fn which generates llvm.func
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

(module
  (do
    (func.func {:sym_name "clock_ms"
                :function_type (-> [] [i64])
                :sym_visibility "private"}
      (region))

    ;; Test 1: Large matmul (768x768) - like attention projection
    (func.func {:sym_name "test_matmul_768x768"
                :function_type (-> [memref<1x768xf32>
                                    memref<1x768xf32>
                                    memref<768x768xbf16>] [])}
      (region
        (block [(: out memref<1x768xf32>)
                (: inp memref<1x768xf32>)
                (: weight memref<768x768xbf16>)]

          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region (block [(: in f32) (: _out f32)] (linalg.yield in))))

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

          (func.return))))

    ;; Test 2: QKV matmul (768x2304) - largest matmul
    (func.func {:sym_name "test_matmul_768x2304"
                :function_type (-> [memref<1x2304xf32>
                                    memref<1x768xf32>
                                    memref<768x2304xbf16>] [])}
      (region
        (block [(: out memref<1x2304xf32>)
                (: inp memref<1x768xf32>)
                (: weight memref<768x2304xbf16>)]

          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region (block [(: in f32) (: _out f32)] (linalg.yield in))))

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

          (func.return))))

    ;; Test 3: FC matmul (768x3072) - MLP expansion
    (func.func {:sym_name "test_matmul_768x3072"
                :function_type (-> [memref<1x3072xf32>
                                    memref<1x768xf32>
                                    memref<768x3072xbf16>] [])}
      (region
        (block [(: out memref<1x3072xf32>)
                (: inp memref<1x768xf32>)
                (: weight memref<768x3072xbf16>)]

          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region (block [(: in f32) (: _out f32)] (linalg.yield in))))

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

          (func.return))))

    ;; Test 4: FC projection (3072x768) - MLP contraction
    (func.func {:sym_name "test_matmul_3072x768"
                :function_type (-> [memref<1x768xf32>
                                    memref<1x3072xf32>
                                    memref<3072x768xbf16>] [])}
      (region
        (block [(: out memref<1x768xf32>)
                (: inp memref<1x3072xf32>)
                (: weight memref<3072x768xbf16>)]

          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region (block [(: in f32) (: _out f32)] (linalg.yield in))))

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

          (func.return))))

    ;; Test 5: Batched attention matmul (12 heads, 64x64)
    (func.func {:sym_name "test_batch_matmul"
                :function_type (-> [memref<12x1x64xf32>
                                    memref<12x1x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: out memref<12x1x64xf32>)
                (: Q memref<12x1x64xf32>)
                (: K_t memref<12x64x64xf32>)]

          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region (block [(: in f32) (: _out f32)] (linalg.yield in))))

          (linalg.batch_matmul {:ins 2 :outs 1} Q K_t out
            (region
              (block [(: q_val f32) (: k_val f32) (: accum f32)]
                (def prod (arith.mulf q_val k_val))
                (def sum (arith.addf accum prod))
                (linalg.yield sum))))

          (func.return))))

    ;; Test 6: LayerNorm (768 elements)
    (func.func {:sym_name "test_layernorm"
                :function_type (-> [memref<1x768xf32>
                                    memref<1x768xf32>
                                    memref<768xf32>
                                    memref<768xf32>] [])}
      (region
        (block [(: out memref<1x768xf32>)
                (: inp memref<1x768xf32>)
                (: gamma memref<768xf32>)
                (: beta memref<768xf32>)]

          (def c0 (: 0 index))
          (def c768 (: 768 index))
          (def c1 (: 1 index))
          (def eps (: 1e-5 f32))
          (def zero (: 0.0 f32))
          (def n_f32 (: 768.0 f32))

          ;; Compute mean
          (def sum_val (scf.for {:result f32} c0 c768 c1 zero
            (region
              (block [(: i index) (: acc f32)]
                (def val (memref.load {:result f32} inp c0 i))
                (def new_acc (arith.addf acc val))
                (scf.yield new_acc)))))
          (def mean (arith.divf sum_val n_f32))

          ;; Compute variance
          (def var_sum (scf.for {:result f32} c0 c768 c1 zero
            (region
              (block [(: i index) (: acc f32)]
                (def val (memref.load {:result f32} inp c0 i))
                (def diff (arith.subf val mean))
                (def sq (arith.mulf diff diff))
                (def new_acc (arith.addf acc sq))
                (scf.yield new_acc)))))
          (def var (arith.divf var_sum n_f32))
          (def std (math.sqrt (arith.addf var eps)))
          (def inv_std (arith.divf (: 1.0 f32) std))

          ;; Normalize
          (scf.for c0 c768 c1
            (region
              (block [(: i index)]
                (def val (memref.load {:result f32} inp c0 i))
                (def g (memref.load {:result f32} gamma i))
                (def b (memref.load {:result f32} beta i))
                (def norm (arith.mulf (arith.subf val mean) inv_std))
                (def scaled (arith.addf (arith.mulf norm g) b))
                (memref.store scaled out c0 i)
                (scf.yield))))

          (func.return))))

    ;; Test 7: GELU (3072 elements)
    (func.func {:sym_name "test_gelu"
                :function_type (-> [memref<1x3072xf32>
                                    memref<1x3072xf32>] [])}
      (region
        (block [(: out memref<1x3072xf32>)
                (: inp memref<1x3072xf32>)]

          (def c0 (: 0 index))
          (def c3072 (: 3072 index))
          (def c1 (: 1 index))
          (def half (: 0.5 f32))
          (def one (: 1.0 f32))
          (def sqrt_2_over_pi (: 0.7978845608 f32))
          (def coeff (: 0.044715 f32))

          (scf.for c0 c3072 c1
            (region
              (block [(: i index)]
                (def x (memref.load {:result f32} inp c0 i))
                (def x3 (arith.mulf x (arith.mulf x x)))
                (def inner (arith.mulf sqrt_2_over_pi (arith.addf x (arith.mulf coeff x3))))
                (def tanh_val (math.tanh inner))
                (def gelu (arith.mulf (arith.mulf half x) (arith.addf one tanh_val)))
                (memref.store gelu out c0 i)
                (scf.yield))))

          (func.return))))

    ;; Test 8: Logits matmul (768 x 50257) - this is the final projection
    (func.func {:sym_name "test_matmul_logits"
                :function_type (-> [memref<1x50257xf32>
                                    memref<1x768xf32>
                                    memref<768x50257xf32>] [])}
      (region
        (block [(: out memref<1x50257xf32>)
                (: inp memref<1x768xf32>)
                (: weight memref<768x50257xf32>)]

          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero out
            (region (block [(: in f32) (: _out f32)] (linalg.yield in))))

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
              (block [(: a f32) (: b f32) (: c f32)]
                (def mul (arith.mulf a b))
                (def sum (arith.addf c mul))
                (linalg.yield sum))))

          (func.return))))

    (defn main []
      (def c0 (: 0 index))
      (def c1 (: 1 index))
      (def num_iters (: 100 index))

      ;; Allocate test buffers
      (def buf_1x768 (memref.alloc {:result memref<1x768xf32>}))
      (def buf_1x768_2 (memref.alloc {:result memref<1x768xf32>}))
      (def buf_1x2304 (memref.alloc {:result memref<1x2304xf32>}))
      (def buf_1x3072 (memref.alloc {:result memref<1x3072xf32>}))
      (def buf_1x3072_2 (memref.alloc {:result memref<1x3072xf32>}))
      (def w_768x768 (memref.alloc {:result memref<768x768xbf16>}))
      (def w_768x2304 (memref.alloc {:result memref<768x2304xbf16>}))
      (def w_768x3072 (memref.alloc {:result memref<768x3072xbf16>}))
      (def w_3072x768 (memref.alloc {:result memref<3072x768xbf16>}))
      (def gamma (memref.alloc {:result memref<768xf32>}))
      (def beta (memref.alloc {:result memref<768xf32>}))
      (def Q_batch (memref.alloc {:result memref<12x1x64xf32>}))
      (def K_batch (memref.alloc {:result memref<12x64x64xf32>}))
      (def attn_out (memref.alloc {:result memref<12x1x64xf32>}))
      ;; Logits buffers (768 x 50257)
      (def logits_out (memref.alloc {:result memref<1x50257xf32>}))
      (def wte_weight (memref.alloc {:result memref<768x50257xf32>}))

      ;; Register for GPU FIRST (before any linalg ops which become GPU kernels)
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} buf_1x768))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} buf_1x768_2))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} buf_1x2304))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} buf_1x3072))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} buf_1x3072_2))
      (gpu.host_register (memref.cast {:result "memref<*xbf16>"} w_768x768))
      (gpu.host_register (memref.cast {:result "memref<*xbf16>"} w_768x2304))
      (gpu.host_register (memref.cast {:result "memref<*xbf16>"} w_768x3072))
      (gpu.host_register (memref.cast {:result "memref<*xbf16>"} w_3072x768))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} gamma))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} beta))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} Q_batch))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} K_batch))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} logits_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} wte_weight))

      ;; Initialize with 1.0 (linalg.fill becomes GPU kernel)
      (def one_f32 (: 1.0 f32))
      (def one_bf16 (arith.truncf {:result bf16} one_f32))
      (linalg.fill {:ins 1 :outs 1} one_f32 buf_1x768
        (region (block [(: in f32) (: _out f32)] (linalg.yield in))))
      (linalg.fill {:ins 1 :outs 1} one_f32 buf_1x3072
        (region (block [(: in f32) (: _out f32)] (linalg.yield in))))
      (linalg.fill {:ins 1 :outs 1} one_bf16 w_768x768
        (region (block [(: in bf16) (: _out bf16)] (linalg.yield in))))
      (linalg.fill {:ins 1 :outs 1} one_bf16 w_768x2304
        (region (block [(: in bf16) (: _out bf16)] (linalg.yield in))))
      (linalg.fill {:ins 1 :outs 1} one_bf16 w_768x3072
        (region (block [(: in bf16) (: _out bf16)] (linalg.yield in))))
      (linalg.fill {:ins 1 :outs 1} one_bf16 w_3072x768
        (region (block [(: in bf16) (: _out bf16)] (linalg.yield in))))
      (linalg.fill {:ins 1 :outs 1} one_f32 gamma
        (region (block [(: in f32) (: _out f32)] (linalg.yield in))))
      (def zero_f32 (: 0.0 f32))
      (linalg.fill {:ins 1 :outs 1} zero_f32 beta
        (region (block [(: in f32) (: _out f32)] (linalg.yield in))))
      (linalg.fill {:ins 1 :outs 1} one_f32 Q_batch
        (region (block [(: in f32) (: _out f32)] (linalg.yield in))))
      (linalg.fill {:ins 1 :outs 1} one_f32 K_batch
        (region (block [(: in f32) (: _out f32)] (linalg.yield in))))
      (linalg.fill {:ins 1 :outs 1} one_f32 wte_weight
        (region (block [(: in f32) (: _out f32)] (linalg.yield in))))

      (print "=== GPT-2 Operation Profiling (100 iterations each) ===\n\n")

      ;; Test 1: 768x768 matmul (attention projection)
      (def t1_start (func.call {:callee "@clock_ms" :result i64}))
      (scf.for c0 num_iters c1
        (region
          (block [(: i index)]
            (func.call {:callee "@test_matmul_768x768"} buf_1x768_2 buf_1x768 w_768x768)
            (scf.yield))))
      (def t1_end (func.call {:callee "@clock_ms" :result i64}))
      (def t1 (arith.subi t1_end t1_start))
      (print "Matmul 768x768 (attn proj):    %ld ms total, " t1)
      (print "%ld ms/iter\n" (arith.divsi t1 (: 100 i64)))

      ;; Test 2: 768x2304 matmul (QKV projection)
      (def t2_start (func.call {:callee "@clock_ms" :result i64}))
      (scf.for c0 num_iters c1
        (region
          (block [(: i index)]
            (func.call {:callee "@test_matmul_768x2304"} buf_1x2304 buf_1x768 w_768x2304)
            (scf.yield))))
      (def t2_end (func.call {:callee "@clock_ms" :result i64}))
      (def t2 (arith.subi t2_end t2_start))
      (print "Matmul 768x2304 (QKV):         %ld ms total, " t2)
      (print "%ld ms/iter\n" (arith.divsi t2 (: 100 i64)))

      ;; Test 3: 768x3072 matmul (FC expansion)
      (def t3_start (func.call {:callee "@clock_ms" :result i64}))
      (scf.for c0 num_iters c1
        (region
          (block [(: i index)]
            (func.call {:callee "@test_matmul_768x3072"} buf_1x3072 buf_1x768 w_768x3072)
            (scf.yield))))
      (def t3_end (func.call {:callee "@clock_ms" :result i64}))
      (def t3 (arith.subi t3_end t3_start))
      (print "Matmul 768x3072 (FC):          %ld ms total, " t3)
      (print "%ld ms/iter\n" (arith.divsi t3 (: 100 i64)))

      ;; Test 4: 3072x768 matmul (FC projection)
      (def t4_start (func.call {:callee "@clock_ms" :result i64}))
      (scf.for c0 num_iters c1
        (region
          (block [(: i index)]
            (func.call {:callee "@test_matmul_3072x768"} buf_1x768_2 buf_1x3072 w_3072x768)
            (scf.yield))))
      (def t4_end (func.call {:callee "@clock_ms" :result i64}))
      (def t4 (arith.subi t4_end t4_start))
      (print "Matmul 3072x768 (FC proj):     %ld ms total, " t4)
      (print "%ld ms/iter\n" (arith.divsi t4 (: 100 i64)))

      ;; Test 5: Batched attention matmul
      (def t5_start (func.call {:callee "@clock_ms" :result i64}))
      (scf.for c0 num_iters c1
        (region
          (block [(: i index)]
            (func.call {:callee "@test_batch_matmul"} attn_out Q_batch K_batch)
            (scf.yield))))
      (def t5_end (func.call {:callee "@clock_ms" :result i64}))
      (def t5 (arith.subi t5_end t5_start))
      (print "Batch matmul 12x1x64x64:       %ld ms total, " t5)
      (print "%ld ms/iter\n" (arith.divsi t5 (: 100 i64)))

      ;; Test 6: LayerNorm
      (def t6_start (func.call {:callee "@clock_ms" :result i64}))
      (scf.for c0 num_iters c1
        (region
          (block [(: i index)]
            (func.call {:callee "@test_layernorm"} buf_1x768_2 buf_1x768 gamma beta)
            (scf.yield))))
      (def t6_end (func.call {:callee "@clock_ms" :result i64}))
      (def t6 (arith.subi t6_end t6_start))
      (print "LayerNorm 768:                 %ld ms total, " t6)
      (print "%ld ms/iter\n" (arith.divsi t6 (: 100 i64)))

      ;; Test 7: GELU
      (def t7_start (func.call {:callee "@clock_ms" :result i64}))
      (scf.for c0 num_iters c1
        (region
          (block [(: i index)]
            (func.call {:callee "@test_gelu"} buf_1x3072_2 buf_1x3072)
            (scf.yield))))
      (def t7_end (func.call {:callee "@clock_ms" :result i64}))
      (def t7 (arith.subi t7_end t7_start))
      (print "GELU 3072:                     %ld ms total, " t7)
      (print "%ld ms/iter\n" (arith.divsi t7 (: 100 i64)))

      ;; Test 8: Logits matmul (768 x 50257) - this is HUGE
      (def t8_start (func.call {:callee "@clock_ms" :result i64}))
      (scf.for c0 num_iters c1
        (region
          (block [(: i index)]
            (func.call {:callee "@test_matmul_logits"} logits_out buf_1x768 wte_weight)
            (scf.yield))))
      (def t8_end (func.call {:callee "@clock_ms" :result i64}))
      (def t8 (arith.subi t8_end t8_start))
      (print "Matmul 768x50257 (LOGITS):     %ld ms total, " t8)
      (print "%ld ms/iter\n" (arith.divsi t8 (: 100 i64)))

      ;; Summary
      (print "\n=== Per-Layer Estimate (12 layers) ===\n")
      (print "Each layer has: 2 LayerNorm, 1 QKV matmul, 2 batch matmuls,\n")
      (print "                1 attn proj matmul, 1 FC matmul, 1 GELU, 1 FC proj matmul\n")

      ;; Per-layer breakdown estimate
      (def ln_per_layer (arith.muli (arith.divsi t6 (: 100 i64)) (: 2 i64)))
      (def qkv_time (arith.divsi t2 (: 100 i64)))
      (def attn_matmul (arith.muli (arith.divsi t5 (: 100 i64)) (: 2 i64)))
      (def attn_proj (arith.divsi t1 (: 100 i64)))
      (def fc_time (arith.divsi t3 (: 100 i64)))
      (def gelu_time (arith.divsi t7 (: 100 i64)))
      (def fcproj_time (arith.divsi t4 (: 100 i64)))

      (def layer_total (arith.addi ln_per_layer
                         (arith.addi qkv_time
                           (arith.addi attn_matmul
                             (arith.addi attn_proj
                               (arith.addi fc_time
                                 (arith.addi gelu_time fcproj_time)))))))

      (print "Estimated per-layer time: %ld ms\n" layer_total)
      (print "Estimated 12-layer time:  %ld ms\n" (arith.muli layer_total (: 12 i64)))

      ;; Cleanup
      (memref.dealloc buf_1x768)
      (memref.dealloc buf_1x768_2)
      (memref.dealloc buf_1x2304)
      (memref.dealloc buf_1x3072)
      (memref.dealloc buf_1x3072_2)
      (memref.dealloc w_768x768)
      (memref.dealloc w_768x2304)
      (memref.dealloc w_768x3072)
      (memref.dealloc w_3072x768)
      (memref.dealloc gamma)
      (memref.dealloc beta)
      (memref.dealloc Q_batch)
      (memref.dealloc K_batch)
      (memref.dealloc attn_out)
      (func.return))))