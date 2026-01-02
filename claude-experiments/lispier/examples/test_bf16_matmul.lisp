;; Test BF16 matmul: bf16 weights, f32 activations and accumulation
;; out (f32) = inp (f32) @ weight (bf16)

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

    (defn main []
      (def c0 (: 0 index))
      (def c1 (: 1 index))
      (def c2 (: 2 index))
      (def c3 (: 3 index))
      (def c4 (: 4 index))

      ;; Test: (2x4) @ (4x3) = (2x3) with bf16 weights
      ;; inp is f32, weight is bf16, out is f32

      (def inp (memref.alloc {:result memref<2x4xf32>}))
      (def weight (memref.alloc {:result memref<4x3xbf16>}))
      (def out (memref.alloc {:result memref<2x3xf32>}))

      ;; Initialize inp with 1.0 (f32)
      (def one_f32 (: 1.0 f32))
      (linalg.fill {:ins 1 :outs 1} one_f32 inp
        (region
          (block [(: in f32) (: _out f32)]
            (linalg.yield in))))

      ;; Initialize weight with 0.5 (bf16)
      ;; Each output = sum over 4 inputs = 4 * 1.0 * 0.5 = 2.0
      (def half_bf16 (arith.truncf {:result bf16} (: 0.5 f32)))
      (linalg.fill {:ins 1 :outs 1} half_bf16 weight
        (region
          (block [(: in bf16) (: _out bf16)]
            (linalg.yield in))))

      ;; Zero out
      (def zero (: 0.0 f32))
      (linalg.fill {:ins 1 :outs 1} zero out
        (region
          (block [(: in f32) (: _out f32)]
            (linalg.yield in))))

      ;; Register for GPU
      (def inp_unranked (memref.cast {:result "memref<*xf32>"} inp))
      (def weight_unranked (memref.cast {:result "memref<*xbf16>"} weight))
      (def out_unranked (memref.cast {:result "memref<*xf32>"} out))
      (gpu.host_register inp_unranked)
      (gpu.host_register weight_unranked)
      (gpu.host_register out_unranked)

      ;; Matmul using linalg.generic with mixed types
      ;; inp: f32, weight: bf16, out: f32
      ;; We extend bf16 to f32 before multiplication
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
            ;; Extend bf16 weight to f32
            (def b_f32 (arith.extf {:result f32} b))
            ;; Multiply-accumulate in f32
            (def mul (arith.mulf a b_f32))
            (def sum (arith.addf c mul))
            (linalg.yield sum))))

      ;; Print results - should all be 2.0
      (func.call {:callee "@printF32"} (memref.load out c0 c0))
      (func.call {:callee "@printNewline"})
      (func.call {:callee "@printF32"} (memref.load out c0 c1))
      (func.call {:callee "@printNewline"})
      (func.call {:callee "@printF32"} (memref.load out c1 c2))
      (func.call {:callee "@printNewline"})

      ;; Cleanup
      (gpu.host_unregister inp_unranked)
      (gpu.host_unregister weight_unranked)
      (gpu.host_unregister out_unranked)
      (memref.dealloc inp)
      (memref.dealloc weight)
      (memref.dealloc out)
      (func.return))))