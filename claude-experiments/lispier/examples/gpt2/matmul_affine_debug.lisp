;; Minimal test: Affine path without GPU to see what we get

(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect linalg)
(require-dialect scf)
(require-dialect affine)

;; Just affine passes - no GPU
(compilation
  (target cpu
    (pass convert-linalg-to-affine-loops)
    (pass canonicalize)
    (pass affine-parallelize)
    (pass lower-affine)
    (pass canonicalize)
    (pass convert-scf-to-cf)
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-index-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    (func.func {:sym_name "matmul_test"
                :function_type (-> [memref<64x64xf32> memref<64x64xf32> memref<64x64xf32>] [])}
      (region
        (block [(: A memref<64x64xf32>) (: B memref<64x64xf32>) (: C memref<64x64xf32>)]
          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero C
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))

          (linalg.generic
            {:ins 2 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2)->(d0,d2)>
                             affine_map<(d0,d1,d2)->(d2,d1)>
                             affine_map<(d0,d1,d2)->(d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<reduction>"]}
            A B C
            (region
              (block [(: a f32) (: b f32) (: c f32)]
                (def mul (arith.mulf a b))
                (def sum (arith.addf c mul))
                (linalg.yield sum))))

          (func.return))))

    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          (def A (memref.alloc {:result memref<64x64xf32>}))
          (def B (memref.alloc {:result memref<64x64xf32>}))
          (def C (memref.alloc {:result memref<64x64xf32>}))

          (func.call {:callee "@matmul_test"} A B C)

          (memref.dealloc A)
          (memref.dealloc B)
          (memref.dealloc C)

          (func.return))))))