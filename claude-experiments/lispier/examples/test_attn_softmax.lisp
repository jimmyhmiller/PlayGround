;; Test case mimicking attention_forward calling causal_softmax
;; with extra buffer parameters - reproduces the error scenario
(require-dialect gpu)
(require-dialect func)
(require-dialect memref)
(require-dialect arith)
(require-dialect scf)
(require-dialect linalg)
(require-dialect llvm)

(extern-fn printf (-> [!llvm.ptr ...] [i32]))

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
    (pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
    (pass gpu-to-llvm {:use-bare-pointers-for-kernels true})
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-index-to-llvm)
    (pass convert-math-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; causal_softmax with extra buffer parameters
    (func.func {:sym_name "causal_softmax"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64xf32>
                                    memref<12x64xf32>] [])}
      (region
        (block [(: weights memref<12x64x64xf32>)
                (: scores memref<12x64x64xf32>)
                (: masked_scores memref<12x64x64xf32>)
                (: max_buf memref<12x64xf32>)
                (: sum_buf memref<12x64xf32>)]
          (func.return))))

    ;; attention_forward that calls causal_softmax
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

          ;; Call causal_softmax
          (func.call "causal_softmax" weights scores masked_scores max_buf sum_buf)

          (func.return))))

    (func.func {:sym_name "main" :function_type (-> [] [i32])}
      (region
        (block []
          (def out (memref.alloc {:result memref<64x768xf32>}))
          (def qkv (memref.alloc {:result memref<64x2304xf32>}))
          (def Q (memref.alloc {:result memref<12x64x64xf32>}))
          (def K (memref.alloc {:result memref<12x64x64xf32>}))
          (def V (memref.alloc {:result memref<12x64x64xf32>}))
          (def K_t (memref.alloc {:result memref<12x64x64xf32>}))
          (def scores (memref.alloc {:result memref<12x64x64xf32>}))
          (def weights (memref.alloc {:result memref<12x64x64xf32>}))
          (def values (memref.alloc {:result memref<12x64x64xf32>}))
          (def masked_scores (memref.alloc {:result memref<12x64x64xf32>}))
          (def max_buf (memref.alloc {:result memref<12x64xf32>}))
          (def sum_buf (memref.alloc {:result memref<12x64xf32>}))

          (func.call "attention_forward" out qkv Q K V K_t scores weights values
                     masked_scores max_buf sum_buf)

          (print "Test passed\n")
          (def ret (: 0 i32))
          (func.return ret))))))
