;; Test linalg.generic with 3D reduction iterator on GPU
(require-dialect gpu)
(require-dialect func)
(require-dialect memref)
(require-dialect arith)
(require-dialect scf)
(require-dialect linalg)
(require-dialect llvm)

;; External printf declaration
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

;; GPU target with reduction support
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
    (pass convert-math-to-rocdl)
    (pass gpu.module convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
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
    ;; Test: Max over last dimension of 3D tensor (12x64x64 -> 12x64)
    (func.func {:sym_name "max_reduce"
                :function_type (-> [memref<12x64x64xf32> memref<12x64xf32>] [])}
      (region
        (block [(: inp memref<12x64x64xf32>)
                (: out memref<12x64xf32>)]

          (linalg.generic
            {:indexing_maps [affine_map<(d0,d1,d2)->(d0,d1,d2)>
                             affine_map<(d0,d1,d2)->(d0,d1)>]
             :iterator_types ["parallel" "parallel" "reduction"]}
            inp out
            (region
              (block [(: x f32) (: acc f32)]
                (def max_val (arith.maximumf x acc))
                (linalg.yield max_val))))

          (func.return))))

    (func.func {:sym_name "main" :function_type (-> [] [i32])}
      (region
        (block []
          (def inp (memref.alloc {:result memref<12x64x64xf32>}))
          (def out (memref.alloc {:result memref<12x64xf32>}))

          (gpu.host_register (memref.cast {:result "memref<*xf32>"} inp))
          (gpu.host_register (memref.cast {:result "memref<*xf32>"} out))

          ;; Initialize input to 1.0
          (def one (: 1.0 f32))
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c12 (: 12 index))
          (def c64 (: 64 index))

          (scf.for c0 c12 c1
            (region
              (block [(: h index)]
                (scf.for c0 c64 c1
                  (region
                    (block [(: i index)]
                      (scf.for c0 c64 c1
                        (region
                          (block [(: j index)]
                            (memref.store one inp h i j)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize output to -inf
          (def neg_inf (: -1e9 f32))
          (scf.for c0 c12 c1
            (region
              (block [(: h index)]
                (scf.for c0 c64 c1
                  (region
                    (block [(: i index)]
                      (memref.store neg_inf out h i)
                      (scf.yield))))
                (scf.yield))))

          ;; Call max_reduce
          (func.call "max_reduce" inp out)

          ;; Print first result
          (def val (memref.load {:result f32} out c0 c0))
          (def val_f64 (arith.extf {:result f64} val))
          (print "Max value (should be 1.0): %f\n" val_f64)

          (def ret (: 0 i32))
          (func.return ret))))))
