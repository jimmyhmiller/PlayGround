;; Test linalg.generic with reduction iterator on GPU
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
    ;; Test: Sum each row of a 64x768 matrix (like layernorm mean)
    (func.func {:sym_name "row_sum"
                :function_type (-> [memref<64x768xf32> memref<64xf32>] [])}
      (region
        (block [(: inp memref<64x768xf32>)
                (: out memref<64xf32>)]

          ;; Use linalg.generic with reduction to sum each row
          ;; parallel over d0 (rows), reduction over d1 (columns)
          (linalg.generic
            {:indexing_maps [affine_map<(d0,d1)->(d0,d1)>
                             affine_map<(d0,d1)->(d0)>]
             :iterator_types ["parallel" "reduction"]}
            inp out
            (region
              (block [(: x f32) (: acc f32)]
                (def sum (arith.addf x acc))
                (linalg.yield sum))))

          (func.return))))

    (func.func {:sym_name "main" :function_type (-> [] [i32])}
      (region
        (block []
          ;; Allocate input and output
          (def inp (memref.alloc {:result memref<64x768xf32>}))
          (def out (memref.alloc {:result memref<64xf32>}))

          ;; Register with GPU
          (gpu.host_register (memref.cast {:result "memref<*xf32>"} inp))
          (gpu.host_register (memref.cast {:result "memref<*xf32>"} out))

          ;; Initialize input to 1.0
          (def one (: 1.0 f32))
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))

          (scf.for c0 c64 c1
            (region
              (block [(: i index)]
                (scf.for c0 c768 c1
                  (region
                    (block [(: j index)]
                      (memref.store one inp i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize output to 0.0
          (def zero (: 0.0 f32))
          (scf.for c0 c64 c1
            (region
              (block [(: i index)]
                (memref.store zero out i)
                (scf.yield))))

          ;; Call row_sum
          (func.call "row_sum" inp out)

          ;; Print first few results (should be 768.0 for each row)
          (def c4 (: 4 index))
          (scf.for c0 c4 c1
            (region
              (block [(: i index)]
                (def val (memref.load {:result f32} out i))
                (def i_i64 (arith.index_cast {:result i64} i))
                (def val_f64 (arith.extf {:result f64} val))
                (print "Row %ld sum: %f (expected 768.0)\n" i_i64 val_f64)
                (scf.yield))))

          ;; Return 0
          (def ret (: 0 i32))
          (func.return ret))))))
