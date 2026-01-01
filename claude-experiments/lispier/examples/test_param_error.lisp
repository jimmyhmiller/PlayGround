;; Minimal test case to reproduce "invalid operand type" error
;; when adding extra buffer parameters to a function
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
    ;; Test function with extra buffer parameters - similar to causal_softmax
    (func.func {:sym_name "test_func"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64xf32>
                                    memref<12x64xf32>] [])}
      (region
        (block [(: out memref<12x64x64xf32>)
                (: inp memref<12x64x64xf32>)
                (: temp memref<12x64x64xf32>)
                (: buf1 memref<12x64xf32>)
                (: buf2 memref<12x64xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c12 (: 12 index))
          (def c64 (: 64 index))

          ;; Simple copy operation
          (scf.for c0 c12 c1
            (region
              (block [(: h index)]
                (scf.for c0 c64 c1
                  (region
                    (block [(: i index)]
                      (scf.for c0 c64 c1
                        (region
                          (block [(: j index)]
                            (def val (memref.load {:result f32} inp h i j))
                            (memref.store val out h i j)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))

          (func.return))))

    (func.func {:sym_name "main" :function_type (-> [] [i32])}
      (region
        (block []
          (def out (memref.alloc {:result memref<12x64x64xf32>}))
          (def inp (memref.alloc {:result memref<12x64x64xf32>}))
          (def temp (memref.alloc {:result memref<12x64x64xf32>}))
          (def buf1 (memref.alloc {:result memref<12x64xf32>}))
          (def buf2 (memref.alloc {:result memref<12x64xf32>}))

          (gpu.host_register (memref.cast {:result "memref<*xf32>"} out))
          (gpu.host_register (memref.cast {:result "memref<*xf32>"} inp))
          (gpu.host_register (memref.cast {:result "memref<*xf32>"} temp))
          (gpu.host_register (memref.cast {:result "memref<*xf32>"} buf1))
          (gpu.host_register (memref.cast {:result "memref<*xf32>"} buf2))

          (func.call "test_func" out inp temp buf1 buf2)

          (print "Test passed\n")
          (def ret (: 0 i32))
          (func.return ret))))))
