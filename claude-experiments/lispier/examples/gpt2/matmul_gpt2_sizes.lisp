;; Test GPU matmul with GPT-2 sized matrices
;; Using gpu.host_register to make memory accessible to GPU

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect linalg)
(require-dialect scf)

;; Compilation pipeline for AMD GPU via linalg lowering
(compilation
  (target rocm
    ;; Linalg to parallel loops, then to GPU
    (pass convert-linalg-to-parallel-loops)
    (pass gpu-map-parallel-loops)
    (pass convert-parallel-loops-to-gpu)
    ;; Affine lowering
    (pass lower-affine)
    ;; CRITICAL: SCF to CF BEFORE GPU outlining
    (pass convert-scf-to-cf)
    ;; GPU lowering
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
    ;; Host-side LLVM lowering
    (pass gpu-to-llvm {:use-bare-pointers-for-kernels true})
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-index-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; QKV matmul kernel: (64,768) @ (768,2304) -> (64,2304)
    (func.func {:sym_name "matmul_qkv_kernel"
                :function_type (-> [memref<64x2304xf32> memref<64x768xf32> memref<768x2304xf32>] [])}
      (region
        (block [(: out memref<64x2304xf32>) (: inp memref<64x768xf32>) (: weight memref<768x2304xf32>)]
          (linalg.matmul inp weight out)
          (func.return))))

    ;; Main function
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate host matrices
          (def inp (memref.alloc {:result memref<64x768xf32>}))
          (def weight (memref.alloc {:result memref<768x2304xf32>}))
          (def out (memref.alloc {:result memref<64x2304xf32>}))

          ;; Initialize with small values
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def c2304 (: 2304 index))
          (def one (: 1.0 f32))
          (def zero (: 0.0 f32))
          (def small (: 0.001 f32))

          ;; Initialize input to 1.0
          (scf.for c0 c64 c1
            (region
              (block [(: i index)]
                (scf.for c0 c768 c1
                  (region
                    (block [(: j index)]
                      (memref.store one inp i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize weight to 0.001
          (scf.for c0 c768 c1
            (region
              (block [(: i index)]
                (scf.for c0 c2304 c1
                  (region
                    (block [(: j index)]
                      (memref.store small weight i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize output to 0
          (scf.for c0 c64 c1
            (region
              (block [(: i index)]
                (scf.for c0 c2304 c1
                  (region
                    (block [(: j index)]
                      (memref.store zero out i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Cast for GPU registration
          (def inp_unranked (memref.cast {:result "memref<*xf32>"} inp))
          (def weight_unranked (memref.cast {:result "memref<*xf32>"} weight))
          (def out_unranked (memref.cast {:result "memref<*xf32>"} out))

          ;; Register with GPU - makes memory accessible from device
          (gpu.host_register inp_unranked)
          (gpu.host_register weight_unranked)
          (gpu.host_register out_unranked)

          ;; Run kernel
          (func.call "matmul_qkv_kernel" out inp weight)

          ;; Print result - out[0][0] should be 768 * 0.001 = 0.768
          (def result (memref.load out c0 c0))
          (func.call "printF32" result)
          (func.call "printNewline")

          ;; Unregister
          (gpu.host_unregister inp_unranked)
          (gpu.host_unregister weight_unranked)
          (gpu.host_unregister out_unranked)

          ;; Cleanup
          (memref.dealloc inp)
          (memref.dealloc weight)
          (memref.dealloc out)

          (func.return))))

    ;; External function declarations
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"})

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"})))
