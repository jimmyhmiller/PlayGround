;; GPU matmul test using gpu.host_register only
;; This avoids the gpu.alloc/memcpy/dealloc which aren't being lowered properly

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
    ;; Matrix multiplication using linalg - operates on host-registered memory
    ;; 64x64 matrices for real GPU usage
    (func.func {:sym_name "matmul_kernel"
                :function_type (-> [memref<64x64xf32> memref<64x64xf32> memref<64x64xf32>] [])}
      (region
        (block [(: A memref<64x64xf32>) (: B memref<64x64xf32>) (: C memref<64x64xf32>)]
          (linalg.matmul A B C)
          (func.return))))

    ;; Main function
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate host matrices
          (def A (memref.alloc {:result memref<64x64xf32>}))
          (def B (memref.alloc {:result memref<64x64xf32>}))
          (def C (memref.alloc {:result memref<64x64xf32>}))

          ;; Initialize
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def one (: 1.0 f32))
          (def zero (: 0.0 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: i index)]
                (scf.for c0 c64 c1
                  (region
                    (block [(: j index)]
                      (memref.store one A i j)
                      (memref.store one B i j)
                      (memref.store zero C i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Cast for GPU registration
          (def A_unranked (memref.cast {:result "memref<*xf32>"} A))
          (def B_unranked (memref.cast {:result "memref<*xf32>"} B))
          (def C_unranked (memref.cast {:result "memref<*xf32>"} C))

          ;; Register with GPU - makes memory accessible from device
          (gpu.host_register A_unranked)
          (gpu.host_register B_unranked)
          (gpu.host_register C_unranked)

          ;; Run kernel directly on registered memory
          ;; The GPU runtime will handle the coherency
          (func.call "matmul_kernel" A B C)

          ;; Print result - C[0][0] should be 64.0 for 64x64 all-ones matmul
          (def result (memref.load C c0 c0))
          (func.call "printF32" result)
          (func.call "printNewline")

          ;; Unregister
          (gpu.host_unregister A_unranked)
          (gpu.host_unregister B_unranked)
          (gpu.host_unregister C_unranked)

          ;; Cleanup
          (memref.dealloc A)
          (memref.dealloc B)
          (memref.dealloc C)

          (func.return))))

    ;; External function declarations
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"})

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"})))