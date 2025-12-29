;; Simple GPU matmul test using gpu.alloc/memcpy
;; Avoids the missing mgpuMemGetDeviceMemRef2dFloat issue

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
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-index-to-llvm)
    (pass convert-func-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass gpu-to-llvm {:use-bare-pointers-for-kernels true})
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; Matrix multiplication using linalg
    ;; C = A * B (4x4 matrices)
    (func.func {:sym_name "matmul_kernel"
                :function_type (-> [memref<4x4xf32> memref<4x4xf32> memref<4x4xf32>] [])}
      (region
        (block [(: A memref<4x4xf32>) (: B memref<4x4xf32>) (: C memref<4x4xf32>)]
          (linalg.matmul A B C)
          (func.return))))

    ;; Main function - GPU matmul using gpu.alloc/memcpy
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate host matrices
          (def A_host (memref.alloc {:result memref<4x4xf32>}))
          (def B_host (memref.alloc {:result memref<4x4xf32>}))
          (def C_host (memref.alloc {:result memref<4x4xf32>}))

          ;; Simple initialization
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c4 (: 4 index))
          (def one (: 1.0 f32))
          (def zero (: 0.0 f32))

          ;; Init A and B to 1.0, C to 0.0
          (scf.for c0 c4 c1
            (region
              (block [(: i index)]
                (scf.for c0 c4 c1
                  (region
                    (block [(: j index)]
                      (memref.store one A_host i j)
                      (memref.store one B_host i j)
                      (memref.store zero C_host i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Allocate device memory
          (def A_dev (gpu.alloc {:result memref<4x4xf32>}))
          (def B_dev (gpu.alloc {:result memref<4x4xf32>}))
          (def C_dev (gpu.alloc {:result memref<4x4xf32>}))

          ;; Copy to device
          (gpu.memcpy A_dev A_host)
          (gpu.memcpy B_dev B_host)
          (gpu.memcpy C_dev C_host)

          ;; Run kernel
          (func.call "matmul_kernel" A_dev B_dev C_dev)

          ;; Copy result back
          (gpu.memcpy C_host C_dev)

          ;; Print a sample value (C[0][0] should be 4.0 for 4x4 all-ones matmul)
          (def result (memref.load C_host c0 c0))
          (func.call "printF32" result)

          ;; Cleanup GPU memory
          (gpu.dealloc A_dev)
          (gpu.dealloc B_dev)
          (gpu.dealloc C_dev)

          ;; Cleanup host memory
          (memref.dealloc A_host)
          (memref.dealloc B_host)
          (memref.dealloc C_host)

          (func.return))))

    ;; External function declaration for printing
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"})))