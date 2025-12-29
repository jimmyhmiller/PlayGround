;; Minimal GPU matmul test
;; Trying to isolate the bare pointer / conversion cast issue

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
    ;; Affine lowering before GPU outlining
    (pass lower-affine)
    ;; GPU lowering
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
    ;; LLVM lowering
    (pass convert-scf-to-cf)
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
    ;; C = A * B (4x4 matrices for testing)
    (func.func {:sym_name "matmul_kernel"
                :function_type (-> [memref<4x4xf32> memref<4x4xf32> memref<4x4xf32>] [])}
      (region
        (block [(: A memref<4x4xf32>) (: B memref<4x4xf32>) (: C memref<4x4xf32>)]
          (linalg.matmul A B C)
          (func.return))))

    ;; Main function - minimal GPU test
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate 4x4 matrices on host
          (def A (memref.alloc {:result memref<4x4xf32>}))
          (def B (memref.alloc {:result memref<4x4xf32>}))
          (def C (memref.alloc {:result memref<4x4xf32>}))

          ;; Simple initialization with nested loops (on host)
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
                      (memref.store one A i j)
                      (memref.store one B i j)
                      (memref.store zero C i j)
                      (scf.yield))))
                (scf.yield))))

          ;; GPU path: Register memory
          (def A_dyn (memref.cast {:result "memref<?x?xf32>"} A))
          (def B_dyn (memref.cast {:result "memref<?x?xf32>"} B))
          (def C_dyn (memref.cast {:result "memref<?x?xf32>"} C))

          (def A_unranked (memref.cast {:result "memref<*xf32>"} A_dyn))
          (def B_unranked (memref.cast {:result "memref<*xf32>"} B_dyn))
          (def C_unranked (memref.cast {:result "memref<*xf32>"} C_dyn))

          (gpu.host_register A_unranked)
          (gpu.host_register B_unranked)
          (gpu.host_register C_unranked)

          ;; Get device memrefs
          (def A_dev_dyn (func.call {:result "memref<?x?xf32>"} "mgpuMemGetDeviceMemRef2dFloat" A_dyn))
          (def B_dev_dyn (func.call {:result "memref<?x?xf32>"} "mgpuMemGetDeviceMemRef2dFloat" B_dyn))
          (def C_dev_dyn (func.call {:result "memref<?x?xf32>"} "mgpuMemGetDeviceMemRef2dFloat" C_dyn))

          (def A_dev (memref.cast {:result memref<4x4xf32>} A_dev_dyn))
          (def B_dev (memref.cast {:result memref<4x4xf32>} B_dev_dyn))
          (def C_dev (memref.cast {:result memref<4x4xf32>} C_dev_dyn))

          ;; Run linalg matmul on GPU
          (func.call "matmul_kernel" A_dev B_dev C_dev)

          ;; Print result
          (func.call "printMemrefF32" C_unranked)

          ;; Cleanup
          (memref.dealloc A)
          (memref.dealloc B)
          (memref.dealloc C)

          (func.return))))

    ;; External function declarations
    (func.func {:sym_name "mgpuMemGetDeviceMemRef2dFloat"
                :function_type (-> ["memref<?x?xf32>"] ["memref<?x?xf32>"])
                :sym_visibility "private"})

    (func.func {:sym_name "printMemrefF32"
                :function_type (-> ["memref<*xf32>"] [])
                :sym_visibility "private"})))
