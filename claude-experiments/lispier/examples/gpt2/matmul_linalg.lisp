;; GPU Matrix Multiplication via Linalg
;; Phase 2: Use linalg.matmul and let MLIR handle GPU lowering
;;
;; This version uses linalg.matmul which can be automatically:
;; 1. Tiled for cache efficiency
;; 2. Parallelized to parallel loops
;; 3. Mapped to GPU via convert-parallel-loops-to-gpu
;;
;; This is the preferred approach vs manual gpu.launch kernels

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect linalg)
(require-dialect scf)

;; Compilation pipeline for AMD GPU via linalg lowering
;; Key insight: convert-scf-to-cf MUST happen before gpu-kernel-outlining
;; so that no scf.for loops remain inside GPU kernels
(compilation
  (target rocm
    ;; Linalg to parallel loops, then to GPU
    (pass convert-linalg-to-parallel-loops)
    (pass gpu-map-parallel-loops)
    (pass convert-parallel-loops-to-gpu)
    ;; Affine lowering
    (pass lower-affine)
    ;; CRITICAL: SCF to CF BEFORE GPU outlining (eliminates scf.for in kernels)
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
    ;; C = A * B (64x64 matrices)
    (func.func {:sym_name "matmul_linalg"
                :function_type (-> [memref<64x64xf32> memref<64x64xf32> memref<64x64xf32>] [])}
      (region
        (block [(: A memref<64x64xf32>) (: B memref<64x64xf32>) (: C memref<64x64xf32>)]
          ;; Simple and clean - linalg.matmul handles everything
          (linalg.matmul A B C)
          (func.return))))

    ;; Initialize matrix with sequential values
    (func.func {:sym_name "init_matrix"
                :function_type (-> [memref<64x64xf32> f32] [])}
      (region
        (block [(: M memref<64x64xf32>) (: scale f32)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))

          (scf.for c0 c64 c1
            (region
              (block [(: i index)]
                (scf.for c0 c64 c1
                  (region
                    (block [(: j index)]
                      (def i_i64 (arith.index_cast {:result i64} i))
                      (def j_i64 (arith.index_cast {:result i64} j))
                      (def c64_i64 (: 64 i64))
                      (def idx_part (arith.muli i_i64 c64_i64))
                      (def idx (arith.addi idx_part j_i64))
                      (def idx_f32 (arith.sitofp {:result f32} idx))
                      (def val (arith.mulf idx_f32 scale))
                      (memref.store val M i j)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; Zero initialize a matrix using linalg.fill
    (func.func {:sym_name "zero_matrix"
                :function_type (-> [memref<64x64xf32>] [])}
      (region
        (block [(: M memref<64x64xf32>)]
          (def zero (: 0.0 f32))
          (linalg.fill zero M)
          (func.return))))

    ;; Main function - test linalg matmul
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate matrices
          (def A (memref.alloc {:result memref<64x64xf32>}))
          (def B (memref.alloc {:result memref<64x64xf32>}))
          (def C (memref.alloc {:result memref<64x64xf32>}))

          ;; Initialize
          (def scale_a (: 0.001 f32))
          (def scale_b (: 0.001 f32))
          (func.call "init_matrix" A scale_a)
          (func.call "init_matrix" B scale_b)
          (func.call "zero_matrix" C)

          ;; Cast for GPU registration
          (def A_dyn (memref.cast {:result "memref<?x?xf32>"} A))
          (def B_dyn (memref.cast {:result "memref<?x?xf32>"} B))
          (def C_dyn (memref.cast {:result "memref<?x?xf32>"} C))

          (def A_unranked (memref.cast {:result "memref<*xf32>"} A_dyn))
          (def B_unranked (memref.cast {:result "memref<*xf32>"} B_dyn))
          (def C_unranked (memref.cast {:result "memref<*xf32>"} C_dyn))

          ;; Register with GPU
          (gpu.host_register A_unranked)
          (gpu.host_register B_unranked)
          (gpu.host_register C_unranked)

          ;; Get device memrefs
          (def A_dev_dyn (func.call {:result "memref<?x?xf32>"} "mgpuMemGetDeviceMemRef2dFloat" A_dyn))
          (def B_dev_dyn (func.call {:result "memref<?x?xf32>"} "mgpuMemGetDeviceMemRef2dFloat" B_dyn))
          (def C_dev_dyn (func.call {:result "memref<?x?xf32>"} "mgpuMemGetDeviceMemRef2dFloat" C_dyn))

          (def A_dev (memref.cast {:result memref<64x64xf32>} A_dev_dyn))
          (def B_dev (memref.cast {:result memref<64x64xf32>} B_dev_dyn))
          (def C_dev (memref.cast {:result memref<64x64xf32>} C_dev_dyn))

          ;; Run linalg matmul (will be lowered to GPU automatically)
          (func.call "matmul_linalg" A_dev B_dev C_dev)

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