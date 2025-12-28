;; GPU Matrix Multiplication Benchmark
;; Phase 1: Validate GPU pipeline with a working matmul kernel
;;
;; Strategy: Each thread computes one element of the output matrix
;; C[i,j] = sum(A[i,k] * B[k,j]) for k in 0..K
;;
;; Matrices: A is MxK, B is KxN, C is MxN

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect scf)

;; Compilation pipeline for AMD GPU
(compilation
  (target rocm
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl)
    (pass gpu-module-to-binary)
    (pass convert-scf-to-cf)
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-func-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass gpu-to-llvm)
    (pass reconcile-unrealized-casts)))

;; Matrix dimensions - 64x64 for initial testing
;; M = rows of A and C
;; K = cols of A, rows of B
;; N = cols of B and C

(module
  (do
    ;; GPU Matrix Multiplication Kernel
    ;; C = A * B
    ;; A: memref<64x64xf32> (MxK)
    ;; B: memref<64x64xf32> (KxN)
    ;; C: memref<64x64xf32> (MxN)
    (func.func {:sym_name "matmul_kernel"
                :function_type (-> [memref<64x64xf32> memref<64x64xf32> memref<64x64xf32>] [])}
      (region
        (block [(: A memref<64x64xf32>) (: B memref<64x64xf32>) (: C memref<64x64xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c8 (: 8 index))

          ;; Launch GPU kernel with 8x8 grid of 8x8 blocks = 64x64 threads
          ;; Each thread computes one element of C
          ;; operandSegmentSizes: [async, gridX/Y/Z, blockX/Y/Z, clusterX/Y/Z, dynamicSharedMem]
          (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
            c8 c8 c1 c8 c8 c1
            (region
              (block [(: bx index) (: by index) (: bz index)
                      (: tx index) (: ty index) (: tz index)
                      (: gridDimX index) (: gridDimY index) (: gridDimZ index)
                      (: blockDimX index) (: blockDimY index) (: blockDimZ index)]
                ;; Compute global row (i) and column (j)
                ;; i = bx * blockDimX + tx
                ;; j = by * blockDimY + ty
                (def i_part (arith.muli bx blockDimX))
                (def i (arith.addi i_part tx))
                (def j_part (arith.muli by blockDimY))
                (def j (arith.addi j_part ty))

                ;; Initialize accumulator
                (def zero (: 0.0 f32))

                ;; Compute dot product: sum over k of A[i,k] * B[k,j]
                ;; Using scf.for loop
                (def result (scf.for {:result f32} c0 c64 c1 zero
                  (region
                    (block [(: k index) (: acc f32)]
                      (def a_val (memref.load {:result f32} A i k))
                      (def b_val (memref.load {:result f32} B k j))
                      (def prod (arith.mulf a_val b_val))
                      (def new_acc (arith.addf acc prod))
                      (scf.yield new_acc)))))

                ;; Store result
                (memref.store result C i j)
                (gpu.terminator))))

          (func.return))))

    ;; CPU reference matmul for validation
    (func.func {:sym_name "matmul_cpu"
                :function_type (-> [memref<64x64xf32> memref<64x64xf32> memref<64x64xf32>] [])}
      (region
        (block [(: A memref<64x64xf32>) (: B memref<64x64xf32>) (: C memref<64x64xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))

          ;; Triple nested loop
          (scf.for c0 c64 c1
            (region
              (block [(: i index)]
                (scf.for c0 c64 c1
                  (region
                    (block [(: j index)]
                      ;; Initialize C[i,j] = 0
                      (def zero (: 0.0 f32))
                      (memref.store zero C i j)

                      ;; Accumulate dot product
                      (scf.for c0 c64 c1
                        (region
                          (block [(: k index)]
                            (def a_val (memref.load {:result f32} A i k))
                            (def b_val (memref.load {:result f32} B k j))
                            (def c_val (memref.load {:result f32} C i j))
                            (def prod (arith.mulf a_val b_val))
                            (def new_c (arith.addf c_val prod))
                            (memref.store new_c C i j)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
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
                      ;; value = (i * 64 + j) * scale
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

    ;; Compare two matrices and return max absolute difference
    (func.func {:sym_name "max_diff"
                :function_type (-> [memref<64x64xf32> memref<64x64xf32>] [f32])}
      (region
        (block [(: A memref<64x64xf32>) (: B memref<64x64xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def zero (: 0.0 f32))

          (def result (scf.for {:result f32} c0 c64 c1 zero
            (region
              (block [(: i index) (: max_outer f32)]
                (def inner_max (scf.for {:result f32} c0 c64 c1 max_outer
                  (region
                    (block [(: j index) (: max_inner f32)]
                      (def a_val (memref.load {:result f32} A i j))
                      (def b_val (memref.load {:result f32} B i j))
                      (def diff (arith.subf a_val b_val))
                      ;; abs(diff) - use select with negative comparison
                      (def neg_diff (arith.negf diff))
                      (def is_neg (arith.cmpf {:predicate "olt"} diff zero))
                      (def abs_diff (arith.select is_neg neg_diff diff))
                      ;; max(max_inner, abs_diff)
                      (def is_greater (arith.cmpf {:predicate "ogt"} abs_diff max_inner))
                      (def new_max (arith.select is_greater abs_diff max_inner))
                      (scf.yield new_max)))))
                (scf.yield inner_max)))))
          (func.return result))))

    ;; Main function - test GPU matmul vs CPU reference
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))

          ;; Allocate matrices
          (def A (memref.alloc {:result memref<64x64xf32>}))
          (def B (memref.alloc {:result memref<64x64xf32>}))
          (def C_gpu (memref.alloc {:result memref<64x64xf32>}))
          (def C_cpu (memref.alloc {:result memref<64x64xf32>}))

          ;; Initialize A and B with small values to avoid overflow
          (def scale_a (: 0.001 f32))
          (def scale_b (: 0.001 f32))
          (func.call "init_matrix" A scale_a)
          (func.call "init_matrix" B scale_b)

          ;; Cast to dynamic memrefs for GPU registration
          (def A_dyn (memref.cast {:result "memref<?x?xf32>"} A))
          (def B_dyn (memref.cast {:result "memref<?x?xf32>"} B))
          (def C_gpu_dyn (memref.cast {:result "memref<?x?xf32>"} C_gpu))

          ;; Cast to unranked for GPU host registration
          (def A_unranked (memref.cast {:result "memref<*xf32>"} A_dyn))
          (def B_unranked (memref.cast {:result "memref<*xf32>"} B_dyn))
          (def C_gpu_unranked (memref.cast {:result "memref<*xf32>"} C_gpu_dyn))

          ;; Register with GPU
          (gpu.host_register A_unranked)
          (gpu.host_register B_unranked)
          (gpu.host_register C_gpu_unranked)

          ;; Get device memrefs
          (def A_dev_dyn (func.call {:result "memref<?x?xf32>"} "mgpuMemGetDeviceMemRef2dFloat" A_dyn))
          (def B_dev_dyn (func.call {:result "memref<?x?xf32>"} "mgpuMemGetDeviceMemRef2dFloat" B_dyn))
          (def C_dev_dyn (func.call {:result "memref<?x?xf32>"} "mgpuMemGetDeviceMemRef2dFloat" C_gpu_dyn))

          ;; Cast back to fixed size for kernel
          (def A_dev (memref.cast {:result memref<64x64xf32>} A_dev_dyn))
          (def B_dev (memref.cast {:result memref<64x64xf32>} B_dev_dyn))
          (def C_dev (memref.cast {:result memref<64x64xf32>} C_dev_dyn))

          ;; Run GPU matmul
          (func.call "matmul_kernel" A_dev B_dev C_dev)

          ;; Run CPU reference
          (func.call "matmul_cpu" A B C_cpu)

          ;; Compare results
          (def diff (func.call {:result f32} "max_diff" C_gpu C_cpu))

          ;; Print result
          (func.call "printMemrefF32" C_gpu_unranked)

          ;; Clean up
          (memref.dealloc A)
          (memref.dealloc B)
          (memref.dealloc C_gpu)
          (memref.dealloc C_cpu)

          (func.return))))

    ;; External function declarations
    (func.func {:sym_name "mgpuMemGetDeviceMemRef2dFloat"
                :function_type (-> ["memref<?x?xf32>"] ["memref<?x?xf32>"])
                :sym_visibility "private"})

    (func.func {:sym_name "printMemrefF32"
                :function_type (-> ["memref<*xf32>"] [])
                :sym_visibility "private"})))
