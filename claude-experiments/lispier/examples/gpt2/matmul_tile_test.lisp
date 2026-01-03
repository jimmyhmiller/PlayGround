;; GPU Matrix Multiplication - Tile Size Experiment
;;
;; Test different tile sizes with the parallel loop path
;; Baseline uses 16x16, let's try 32x32

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect linalg)
(require-dialect scf)

;; Compilation pipeline - same as GPT-2 but with 32x32 tile size
(compilation
  (target rocm
    (pass convert-linalg-to-parallel-loops)
    ;; Try 32x32 tiles instead of 16x16
    (pass scf-parallel-loop-tiling {:parallel-loop-tile-sizes "32,32"})
    (pass gpu-map-parallel-loops)
    (pass convert-parallel-loops-to-gpu)
    (pass lower-affine)
    (pass convert-scf-to-cf)
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl)
    (pass gpu-module-to-binary)
    (pass gpu-to-llvm)
    (pass expand-strided-metadata)
    (pass lower-affine)
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-index-to-llvm)
    (pass convert-math-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; Matrix multiplication using linalg.generic
    ;; C = A * B (64x64 matrices for testing, GPT-2 uses larger)
    (func.func {:sym_name "matmul_test"
                :function_type (-> [memref<64x64xf32> memref<64x64xf32> memref<64x64xf32>] [])}
      (region
        (block [(: A memref<64x64xf32>) (: B memref<64x64xf32>) (: C memref<64x64xf32>)]
          ;; Zero output for accumulation
          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero C
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))

          ;; Core matmul using linalg.generic
          (linalg.generic
            {:ins 2 :outs 1
             :indexing_maps [affine_map<(d0,d1,d2)->(d0,d2)>
                             affine_map<(d0,d1,d2)->(d2,d1)>
                             affine_map<(d0,d1,d2)->(d0,d1)>]
             :iterator_types ["#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<parallel>"
                              "#linalg.iterator_type<reduction>"]}
            A B C
            (region
              (block [(: a f32) (: b f32) (: c f32)]
                (def mul (arith.mulf a b))
                (def sum (arith.addf c mul))
                (linalg.yield sum))))

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

    ;; Main function
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
          (func.call {:callee "@init_matrix"} A scale_a)
          (func.call {:callee "@init_matrix"} B scale_b)

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
          (def A_dev_dyn (func.call {:callee "@mgpuMemGetDeviceMemRef2dFloat" :result "memref<?x?xf32>"} A_dyn))
          (def B_dev_dyn (func.call {:callee "@mgpuMemGetDeviceMemRef2dFloat" :result "memref<?x?xf32>"} B_dyn))
          (def C_dev_dyn (func.call {:callee "@mgpuMemGetDeviceMemRef2dFloat" :result "memref<?x?xf32>"} C_dyn))

          (def A_dev (memref.cast {:result memref<64x64xf32>} A_dev_dyn))
          (def B_dev (memref.cast {:result memref<64x64xf32>} B_dev_dyn))
          (def C_dev (memref.cast {:result memref<64x64xf32>} C_dev_dyn))

          ;; Run matmul
          (func.call {:callee "@matmul_test"} A_dev B_dev C_dev)

          ;; Print result
          (func.call {:callee "@printMemrefF32"} C_unranked)

          ;; Cleanup
          (memref.dealloc A)
          (memref.dealloc B)
          (memref.dealloc C)

          (func.return))))

    ;; External function declarations
    (func.func {:sym_name "mgpuMemGetDeviceMemRef2dFloat"
                :function_type (-> ["memref<?x?xf32>"] ["memref<?x?xf32>"])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "printMemrefF32"
                :function_type (-> ["memref<*xf32>"] [])
                :sym_visibility "private"}
      (region))))