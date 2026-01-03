;; GPU Matrix Multiplication via Affine Loop Tiling
;;
;; Experiment: Test the affine loop path from Stephen Diehl's blog post
;; https://www.stephendiehl.com/posts/mlir_linear_algebra/
;;
;; This version uses:
;; 1. convert-linalg-to-affine-loops (instead of parallel loops)
;; 2. affine-loop-tile for cache-efficient tiling
;; 3. affine-parallelize to enable GPU mapping
;;
;; Hypothesis: Affine dialect has richer loop analysis that may produce
;; better tiled code than the direct parallel loop path.

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect linalg)
(require-dialect scf)
(require-dialect affine)

;; Compilation pipeline for AMD GPU via affine loop tiling
;; Key difference from matmul_parallel.lisp:
;; - Uses affine loops + tiling instead of direct parallel loops
;; Pipeline: linalg → affine → tiled affine → affine.parallel → scf.parallel → GPU
(compilation
  (target rocm
    ;; Step 1: Convert linalg to affine loops (blog approach)
    (pass convert-linalg-to-affine-loops)

    ;; Step 2: Canonicalize before parallelize
    (pass canonicalize)

    ;; Step 3: Convert to affine.parallel loops
    (pass affine-parallelize)

    ;; Step 4: Lower affine.parallel to scf.parallel
    (pass lower-affine)

    ;; Step 5: Tile the scf.parallel loops (same as working parallel version)
    (pass scf-parallel-loop-tiling {:parallel-loop-tile-sizes "16,16"})

    ;; Step 6: Map scf.parallel to GPU grid/blocks
    (pass gpu-map-parallel-loops)

    ;; Step 7: Convert parallel loops to gpu.launch
    (pass convert-parallel-loops-to-gpu)

    ;; Step 8: SCF to CF before GPU outlining
    (pass convert-scf-to-cf)

    ;; Step 9: GPU lowering (no bare-ptr - matches working GPT-2 config)
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl)
    (pass gpu-module-to-binary)

    ;; Step 10: Host-side LLVM lowering
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
    ;; C = A * B (64x64 matrices)
    (func.func {:sym_name "matmul_affine"
                :function_type (-> [memref<64x64xf32> memref<64x64xf32> memref<64x64xf32>] [])}
      (region
        (block [(: A memref<64x64xf32>) (: B memref<64x64xf32>) (: C memref<64x64xf32>)]
          ;; Zero output for accumulation
          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero C
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))

          ;; Core matmul using linalg.generic - will be converted to affine loops, then tiled
          ;; C[i,j] = sum_k A[i,k] * B[k,j]
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

    ;; Zero initialize a matrix using linalg.fill
    (func.func {:sym_name "zero_matrix"
                :function_type (-> [memref<64x64xf32>] [])}
      (region
        (block [(: M memref<64x64xf32>)]
          (def zero (: 0.0 f32))
          (linalg.fill {:ins 1 :outs 1} zero M
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))
          (func.return))))

    ;; Main function - test affine tiled matmul
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
          (func.call {:callee "@zero_matrix"} C)

          ;; Cast for GPU registration
          (def A_unranked (memref.cast {:result "memref<*xf32>"} A))
          (def B_unranked (memref.cast {:result "memref<*xf32>"} B))
          (def C_unranked (memref.cast {:result "memref<*xf32>"} C))

          ;; Register with GPU - host memory becomes accessible to GPU
          (gpu.host_register A_unranked)
          (gpu.host_register B_unranked)
          (gpu.host_register C_unranked)

          ;; Run affine-tiled matmul - pass host memrefs directly
          ;; The linalg->GPU path handles memory automatically
          (func.call {:callee "@matmul_affine"} A B C)

          ;; Print result
          (func.call {:callee "@printMemrefF32"} C_unranked)

          ;; Cleanup
          (memref.dealloc A)
          (memref.dealloc B)
          (memref.dealloc C)

          (func.return))))

    ;; External function declaration
    (func.func {:sym_name "printMemrefF32"
                :function_type (-> ["memref<*xf32>"] [])
                :sym_visibility "private"}
      (region))))