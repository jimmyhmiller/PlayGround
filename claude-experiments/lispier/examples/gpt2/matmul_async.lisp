;; GPU Matrix Multiplication using Async Pattern
;; Uses gpu.alloc async, gpu.memcpy async, gpu.launch_func async
;;
;; This pattern works with the AMD GPU runtime (no mgpuMemGetDeviceMemRef2dFloat needed)

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect scf)
(require-dialect vector)

(module
  (do
    ;; GPU Matrix Multiplication Kernel (outlined to gpu.module)
    (gpu.module {:sym_name "kernels"}
      (do
        (gpu.func {:sym_name "matmul_kernel" :kernel ""}
          (: A memref<64x64xf32>) (: B memref<64x64xf32>) (: C memref<64x64xf32>)
          (region
            (block []
              (def bx (gpu.block_id {:dimension "x" :result index}))
              (def by (gpu.block_id {:dimension "y" :result index}))
              (def tx (gpu.thread_id {:dimension "x" :result index}))
              (def ty (gpu.thread_id {:dimension "y" :result index}))
              (def bdimx (gpu.block_dim {:dimension "x" :result index}))
              (def bdimy (gpu.block_dim {:dimension "y" :result index}))

              ;; Compute global indices
              (def i_part (arith.muli bx bdimx))
              (def i (arith.addi i_part tx))
              (def j_part (arith.muli by bdimy))
              (def j (arith.addi j_part ty))

              (def c0 (: 0 index))
              (def c1 (: 1 index))
              (def c64 (: 64 index))
              (def zero (: 0.0 f32))

              ;; Compute dot product
              (def result (scf.for {:result f32} c0 c64 c1 zero
                (region
                  (block [(: k index) (: acc f32)]
                    (def a_val (memref.load {:result f32} A i k))
                    (def b_val (memref.load {:result f32} B k j))
                    (def prod (arith.mulf a_val b_val))
                    (def new_acc (arith.addf acc prod))
                    (scf.yield new_acc)))))

              (memref.store result C i j)
              (gpu.return))))))

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

    ;; Main function - test GPU matmul
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c8 (: 8 index))

          ;; Host matrices
          (def hostA (memref.alloc {:result memref<64x64xf32>}))
          (def hostB (memref.alloc {:result memref<64x64xf32>}))
          (def hostC (memref.alloc {:result memref<64x64xf32>}))

          ;; Initialize host matrices
          (def scale_a (: 0.001 f32))
          (def scale_b (: 0.001 f32))
          (func.call "init_matrix" hostA scale_a)
          (func.call "init_matrix" hostB scale_b)

          ;; Async GPU operations
          ;; 1. Allocate GPU memory
          (def gpuA_tok1 (gpu.alloc {:operandSegmentSizes array<i32: 1, 0, 0> :result_1 "!gpu.async.token"}))
          (def gpuA (first gpuA_tok1))
          (def tok1 (second gpuA_tok1))

          (def gpuB_tok2 (gpu.alloc {:operandSegmentSizes array<i32: 1, 0, 0> :asyncDependencies tok1 :result_1 "!gpu.async.token"}))
          (def gpuB (first gpuB_tok2))
          (def tok2 (second gpuB_tok2))

          (def gpuC_tok3 (gpu.alloc {:operandSegmentSizes array<i32: 1, 0, 0> :asyncDependencies tok2 :result_1 "!gpu.async.token"}))
          (def gpuC (first gpuC_tok3))
          (def tok3 (second gpuC_tok3))

          ;; 2. Copy host to GPU
          (def tok4 (gpu.memcpy {:asyncDependencies tok3} gpuA hostA))
          (def tok5 (gpu.memcpy {:asyncDependencies tok4} gpuB hostB))

          ;; 3. Launch kernel: 8x8 blocks of 8x8 threads = 64x64 total
          (def tok6 (gpu.launch_func {:asyncDependencies tok5 :kernel "matmul_kernel"}
            "kernels" c8 c8 c1 c8 c8 c1 gpuA gpuB gpuC))

          ;; 4. Copy result back
          (def tok7 (gpu.memcpy {:asyncDependencies tok6} hostC gpuC))

          ;; 5. Wait for completion
          (gpu.wait tok7)

          ;; Print first element to verify
          (def val (memref.load {:result f32} hostC c0 c0))
          (vector.print val)

          ;; Dealloc
          (gpu.dealloc gpuA)
          (gpu.dealloc gpuB)
          (gpu.dealloc gpuC)
          (memref.dealloc hostA)
          (memref.dealloc hostB)
          (memref.dealloc hostC)

          (func.return))))))
