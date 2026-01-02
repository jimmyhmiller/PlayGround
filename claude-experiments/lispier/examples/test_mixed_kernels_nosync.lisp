;; Test mixing linalg-derived kernels with explicit gpu.launch WITHOUT sync
;; Same as test_mixed_kernels but WITHOUT the print between matmul and gpu.launch

(require-dialect memref)
(require-dialect arith)
(require-dialect func)
(require-dialect gpu)
(require-dialect scf)
(require-dialect math)
(require-dialect linalg)

;; Use GPT-2's compilation pipeline
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
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"}
      (region))

    (defn main []
      (def c0 (: 0 index))
      (def c1 (: 1 index))
      (def c2 (: 2 index))
      (def c4 (: 4 index))
      (def c32 (: 32 index))
      (def c8 (: 8 index))

      ;; Allocate matrices for batch matmul
      (def A (memref.alloc {:result memref<2x4x4xf32>}))
      (def B (memref.alloc {:result memref<2x4x4xf32>}))
      (def C (memref.alloc {:result memref<2x4x4xf32>}))
      (def out (memref.alloc {:result memref<2x4x4xf32>}))

      (def one (: 1.0 f32))
      (def zero (: 0.0 f32))

      ;; Initialize A with 1.0
      (linalg.fill {:ins 1 :outs 1} one A
        (region
          (block [(: in f32) (: _out f32)]
            (linalg.yield in))))

      ;; Initialize B with 0.25
      (def quarter (: 0.25 f32))
      (linalg.fill {:ins 1 :outs 1} quarter B
        (region
          (block [(: in f32) (: _out f32)]
            (linalg.yield in))))

      ;; Zero C
      (linalg.fill {:ins 1 :outs 1} zero C
        (region
          (block [(: in f32) (: _out f32)]
            (linalg.yield in))))

      ;; Register for GPU
      (def A_unranked (memref.cast {:result "memref<*xf32>"} A))
      (def B_unranked (memref.cast {:result "memref<*xf32>"} B))
      (def C_unranked (memref.cast {:result "memref<*xf32>"} C))
      (def out_unranked (memref.cast {:result "memref<*xf32>"} out))
      (gpu.host_register A_unranked)
      (gpu.host_register B_unranked)
      (gpu.host_register C_unranked)
      (gpu.host_register out_unranked)

      ;; Step 1: Batch matmul - C[b] = A[b] @ B[b]
      (linalg.batch_matmul {:ins 2 :outs 1} A B C
        (region
          (block [(: a f32) (: b f32) (: c f32)]
            (def prod (arith.mulf a b))
            (def sum (arith.addf c prod))
            (linalg.yield sum))))

      ;; NO PRINT HERE - go straight to GPU launch

      ;; Step 2: Explicit gpu.launch for softmax on C -> out
      (def neg_inf (: -1e30 f32))
      (def c16_i32 (: 16 i32))
      (def c8_i32 (: 8 i32))
      (def c4_i32 (: 4 i32))
      (def c2_i32 (: 2 i32))
      (def c1_i32 (: 1 i32))
      (def c32_i32 (: 32 i32))

      (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
        c8 c1 c1 c32 c1 c1
        (region
          (block [(: block_id index) (: _by index) (: _bz index)
                  (: lane index) (: _ty index) (: _tz index)
                  (: _gridDimX index) (: _gridDimY index) (: _gridDimZ index)
                  (: _blockDimX index) (: _blockDimY index) (: _blockDimZ index)]

            (def batch (arith.divui block_id c4))
            (def row (arith.remui block_id c4))

            (def is_active (arith.cmpi {:predicate 2} lane c4))

            (scf.if is_active
              (region
                (block []
                  (def val (memref.load {:result f32} C batch row lane))

                  (def m2 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} val c2_i32 c32_i32))
                  (def max2 (arith.maximumf val m2))
                  (def m1 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} max2 c1_i32 c32_i32))
                  (def global_max (arith.maximumf max2 m1))

                  (def shifted (arith.subf val global_max))
                  (def exp_val (math.exp shifted))

                  (def s2 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} exp_val c2_i32 c32_i32))
                  (def sum2 (arith.addf exp_val s2))
                  (def s1 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum2 c1_i32 c32_i32))
                  (def total_sum (arith.addf sum2 s1))

                  (def scale (arith.divf one total_sum))
                  (def softmax_val (arith.mulf exp_val scale))

                  (memref.store softmax_val out batch row lane)
                  (scf.yield)))
              (region
                (block []
                  (scf.yield))))

            (gpu.terminator))))

      ;; Print results
      (func.call {:callee "@printF32"} (memref.load out c0 c0 c0))
      (func.call {:callee "@printNewline"})
      (func.call {:callee "@printF32"} (memref.load out c0 c0 c1))
      (func.call {:callee "@printNewline"})
      (func.call {:callee "@printF32"} (memref.load out c1 c1 c1))
      (func.call {:callee "@printNewline"})

      ;; Cleanup
      (gpu.host_unregister A_unranked)
      (gpu.host_unregister B_unranked)
      (gpu.host_unregister C_unranked)
      (gpu.host_unregister out_unranked)
      (memref.dealloc A)
      (memref.dealloc B)
      (memref.dealloc C)
      (memref.dealloc out)
      (func.return))))