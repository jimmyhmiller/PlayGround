;; Basic GPU launch test (no shuffle) - with printF32 output
(require-dialect memref)
(require-dialect arith)
(require-dialect func)
(require-dialect gpu)
(require-dialect scf)

(compilation
  (target rocm
    ;; Affine lowering first
    (pass lower-affine)
    ;; SCF to CF before GPU outlining
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
    ;; External function declarations for printing
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"}
      (region))

    (defn main []
      ;; Allocate output buffer - initialize to -1 to verify kernel writes
      (def out (memref.alloc {:result memref<32xf32>}))
      (def out_unranked (memref.cast {:result "memref<*xf32>"} out))
      (gpu.host_register out_unranked)

      ;; Constants
      (def c0 (: 0 index))
      (def c1 (: 1 index))
      (def c4 (: 4 index))
      (def c32 (: 32 index))
      (def neg_one (: -1.0 f32))

      ;; Initialize to -1
      (scf.for c0 c32 c1
        (region
          (block [(: i index)]
            (memref.store neg_one out i)
            (scf.yield))))

      ;; Launch 1 block of 32 threads - each writes its thread id
      (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
        c1 c1 c1 c32 c1 c1
        (region
          (block [(: bx index) (: by index) (: bz index)
                  (: tx index) (: ty index) (: tz index)
                  (: gridDimX index) (: gridDimY index) (: gridDimZ index)
                  (: blockDimX index) (: blockDimY index) (: blockDimZ index)]
            ;; Each thread writes its lane id
            (def lane_i64 (arith.index_cast {:result i64} tx))
            (def val (arith.sitofp {:result f32} lane_i64))
            (memref.store val out tx)
            (gpu.terminator))))

      ;; Print first 4 results to verify
      ;; Expected: 0.0, 1.0, 2.0, 3.0
      (scf.for c0 c4 c1
        (region
          (block [(: i index)]
            (def v (memref.load out i))
            (func.call {:callee "@printF32"} v)
            (func.call {:callee "@printNewline"})
            (scf.yield))))

      ;; Unregister and cleanup
      (gpu.host_unregister out_unranked)
      (memref.dealloc out)
      (func.return))))
