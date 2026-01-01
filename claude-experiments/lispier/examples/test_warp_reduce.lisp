;; Test warp reduction sum using gpu.shuffle
;; Each thread starts with its lane id (0-31), then reduces to sum (0+1+...+31 = 496)
(require-dialect memref)
(require-dialect arith)
(require-dialect func)
(require-dialect gpu)
(require-dialect scf)

(compilation
  (target rocm
    (pass lower-affine)
    (pass convert-scf-to-cf)
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
    (pass gpu-to-llvm {:use-bare-pointers-for-kernels true})
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-index-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; External function declarations
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"}
      (region))

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"}
      (region))

    (defn main []
      ;; Allocate output buffer - one value per thread
      (def out (memref.alloc {:result memref<32xf32>}))
      (def out_unranked (memref.cast {:result "memref<*xf32>"} out))
      (gpu.host_register out_unranked)

      ;; Constants
      (def c0 (: 0 index))
      (def c1 (: 1 index))
      (def c32 (: 32 index))

      ;; Launch 1 block of 32 threads
      ;; Warp reduction: sum 0+1+2+...+31 = 496
      (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
        c1 c1 c1 c32 c1 c1
        (region
          (block [(: bx index) (: by index) (: bz index)
                  (: tx index) (: ty index) (: tz index)
                  (: gridDimX index) (: gridDimY index) (: gridDimZ index)
                  (: blockDimX index) (: blockDimY index) (: blockDimZ index)]
            ;; Each thread starts with its lane id
            (def lane_i64 (arith.index_cast {:result i64} tx))
            (def val (arith.sitofp {:result f32} lane_i64))

            ;; Constants for shuffle
            (def c16_i32 (: 16 i32))
            (def c8_i32 (: 8 i32))
            (def c4_i32 (: 4 i32))
            (def c2_i32 (: 2 i32))
            (def c1_i32 (: 1 i32))
            (def c32_i32 (: 32 i32))

            ;; Warp reduction: tree-style sum using XOR shuffles
            ;; Step 1: Add value from lane XOR 16
            (def other16 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} val c16_i32 c32_i32))
            (def sum16 (arith.addf val other16))

            ;; Step 2: Add value from lane XOR 8
            (def other8 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum16 c8_i32 c32_i32))
            (def sum8 (arith.addf sum16 other8))

            ;; Step 3: Add value from lane XOR 4
            (def other4 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum8 c4_i32 c32_i32))
            (def sum4 (arith.addf sum8 other4))

            ;; Step 4: Add value from lane XOR 2
            (def other2 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum4 c2_i32 c32_i32))
            (def sum2 (arith.addf sum4 other2))

            ;; Step 5: Add value from lane XOR 1
            (def other1 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum2 c1_i32 c32_i32))
            (def final_sum (arith.addf sum2 other1))

            ;; Store result - all lanes should have the same sum (496)
            (memref.store final_sum out tx)
            (gpu.terminator))))

      ;; Print first result - should be 496 (sum of 0+1+...+31)
      (def result (memref.load out c0))
      (func.call {:callee "@printF32"} result)
      (func.call {:callee "@printNewline"})

      ;; Unregister and cleanup
      (gpu.host_unregister out_unranked)
      (memref.dealloc out)
      (func.return))))
