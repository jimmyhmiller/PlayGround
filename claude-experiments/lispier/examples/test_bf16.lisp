;; Test BF16 operations in MLIR
;; Verify: bf16 type, truncf f32->bf16, extf bf16->f32, arithmetic

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
      (def c4 (: 4 index))

      ;; Test 1: Basic bf16 conversion
      ;; Create f32 value, convert to bf16, convert back to f32
      (def val_f32 (: 3.14159 f32))
      (def val_bf16 (arith.truncf {:result bf16} val_f32))
      (def val_back (arith.extf {:result f32} val_bf16))
      (func.call {:callee "@printF32"} val_back)  ;; Should print ~3.14 (with bf16 precision loss)
      (func.call {:callee "@printNewline"})

      ;; Test 2: bf16 memref - allocate, store, load
      (def buf_bf16 (memref.alloc {:result memref<4xbf16>}))

      ;; Store bf16 values
      (def v0 (arith.truncf {:result bf16} (: 1.0 f32)))
      (def v1 (arith.truncf {:result bf16} (: 2.0 f32)))
      (def v2 (arith.truncf {:result bf16} (: 3.0 f32)))
      (def v3 (arith.truncf {:result bf16} (: 4.0 f32)))
      (memref.store v0 buf_bf16 c0)
      (memref.store v1 buf_bf16 c1)
      (def c2 (: 2 index))
      (def c3 (: 3 index))
      (memref.store v2 buf_bf16 c2)
      (memref.store v3 buf_bf16 c3)

      ;; Load and convert back to f32 for printing
      (def r0 (memref.load {:result bf16} buf_bf16 c0))
      (def r0_f32 (arith.extf {:result f32} r0))
      (func.call {:callee "@printF32"} r0_f32)  ;; Should print 1.0
      (func.call {:callee "@printNewline"})

      (def r3 (memref.load {:result bf16} buf_bf16 c3))
      (def r3_f32 (arith.extf {:result f32} r3))
      (func.call {:callee "@printF32"} r3_f32)  ;; Should print 4.0
      (func.call {:callee "@printNewline"})

      ;; Test 3: bf16 arithmetic (add two bf16 values)
      (def sum_bf16 (arith.addf v0 v1))  ;; 1.0 + 2.0 = 3.0
      (def sum_f32 (arith.extf {:result f32} sum_bf16))
      (func.call {:callee "@printF32"} sum_f32)  ;; Should print 3.0
      (func.call {:callee "@printNewline"})

      ;; Test 4: bf16 on GPU
      (def gpu_in (memref.alloc {:result memref<4xf32>}))
      (def gpu_out (memref.alloc {:result memref<4xbf16>}))

      ;; Initialize input with f32
      (memref.store (: 1.5 f32) gpu_in c0)
      (memref.store (: 2.5 f32) gpu_in c1)
      (memref.store (: 3.5 f32) gpu_in c2)
      (memref.store (: 4.5 f32) gpu_in c3)

      ;; Register for GPU
      (def gpu_in_unranked (memref.cast {:result "memref<*xf32>"} gpu_in))
      (def gpu_out_unranked (memref.cast {:result "memref<*xbf16>"} gpu_out))
      (gpu.host_register gpu_in_unranked)
      (gpu.host_register gpu_out_unranked)

      ;; GPU kernel: convert f32 to bf16
      (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
        c1 c1 c1 c4 c1 c1
        (region
          (block [(: _bx index) (: _by index) (: _bz index)
                  (: tid index) (: _ty index) (: _tz index)
                  (: _gridDimX index) (: _gridDimY index) (: _gridDimZ index)
                  (: _blockDimX index) (: _blockDimY index) (: _blockDimZ index)]

            (def val (memref.load {:result f32} gpu_in tid))
            (def val_bf (arith.truncf {:result bf16} val))
            (memref.store val_bf gpu_out tid)
            (gpu.terminator))))

      ;; Read back and print
      (def out0 (memref.load {:result bf16} gpu_out c0))
      (def out0_f32 (arith.extf {:result f32} out0))
      (func.call {:callee "@printF32"} out0_f32)  ;; Should print 1.5
      (func.call {:callee "@printNewline"})

      (def out3 (memref.load {:result bf16} gpu_out c3))
      (def out3_f32 (arith.extf {:result f32} out3))
      (func.call {:callee "@printF32"} out3_f32)  ;; Should print 4.5
      (func.call {:callee "@printNewline"})

      ;; Cleanup
      (gpu.host_unregister gpu_in_unranked)
      (gpu.host_unregister gpu_out_unranked)
      (memref.dealloc buf_bf16)
      (memref.dealloc gpu_in)
      (memref.dealloc gpu_out)
      (func.return))))