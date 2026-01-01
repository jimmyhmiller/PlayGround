;; Minimal test: subview with GPU pipeline
(require-dialect memref)
(require-dialect arith)
(require-dialect func)
(require-dialect linalg)
(require-dialect gpu)

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
  ;; Try WITHOUT bare pointers first
  (pass convert-gpu-to-rocdl)
  (pass gpu-module-to-binary)
  (pass gpu-to-llvm)
  ;; expand-strided-metadata FIRST - it generates arith/affine ops
  (pass expand-strided-metadata)
  ;; lower-affine again because expand-strided-metadata generates affine.apply
  (pass lower-affine)
  ;; Now convert everything to LLVM
  (pass convert-cf-to-llvm)
  (pass convert-arith-to-llvm)
  (pass convert-index-to-llvm)
  (pass convert-math-to-llvm)
  (pass finalize-memref-to-llvm)
  (pass convert-func-to-llvm)
  (pass reconcile-unrealized-casts)))

(module
  (do
    ;; Simple function that takes a strided memref and does linalg.fill
    (func.func {:sym_name "fill_strided"
                :function_type (-> ["memref<768xf32, strided<[1], offset: ?>>" f32] [])}
      (region
        (block [(: buf "memref<768xf32, strided<[1], offset: ?>>")
                (: val f32)]
          (linalg.fill {:ins 1 :outs 1} val buf
            (region
              (block [(: in f32) (: _out f32)]
                (linalg.yield in))))
          (func.return))))

    (func.func {:sym_name "main" :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate 2D buffer
          (def all_w (memref.alloc {:result memref<12x768xf32>}))
          (gpu.host_register (memref.cast {:result "memref<*xf32>"} all_w))

          ;; Create subview for layer 3
          (def layer (: 3 index))
          (def view (memref.subview {:result "memref<768xf32, strided<[1], offset: ?>>"
                                     :operandSegmentSizes "array<i32: 1, 1, 0, 0>"
                                     :static_offsets "array<i64: -9223372036854775808, 0>"
                                     :static_sizes "array<i64: 1, 768>"
                                     :static_strides "array<i64: 1, 1>"}
                        all_w layer))

          ;; Fill with value
          (def val (: 42.0 f32))
          (func.call {:callee "@fill_strided"} view val)

          (memref.dealloc all_w)
          (func.return))))))
