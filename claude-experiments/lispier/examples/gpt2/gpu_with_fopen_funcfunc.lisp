;; GPU test with file I/O using func.func declarations instead of extern-fn
(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect linalg)
(require-dialect scf)
(require-dialect llvm)

(compilation
  (target rocm
    (pass convert-linalg-to-parallel-loops)
    (pass gpu-map-parallel-loops)
    (pass convert-parallel-loops-to-gpu)
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
    ;; Global paths
    (llvm.mlir.global {:sym_name "testfile_path"
                       :linkage 0
                       :global_type !llvm.array<10 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "/dev/null\0" :result !llvm.array<10 x i8>}))
          (llvm.return s))))

    (llvm.mlir.global {:sym_name "read_mode"
                       :linkage 0
                       :global_type !llvm.array<2 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "r\0" :result !llvm.array<2 x i8>}))
          (llvm.return s))))

    (func.func {:sym_name "main" :function_type (-> [] [])}
      (region
        (block []
          ;; Test file open/close using func.call
          (def path (llvm.mlir.addressof {:global_name @testfile_path :result !llvm.ptr}))
          (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))

          ;; Call fopen via func.call (declared as func.func below)
          (def file (func.call {:result !llvm.ptr} "fopen" path mode))
          (def _close_result (func.call {:result i32} "fclose" file))

          ;; Now do GPU operations
          (def A (memref.alloc {:result memref<4x4xf32>}))
          (def B (memref.alloc {:result memref<4x4xf32>}))
          (def C (memref.alloc {:result memref<4x4xf32>}))

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c4 (: 4 index))
          (def one (: 1.0 f32))
          (def zero (: 0.0 f32))

          (scf.for c0 c4 c1 (region (block [(: i index)]
            (scf.for c0 c4 c1 (region (block [(: j index)]
              (memref.store one A i j)
              (memref.store one B i j)
              (memref.store zero C i j)
              (scf.yield))))
            (scf.yield))))

          (gpu.host_register (memref.cast {:result "memref<*xf32>"} A))
          (gpu.host_register (memref.cast {:result "memref<*xf32>"} B))
          (gpu.host_register (memref.cast {:result "memref<*xf32>"} C))

          (linalg.matmul A B C)

          (def result (memref.load C c0 c0))
          (func.call "printF32" result)
          (func.call "printNewline")

          (memref.dealloc A)
          (memref.dealloc B)
          (memref.dealloc C)
          (func.return))))

    ;; Declare libc functions as func.func (will be converted to llvm.func by pipeline)
    (func.func {:sym_name "fopen"
                :function_type (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr])
                :sym_visibility "private"})

    (func.func {:sym_name "fclose"
                :function_type (-> [!llvm.ptr] [i32])
                :sym_visibility "private"})

    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"})

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"})))
