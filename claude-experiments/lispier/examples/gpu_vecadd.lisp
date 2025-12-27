;; GPU Vector Addition Example
;; Translated from MLIR GPU dialect example

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect scf)

;; Compilation pipeline - chip is detected at runtime
;; Use bare pointer calling convention to avoid ciface wrapper issues with GPU runtime
(compilation
  (target rocm
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
    (pass convert-scf-to-cf)
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-func-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass gpu-to-llvm {:use-bare-pointers-for-kernels true :use-bare-pointers-for-host true})
    (pass reconcile-unrealized-casts))

  (target cuda
    (pass gpu-kernel-outlining)
    (pass nvvm-attach-target)
    (pass convert-gpu-to-nvvm {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
    (pass convert-scf-to-cf)
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-func-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass gpu-to-llvm {:use-bare-pointers-for-kernels true :use-bare-pointers-for-host true})
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; Vector addition kernel
    (func.func {:sym_name "vecadd"
                :function_type (-> [memref<5xf32> memref<5xf32> memref<5xf32>] [])}
      (region
        (block [(: arg0 memref<5xf32>) (: arg1 memref<5xf32>) (: arg2 memref<5xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def block_dim (: 5 index))

          ;; gpu.launch with grid and block dimensions
          ;; Block args: bx by bz tx ty tz gridDimX gridDimY gridDimZ blockDimX blockDimY blockDimZ
          ;; operandSegmentSizes: [async, gridX/Y/Z, blockX/Y/Z, clusterX/Y/Z, dynamicSharedMem]
          (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
            c1 c1 c1 block_dim c1 c1
            (region
              (block [(: bx index) (: by index) (: bz index)
                      (: tx index) (: ty index) (: tz index)
                      (: gridDimX index) (: gridDimY index) (: gridDimZ index)
                      (: blockDimX index) (: blockDimY index) (: blockDimZ index)]
                (def a (memref.load {:result f32} arg0 tx))
                (def b (memref.load {:result f32} arg1 tx))
                (def c (arith.addf a b))
                (memref.store c arg2 tx)
                (gpu.terminator))))

          (func.return))))

    ;; Main function
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c5 (: 5 index))
          (def cf1dot23 (: 1.23 f32))

          ;; Allocate memrefs
          (def mem0 (memref.alloc {:result memref<5xf32>}))
          (def mem1 (memref.alloc {:result memref<5xf32>}))
          (def mem2 (memref.alloc {:result memref<5xf32>}))

          ;; Cast to dynamic memrefs
          (def dyn0 (memref.cast {:result "memref<?xf32>"} mem0))
          (def dyn1 (memref.cast {:result "memref<?xf32>"} mem1))
          (def dyn2 (memref.cast {:result "memref<?xf32>"} mem2))

          ;; Initialize with scf.for loop
          (scf.for c0 c5 c1
            (region
              (block [(: i index)]
                (memref.store cf1dot23 dyn0 i)
                (memref.store cf1dot23 dyn1 i)
                (scf.yield))))

          ;; Cast to unranked memrefs
          (def unranked0 (memref.cast {:result "memref<*xf32>"} dyn0))
          (def unranked1 (memref.cast {:result "memref<*xf32>"} dyn1))
          (def unranked2 (memref.cast {:result "memref<*xf32>"} dyn2))

          ;; Register with GPU
          (gpu.host_register unranked0)
          (gpu.host_register unranked1)
          (gpu.host_register unranked2)

          ;; Get device memrefs
          (def dev0 (func.call {:result "memref<?xf32>"} "mgpuMemGetDeviceMemRef1dFloat" dyn0))
          (def dev1 (func.call {:result "memref<?xf32>"} "mgpuMemGetDeviceMemRef1dFloat" dyn1))
          (def dev2 (func.call {:result "memref<?xf32>"} "mgpuMemGetDeviceMemRef1dFloat" dyn2))

          ;; Cast back to fixed size
          (def fixed0 (memref.cast {:result memref<5xf32>} dev0))
          (def fixed1 (memref.cast {:result memref<5xf32>} dev1))
          (def fixed2 (memref.cast {:result memref<5xf32>} dev2))

          ;; Call vecadd
          (func.call "vecadd" fixed0 fixed1 fixed2)

          ;; Print result
          (func.call "printMemrefF32" unranked2)

          (func.return))))

    ;; External function declarations
    (func.func {:sym_name "mgpuMemGetDeviceMemRef1dFloat"
                :function_type (-> ["memref<?xf32>"] ["memref<?xf32>"])
                :sym_visibility "private"})

    (func.func {:sym_name "printMemrefF32"
                :function_type (-> ["memref<*xf32>"] [])
                :sym_visibility "private"})))