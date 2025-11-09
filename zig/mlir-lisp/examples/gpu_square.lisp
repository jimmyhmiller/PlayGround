;; ========================================
;; GPU Example: Square Kernel
;; ========================================
;;
;; This example demonstrates GPU programming using MLIR's gpu dialect.
;; It implements a simple kernel that squares each element of a 10x10 matrix.
;;
;; Key concepts:
;; - gpu.module: Container for GPU kernels
;; - gpu.func: GPU kernel function (runs on device)
;; - gpu.block_id, gpu.thread_id: Get current block/thread coordinates
;; - gpu.launch_func: Launch kernel from host code

;; ========================================
;; GPU Kernel Module
;; ========================================

(operation
  (name gpu.module)
  (attributes {:sym_name @kernel_module})
  (regions
    (region
      (block
        ;; GPU kernel function - runs on device
        (operation
          (name gpu.func)
          (attributes
            {:sym_name @square_kernel
             :function_type (!function)
                            (inputs memref<10x10xf32> memref<10x10xf32>)
                            (results)
             :kernel unit})
          (regions
            (region
              (block
                ;; Kernel arguments
                (arguments
                  [(: %input memref<10x10xf32>)
                   (: %output memref<10x10xf32>)])

                ;; Get block ID (which block we're in)
                (operation
                  (name gpu.block_id)
                  (result-bindings [%block_x])
                  (result-types index)
                  (attributes {:dimension x}))

                ;; Get thread ID (which thread within block)
                (operation
                  (name gpu.thread_id)
                  (result-bindings [%thread_x])
                  (result-types index)
                  (attributes {:dimension x}))

                ;; Load value from input at [block_x, thread_x]
                (operation
                  (name memref.load)
                  (result-bindings [%val])
                  (result-types f32)
                  (operands %input %block_x %thread_x))

                ;; Square the value: result = val * val
                (operation
                  (name arith.mulf)
                  (result-bindings [%result])
                  (result-types f32)
                  (operands %val %val))

                ;; Store result to output at [block_x, thread_x]
                (operation
                  (name memref.store)
                  (operands %result %output %block_x %thread_x))

                ;; Return from kernel
                (operation
                  (name gpu.return))))))))))

;; ========================================
;; Host Code - Main Function
;; ========================================

(defn main [] i32

  ;; Create index constants for grid/block dimensions
  (constant %c10 (: 10 index))
  (constant %c1 (: 1 index))

  ;; Allocate input and output memrefs on host
  (operation
    (name memref.alloc)
    (result-bindings [%input])
    (result-types memref<10x10xf32>))

  (operation
    (name memref.alloc)
    (result-bindings [%output])
    (result-types memref<10x10xf32>))

  ;; Launch GPU kernel
  ;; - Grid: 10x1x1 blocks
  ;; - Block: 10x1x1 threads per block
  ;; - Total: 10*10 = 100 threads
  (operation
    (name gpu.launch_func)
    (attributes
      {:kernel @kernel_module::@square_kernel
       :blockSizeX %c10
       :blockSizeY %c1
       :blockSizeZ %c1
       :gridSizeX %c10
       :gridSizeY %c1
       :gridSizeZ %c1})
    (operands %input %output))

  ;; Return 0
  (constant %c0 (: 0 i32))
  (return %c0))
