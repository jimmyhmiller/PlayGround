;; ========================================
;; GPU Vector Addition Example
;; ========================================
;;
;; Classic GPU programming example: parallel vector addition
;; Computes: C[i] = A[i] + B[i] for all i
;;
;; This demonstrates:
;; - 1D thread indexing using block_id and thread_id
;; - Memory access patterns for GPU
;; - Grid/block dimension calculations

;; ========================================
;; GPU Kernel Module
;; ========================================

(operation
  (name gpu.module)
  (attributes {:sym_name @vector_add_module})
  (regions
    (region
      (block
        ;; Vector addition kernel
        (operation
          (name gpu.func)
          (attributes
            {:sym_name @vector_add_kernel
             :function_type (!function)
                            (inputs memref<1024xf32> memref<1024xf32> memref<1024xf32>)
                            (results)
             :kernel unit})
          (regions
            (region
              (block
                (arguments
                  [(: %a memref<1024xf32>)
                   (: %b memref<1024xf32>)
                   (: %c memref<1024xf32>)])

                ;; Calculate global thread index
                ;; global_idx = block_id * block_dim + thread_id

                ;; Get block ID
                (operation
                  (name gpu.block_id)
                  (result-bindings [%block_id])
                  (result-types index)
                  (attributes {:dimension x}))

                ;; Get block dimension
                (operation
                  (name gpu.block_dim)
                  (result-bindings [%block_dim])
                  (result-types index)
                  (attributes {:dimension x}))

                ;; Get thread ID within block
                (operation
                  (name gpu.thread_id)
                  (result-bindings [%thread_id])
                  (result-types index)
                  (attributes {:dimension x}))

                ;; Calculate: block_id * block_dim
                (operation
                  (name arith.muli)
                  (result-bindings [%block_offset])
                  (result-types index)
                  (operands %block_id %block_dim))

                ;; Calculate: global_idx = block_offset + thread_id
                (operation
                  (name arith.addi)
                  (result-bindings [%global_idx])
                  (result-types index)
                  (operands %block_offset %thread_id))

                ;; Bounds check (optional, but good practice)
                ;; if (global_idx < 1024) { ... }
                (constant %size (: 1024 index))

                (operation
                  (name arith.cmpi)
                  (result-bindings [%in_bounds])
                  (result-types i1)
                  (attributes {:predicate (: 2 i64)})
                  (operands %global_idx %size))

                ;; Conditional execution
                (operation
                  (name scf.if)
                  (operands %in_bounds)
                  (regions
                    (region
                      (block
                        ;; Load a[global_idx]
                        (operation
                          (name memref.load)
                          (result-bindings [%a_val])
                          (result-types f32)
                          (operands %a %global_idx))

                        ;; Load b[global_idx]
                        (operation
                          (name memref.load)
                          (result-bindings [%b_val])
                          (result-types f32)
                          (operands %b %global_idx))

                        ;; Compute sum
                        (operation
                          (name arith.addf)
                          (result-bindings [%sum])
                          (result-types f32)
                          (operands %a_val %b_val))

                        ;; Store c[global_idx] = sum
                        (operation
                          (name memref.store)
                          (operands %sum %c %global_idx))

                        (operation
                          (name scf.yield))))))

                ;; Return from kernel
                (operation
                  (name gpu.return))))))))))

;; ========================================
;; Host Code
;; ========================================

(defn main [] i32

  ;; Grid/block configuration
  ;; Total threads needed: 1024
  ;; Block size: 256 threads per block
  ;; Grid size: 1024/256 = 4 blocks

  (constant %block_size (: 256 index))
  (constant %grid_size (: 4 index))
  (constant %c1 (: 1 index))

  ;; Allocate vectors
  (operation
    (name memref.alloc)
    (result-bindings [%a])
    (result-types memref<1024xf32>))

  (operation
    (name memref.alloc)
    (result-bindings [%b])
    (result-types memref<1024xf32>))

  (operation
    (name memref.alloc)
    (result-bindings [%c])
    (result-types memref<1024xf32>))

  ;; Launch kernel with 4 blocks of 256 threads each
  (operation
    (name gpu.launch_func)
    (attributes
      {:kernel @vector_add_module::@vector_add_kernel
       :gridSizeX %grid_size
       :gridSizeY %c1
       :gridSizeZ %c1
       :blockSizeX %block_size
       :blockSizeY %c1
       :blockSizeZ %c1})
    (operands %a %b %c))

  ;; Return success
  (constant %c0 (: 0 i32))
  (return %c0))
