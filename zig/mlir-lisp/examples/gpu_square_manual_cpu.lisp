;; GPU Square example - manually converted to CPU loops
;; This shows what the GPU code would do if emulated on CPU

(defn main [] i64
  (constant %c0 (: 0 index))
  (constant %c1 (: 1 index))
  (constant %c10 (: 10 index))

  ;; Allocate matrices (same as GPU version)
  (operation
    (name memref.alloc)
    (result-bindings [%input])
    (result-types memref<10x10xf32>))

  (operation
    (name memref.alloc)
    (result-bindings [%output])
    (result-types memref<10x10xf32>))

  ;; Manual emulation of:
  ;; gpu.launch_func @kernel blocks in (%c10, %c1, %c1) threads in (%c10, %c1, %c1)
  ;;
  ;; This creates nested loops:
  ;; for block_x in 0..10:
  ;;   for thread_x in 0..10:
  ;;     square_kernel(input, output, block_x, thread_x)

  ;; Outer loop: block_id.x (0 to 10)
  (operation
    (name scf.for)
    (operand-uses %c0 %c10 %c1)
    (regions
      (region
        (block
          (arguments [(: %block_x index)])

          ;; Inner loop: thread_id.x (0 to 10)
          (operation
            (name scf.for)
            (operand-uses %c0 %c10 %c1)
            (regions
              (region
                (block
                  (arguments [(: %thread_x index)])

                  ;; === Inlined GPU kernel body ===
                  ;; Load input[block_x][thread_x]
                  (operation
                    (name memref.load)
                    (result-bindings [%val])
                    (result-types f32)
                    (operand-uses %input %block_x %thread_x))

                  ;; Square it
                  (operation
                    (name arith.mulf)
                    (result-bindings [%squared])
                    (result-types f32)
                    (operand-uses %val %val))

                  ;; Store to output[block_x][thread_x]
                  (operation
                    (name memref.store)
                    (operand-uses %squared %output %block_x %thread_x))

                  (operation
                    (name scf.yield))))))

          (operation
            (name scf.yield))))))

  ;; Return 0
  (constant %ret (: 0 i64))
  (return %ret))
