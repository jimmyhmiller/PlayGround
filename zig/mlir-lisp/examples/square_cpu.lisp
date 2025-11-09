;; CPU version of square - manually inlined GPU kernel as nested loops
(defn main [] i64
  (constant %c0 (: 0 index))
  (constant %c1 (: 1 index))
  (constant %c10 (: 10 index))

  ;; Allocate input and output
  (operation
    (name memref.alloc)
    (result-bindings [%input])
    (result-types memref<10x10xf32>))

  (operation
    (name memref.alloc)
    (result-bindings [%output])
    (result-types memref<10x10xf32>))

  ;; Nested loops: for row in 0..10, for col in 0..10
  (operation
    (name scf.for)
    (operands %c0 %c10 %c1)
    (regions
      (region
        (block
          (arguments [(: %row index)])
          (operation
            (name scf.for)
            (operands %c0 %c10 %c1)
            (regions
              (region
                (block
                  (arguments [(: %col index)])

                  ;; Load input[row][col]
                  (operation
                    (name memref.load)
                    (result-bindings [%val])
                    (result-types f32)
                    (operands %input %row %col))

                  ;; Square it
                  (operation
                    (name arith.mulf)
                    (result-bindings [%squared])
                    (result-types f32)
                    (operands %val %val))

                  ;; Store to output[row][col]
                  (operation
                    (name memref.store)
                    (operands %squared %output %row %col))

                  (operation
                    (name scf.yield))))))

          (operation
            (name scf.yield))))))

  ;; Return 0
  (constant %ret (: 0 i64))
  (return %ret))
