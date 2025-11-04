;; Matrix multiplication: C = A * B
;; A: memref<?x?xf32> (MxK)
;; B: memref<?x?xf32> (KxN)
;; C: memref<?x?xf32> (MxN)
;;
;; This implements: C[i,j] = sum_k(A[i,k] * B[k,j])
;; Using triple nested affine loops with loop-carried accumulator

(defn matmul [(: %A memref<?x?xf32>)
              (: %B memref<?x?xf32>)
              (: %C memref<?x?xf32>)]
  ()

  ;; Constants for indexing - need to reuse these
  (constant %c0 (: 0 index))
  (constant %c1 (: 1 index))

  ;; Get matrix dimensions using op macro
  (op %M (: index) (memref.dim [%A %c0]))
  (op %K (: index) (memref.dim [%A %c1]))
  (op %N (: index) (memref.dim [%B %c1]))

  ;; Outer loop: for i = 0 to M
  (op (scf.for [%c0 %M %c1]
           (region
            (block
                (arguments [ (: %i index) ])

              ;; Middle loop: for j = 0 to N
              (op (scf.for [(constant (: 0 index)) %N %c1]
                       (region
                        (block
                            (arguments [ (: %j index) ])

                          ;; Load initial value from C[i,j] for accumulation
                          (op %acc_init (: f32) (memref.load [%C %i %j]))

                          ;; Inner loop: for k = 0 to K (with accumulator)
                          (op %acc_final (: f32)
                              (scf.for [%c0 %K %c1 %acc_init]
                                   (region
                                    (block
                                        (arguments [ (: %k index) (: %acc f32) ])

                                      ;; Load A[i, k]
                                      (op %a (: f32) (memref.load [%A %i %k]))

                                      ;; Load B[k, j]
                                      (op %b (: f32) (memref.load [%B %k %j]))

                                      ;; Multiply: %prod = %a * %b
                                      (op %prod (: f32) (arith.mulf [%a %b]))

                                      ;; Accumulate: %acc_next = %acc + %prod
                                      (op %acc_next (: f32) (arith.addf [%acc %prod]))

                                      ;; Yield updated accumulator
                                      (operation
                                       (name scf.yield)
                                       (operands %acc_next))))))

                          ;; Store final accumulated result back to C[i,j]
                          (op (memref.store [%acc_final %C %i %j]))

                          ;; Yield (end of j loop)
                          (operation
                           (name scf.yield))))))

              ;; Yield (end of i loop)
              (operation
               (name scf.yield))))))

  ;; Return from function
  (op (func.return [])))

;; Main function to test matmul
;; Initialize A = [[1, 2, 3], [4, 5, 6]] and B = [[1, 2], [3, 4], [5, 6]]
;; Expected C = [[22, 28], [49, 64]]
(defn main [] i64
  ;; Allocate matrices using op macro
  (op %A (: memref<2x3xf32>) (memref.alloc []))
  (op %B (: memref<3x2xf32>) (memref.alloc []))
  (op %C (: memref<2x2xf32>) (memref.alloc []))

  ;; Initialize A using op macro
      (op (memref.store [(constant (: 1.0 f32)) %A (constant (: 0 index)) (constant (: 0 index))]))
      (op (memref.store [(constant (: 2.0 f32)) %A (constant (: 0 index)) (constant (: 1 index))]))
      (op (memref.store [(constant (: 3.0 f32)) %A (constant (: 0 index)) (constant (: 2 index))]))
      (op (memref.store [(constant (: 4.0 f32)) %A (constant (: 1 index)) (constant (: 0 index))]))
      (op (memref.store [(constant (: 5.0 f32)) %A (constant (: 1 index)) (constant (: 1 index))]))
      (op (memref.store [(constant (: 6.0 f32)) %A (constant (: 1 index)) (constant (: 2 index))]))

      ;; Initialize B using op macro
      (op (memref.store [(constant (: 1.0 f32)) %B (constant (: 0 index)) (constant (: 0 index))]))
      (op (memref.store [(constant (: 2.0 f32)) %B (constant (: 0 index)) (constant (: 1 index))]))
      (op (memref.store [(constant (: 3.0 f32)) %B (constant (: 1 index)) (constant (: 0 index))]))
      (op (memref.store [(constant (: 4.0 f32)) %B (constant (: 1 index)) (constant (: 1 index))]))
      (op (memref.store [(constant (: 5.0 f32)) %B (constant (: 2 index)) (constant (: 0 index))]))
      (op (memref.store [(constant (: 6.0 f32)) %B (constant (: 2 index)) (constant (: 1 index))]))

      ;; Initialize C to zeros using op macro
      (op (memref.store [(constant (: 0.0 f32)) %C (constant (: 0 index)) (constant (: 0 index))]))
      (op (memref.store [(constant (: 0.0 f32)) %C (constant (: 0 index)) (constant (: 1 index))]))
      (op (memref.store [(constant (: 0.0 f32)) %C (constant (: 1 index)) (constant (: 0 index))]))
      (op (memref.store [(constant (: 0.0 f32)) %C (constant (: 1 index)) (constant (: 1 index))]))

      ;; Cast static memrefs to dynamic for matmul call using op macro
      (op %A_dyn (: memref<?x?xf32>) (memref.cast [%A]))
      (op %B_dyn (: memref<?x?xf32>) (memref.cast [%B]))
      (op %C_dyn (: memref<?x?xf32>) (memref.cast [%C]))

      ;; Call matmul
      (call @matmul %A_dyn %B_dyn %C_dyn ())

      ;; Load C[0,0] and convert to i64 for return using op macro
      (op %result_f32 (: f32) (memref.load [%C (constant (: 0 index)) (constant (: 0 index))]))
      (op %result_i64 (: i64) (arith.fptosi [%result_f32]))

      ;; Return the result
      (return %result_i64))
