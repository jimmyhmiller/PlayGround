;; Test linalg.generic with reduction iterator (CPU version first)
(require-dialect func)
(require-dialect memref)
(require-dialect arith)
(require-dialect scf)
(require-dialect linalg)
(require-dialect llvm)

;; External printf declaration
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

;; Simple CPU target
(compilation
  (target cpu
    (pass convert-linalg-to-loops)
    (pass convert-scf-to-cf)
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-index-to-llvm)
    (pass convert-math-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; Test: Sum each row of a 4x8 matrix
    (func.func {:sym_name "row_sum"
                :function_type (-> [memref<4x8xf32> memref<4xf32>] [])}
      (region
        (block [(: inp memref<4x8xf32>)
                (: out memref<4xf32>)]

          ;; Use linalg.generic with reduction to sum each row
          (linalg.generic
            {:indexing_maps [affine_map<(d0,d1)->(d0,d1)>
                             affine_map<(d0,d1)->(d0)>]
             :iterator_types ["parallel" "reduction"]}
            inp out
            (region
              (block [(: x f32) (: acc f32)]
                (def sum (arith.addf x acc))
                (linalg.yield sum))))

          (func.return))))

    (func.func {:sym_name "main" :function_type (-> [] [i32])}
      (region
        (block []
          ;; Allocate input and output
          (def inp (memref.alloc {:result memref<4x8xf32>}))
          (def out (memref.alloc {:result memref<4xf32>}))

          ;; Initialize input to 1.0
          (def one (: 1.0 f32))
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c4 (: 4 index))
          (def c8 (: 8 index))

          (scf.for c0 c4 c1
            (region
              (block [(: i index)]
                (scf.for c0 c8 c1
                  (region
                    (block [(: j index)]
                      (memref.store one inp i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize output to 0.0
          (def zero (: 0.0 f32))
          (scf.for c0 c4 c1
            (region
              (block [(: i index)]
                (memref.store zero out i)
                (scf.yield))))

          ;; Call row_sum
          (func.call "row_sum" inp out)

          ;; Print result (should be 8.0 for each row)
          (scf.for c0 c4 c1
            (region
              (block [(: i index)]
                (def val (memref.load {:result f32} out i))
                (def i_i64 (arith.index_cast {:result i64} i))
                (def val_f64 (arith.extf {:result f64} val))
                (print "Row %ld sum: %f\n" i_i64 val_f64)
                (scf.yield))))

          ;; Return 0
          (def ret (: 0 i32))
          (func.return ret))))))
