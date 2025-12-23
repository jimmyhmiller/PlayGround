; GAP: scf.parallel and scf.reduce operations
; scf.parallel is a parallel loop construct
;
; ACTUAL ERROR: "invalid operand type" during IR generation
;   - scf.parallel requires special handling for bounds/steps
;   - scf.reduce for reductions needs specific support
;   - scf.forall also not supported

(require-dialect [func :as f] [arith :as a] [scf :as s] [memref :as m])

(module
  (do
    ; Test 1: Basic scf.parallel - parallel for loop
    ; In MLIR:
    ; scf.parallel (%i) = (%c0) to (%c10) step (%c1) {
    ;   ...
    ;   scf.yield
    ; }
    (f/func {:sym_name "parallel_loop"
             :function_type (-> [memref<10xf32>] [])}
      (region
        (block [(: mem memref<10xf32>)]
          (def c0 (: 0 index))
          (def c10 (: 10 index))
          (def c1 (: 1 index))
          (def val (: 1.0 f32))
          ; scf.parallel with lower bound, upper bound, step
          (s/parallel c0 c10 c1
            (region
              (block [(: i index)]
                (m/store val mem i)
                (s/yield))))
          (f/return))))

    ; Test 2: scf.parallel with 2D iteration space
    (f/func {:sym_name "parallel_2d"
             :function_type (-> [memref<10x10xf32>] [])}
      (region
        (block [(: mem memref<10x10xf32>)]
          (def c0 (: 0 index))
          (def c10 (: 10 index))
          (def c1 (: 1 index))
          (def val (: 2.0 f32))
          ; 2D parallel: two sets of bounds/steps
          (s/parallel [c0 c0] [c10 c10] [c1 c1]
            (region
              (block [(: i index) (: j index)]
                (m/store val mem i j)
                (s/yield))))
          (f/return))))

    ; Test 3: scf.parallel with reduction
    ; Sum all elements in parallel
    (f/func {:sym_name "parallel_sum"
             :function_type (-> [memref<10xf32>] [f32])}
      (region
        (block [(: mem memref<10xf32>)]
          (def c0 (: 0 index))
          (def c10 (: 10 index))
          (def c1 (: 1 index))
          (def init (: 0.0 f32))
          ; scf.parallel with init value and scf.reduce
          (def sum (s/parallel {:result f32} c0 c10 c1 init
            (region
              (block [(: i index)]
                (def elem (m/load {:result f32} mem i))
                ; scf.reduce specifies the reduction operation
                (s/reduce elem
                  (region
                    (block [(: lhs f32) (: rhs f32)]
                      (def combined (a/addf lhs rhs))
                      (s/reduce.return combined))))
                (s/yield)))))
          (f/return sum))))

    ; Test 4: scf.forall - another parallel loop variant (newer)
    ; Similar to scf.parallel but with different semantics
    (f/func {:sym_name "forall_loop"
             :function_type (-> [memref<10xf32>] [])}
      (region
        (block [(: mem memref<10xf32>)]
          (def c0 (: 0 index))
          (def c10 (: 10 index))
          (def val (: 3.0 f32))
          ; scf.forall with range
          (s/forall {:bounds [10]}
            (region
              (block [(: i index)]
                (m/store val mem i)
                (s/forall.in_parallel))))
          (f/return))))))
