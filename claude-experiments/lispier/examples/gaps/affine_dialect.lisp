; WORKING: Affine dialect
; MLIR's affine dialect provides structured loop/memory operations with affine maps
; Syntax: affine_map<(d0, d1) -> (d0 + d1)> and affine_set<(d0) : (d0 >= 0)> as unquoted symbols

(require-dialect [func :as f] [arith :as a] [affine :as aff] [memref :as m])

(module
  (do
    ; Test 1: affine.for loop
    ; In MLIR:
    ; affine.for %i = 0 to 10 {
    ;   ...
    ; }
    (f/func {:sym_name "affine_loop"
             :function_type (-> [memref<10xf32>] [])}
      (region
        (block [(: mem memref<10xf32>)]
          (def c0 (: 0 index))
          (def c10 (: 10 index))
          ; How to express affine.for with bounds?
          (aff/for {:lower_bound 0 :upper_bound 10}
            (region
              (block [(: i index)]
                (def val (: 1.0 f32))
                (aff/store val mem i)
                (aff/yield))))
          (f/return))))

    ; Test 2: affine.load and affine.store
    ; These use affine maps for addressing
    (f/func {:sym_name "affine_load_store"
             :function_type (-> [memref<10xf32>] [f32])}
      (region
        (block [(: mem memref<10xf32>)]
          (def idx (: 5 index))
          ; affine.load uses affine map: affine_map<(d0) -> (d0)>
          (def val (aff/load {:result f32} mem idx))
          (f/return val))))

    ; Test 3: affine.if conditional
    ; In MLIR: affine.if affine_set<(d0) : (d0 >= 0)>(%i) { ... }
    (f/func {:sym_name "affine_conditional"
             :function_type (-> [index] [i32])}
      (region
        (block [(: i index)]
          (def one (: 1 i32))
          (def zero (: 0 i32))
          ; affine_set as unquoted symbol becomes MLIRLiteral
          (def result (aff/if {:condition affine_set<(d0) : (d0 >= 0)>
                               :result i32} i
            (region
              (block []
                (aff/yield one)))
            (region
              (block []
                (aff/yield zero)))))
          (f/return result))))

    ; Test 4: Affine map attribute
    ; affine_map<(d0, d1) -> (d0 + d1)>
    (f/func {:sym_name "affine_apply"
             :function_type (-> [index index] [index])}
      (region
        (block [(: d0 index) (: d1 index)]
          ; affine_map as unquoted symbol becomes MLIRLiteral
          (def result (aff/apply {:map affine_map<(d0, d1) -> (d0 + d1)>} d0 d1))
          (f/return result))))))
