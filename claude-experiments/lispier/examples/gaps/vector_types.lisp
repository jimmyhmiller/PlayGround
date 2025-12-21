; GAP: Vector types
; MLIR supports vector types like vector<4xf32>, vector<4x4xf32>
; This file tests whether lispier can handle vector types and operations

(require-dialect [func :as f] [arith :as a] [vector :as v])

(module
  (do
    ; Test 1: Vector type in function signature
    (f/func {:sym_name "vector_add"
             :function_type (-> [vector<4xf32> vector<4xf32>] [vector<4xf32>])}
      (region
        (block [(: a vector<4xf32>) (: b vector<4xf32>)]
          ; Vector addition using arith.addf should work on vectors
          (def result (a/addf a b))
          (f/return result))))

    ; Test 2: 2D vector type
    (f/func {:sym_name "vector_2d"
             :function_type (-> [vector<4x4xf32>] [vector<4x4xf32>])}
      (region
        (block [(: v vector<4x4xf32>)]
          (f/return v))))

    ; Test 3: vector.broadcast - broadcast scalar to vector
    (f/func {:sym_name "broadcast_scalar"
             :function_type (-> [f32] [vector<4xf32>])}
      (region
        (block [(: scalar f32)]
          (def vec (v/broadcast {:result vector<4xf32>} scalar))
          (f/return vec))))

    ; Test 4: vector.extract - extract scalar from vector
    (f/func {:sym_name "extract_scalar"
             :function_type (-> [vector<4xf32>] [f32])}
      (region
        (block [(: vec vector<4xf32>)]
          (def elem (v/extract {:result f32 :position [0]} vec))
          (f/return elem))))))
