; GAP: Linalg dialect operations
; The linalg dialect provides high-level linear algebra operations
; This file tests linalg dialect support

(require-dialect [func :as f] [arith :as a] [linalg :as l] [memref :as m] [tensor :as t])

(module
  (do
    ; Test 1: linalg.fill - fill a tensor/memref with a value
    (f/func {:sym_name "fill_memref"
             :function_type (-> [f32 memref<4x4xf32>] [])}
      (region
        (block [(: val f32) (: out memref<4x4xf32>)]
          (l/fill val out)
          (f/return))))

    ; Test 2: linalg.copy - copy one tensor/memref to another
    (f/func {:sym_name "copy_memref"
             :function_type (-> [memref<4x4xf32> memref<4x4xf32>] [])}
      (region
        (block [(: src memref<4x4xf32>) (: dst memref<4x4xf32>)]
          (l/copy src dst)
          (f/return))))

    ; Test 3: linalg.matmul - matrix multiplication
    ; C = A * B where A is MxK, B is KxN, C is MxN
    (f/func {:sym_name "matmul"
             :function_type (-> [memref<4x4xf32> memref<4x4xf32> memref<4x4xf32>] [])}
      (region
        (block [(: a memref<4x4xf32>) (: b memref<4x4xf32>) (: c memref<4x4xf32>)]
          (l/matmul a b c)
          (f/return))))

    ; Test 4: linalg.dot - dot product
    (f/func {:sym_name "dot"
             :function_type (-> [memref<4xf32> memref<4xf32> memref<f32>] [])}
      (region
        (block [(: a memref<4xf32>) (: b memref<4xf32>) (: out memref<f32>)]
          (l/dot a b out)
          (f/return))))

    ; Test 5: linalg.matvec - matrix-vector multiplication
    (f/func {:sym_name "matvec"
             :function_type (-> [memref<4x4xf32> memref<4xf32> memref<4xf32>] [])}
      (region
        (block [(: mat memref<4x4xf32>) (: vec memref<4xf32>) (: out memref<4xf32>)]
          (l/matvec mat vec out)
          (f/return))))

    ; Test 6: linalg.generic - the most general linalg operation
    ; This is complex - requires indexing_maps and iterator_types attributes
    (f/func {:sym_name "element_wise_add"
             :function_type (-> [memref<4xf32> memref<4xf32> memref<4xf32>] [])}
      (region
        (block [(: a memref<4xf32>) (: b memref<4xf32>) (: c memref<4xf32>)]
          ; linalg.generic needs:
          ; - indexing_maps: affine maps for each operand
          ; - iterator_types: parallel or reduction for each dimension
          ; - a body region that computes the elementwise operation
          (l/generic {:indexing_maps ["affine_map<(d0) -> (d0)>"
                                      "affine_map<(d0) -> (d0)>"
                                      "affine_map<(d0) -> (d0)>"]
                      :iterator_types ["parallel"]}
            [a b c]
            (region
              (block [(: in_a f32) (: in_b f32) (: out_c f32)]
                (def sum (a/addf in_a in_b))
                (l/yield sum))))
          (f/return))))

    ; Test 7: linalg on tensors (returns new tensor)
    (f/func {:sym_name "matmul_tensor"
             :function_type (-> [tensor<4x4xf32> tensor<4x4xf32> tensor<4x4xf32>] [tensor<4x4xf32>])}
      (region
        (block [(: a tensor<4x4xf32>) (: b tensor<4x4xf32>) (: c tensor<4x4xf32>)]
          (def result (l/matmul {:result tensor<4x4xf32>} a b c))
          (f/return result))))))
