; Linalg dialect operations - NOW WORKING!
; The linalg dialect provides high-level linear algebra operations
;
; STATUS: Basic named ops (fill, copy, matmul, dot, matvec) now work!
; TODO: linalg.generic with indexing_maps still needs work

(require-dialect [func :as f] [arith :as a] [linalg :as l] [memref :as m])

(module
  (do
    ; Test 1: linalg.fill - fill a tensor/memref with a value [WORKS]
    (f/func {:sym_name "fill_memref"
             :function_type (-> [f32 memref<4x4xf32>] [])}
      (region
        (block [(: val f32) (: out memref<4x4xf32>)]
          (l/fill val out)
          (f/return))))

    ; Test 2: linalg.copy - copy one tensor/memref to another [WORKS]
    (f/func {:sym_name "copy_memref"
             :function_type (-> [memref<4x4xf32> memref<4x4xf32>] [])}
      (region
        (block [(: src memref<4x4xf32>) (: dst memref<4x4xf32>)]
          (l/copy src dst)
          (f/return))))

    ; Test 3: linalg.matmul - matrix multiplication [WORKS]
    ; C = A * B where A is MxK, B is KxN, C is MxN
    (f/func {:sym_name "matmul"
             :function_type (-> [memref<4x4xf32> memref<4x4xf32> memref<4x4xf32>] [])}
      (region
        (block [(: a memref<4x4xf32>) (: b memref<4x4xf32>) (: c memref<4x4xf32>)]
          (l/matmul a b c)
          (f/return))))

    ; Test 4: linalg.dot - dot product [WORKS]
    (f/func {:sym_name "dot"
             :function_type (-> [memref<4xf32> memref<4xf32> memref<f32>] [])}
      (region
        (block [(: a memref<4xf32>) (: b memref<4xf32>) (: out memref<f32>)]
          (l/dot a b out)
          (f/return))))

    ; Test 5: linalg.matvec - matrix-vector multiplication [WORKS]
    (f/func {:sym_name "matvec"
             :function_type (-> [memref<4x4xf32> memref<4xf32> memref<4xf32>] [])}
      (region
        (block [(: mat memref<4x4xf32>) (: vec memref<4xf32>) (: out memref<4xf32>)]
          (l/matvec mat vec out)
          (f/return))))))
