; GAP: Advanced memref operations
; The memref dialect has many operations beyond basic load/store
; This file tests advanced memref operations
;
; KNOWN GAPS:
; - Commas in type syntax not supported (e.g., strided<[10, 1], offset: 0>)
; - Complex memref layouts with affine maps may not parse correctly

(require-dialect [func :as f] [arith :as a] [memref :as m])

(module
  (do
    ; Test 1: memref.dim - get dimension size
    ; This should work - tests dynamic memref handling
    (f/func {:sym_name "get_dim"
             :function_type (-> [memref<?xf32>] [index])}
      (region
        (block [(: mem memref<?xf32>)]
          (def c0 (: 0 index))
          (def dim (m/dim {:result index} mem c0))
          (f/return dim))))

    ; Test 2: memref.copy - copy contents between memrefs
    (f/func {:sym_name "copy_memref"
             :function_type (-> [memref<10xf32> memref<10xf32>] [])}
      (region
        (block [(: src memref<10xf32>) (: dst memref<10xf32>)]
          (m/copy src dst)
          (f/return))))

    ; Test 3: memref.realloc - reallocate with new size
    (f/func {:sym_name "realloc_memref"
             :function_type (-> ["memref<?xf32>" index] ["memref<?xf32>"])}
      (region
        (block [(: mem "memref<?xf32>") (: new_size index)]
          (def new_mem (m/realloc {:result "memref<?xf32>"} mem new_size))
          (f/return new_mem))))

    ; Test 4: memref.reshape - reshape a memref
    (f/func {:sym_name "reshape_memref"
             :function_type (-> [memref<16xf32> memref<2xindex>] ["memref<?x?xf32>"])}
      (region
        (block [(: mem memref<16xf32>) (: shape memref<2xindex>)]
          (def reshaped (m/reshape {:result "memref<?x?xf32>"} mem shape))
          (f/return reshaped))))))

; GAP DEMONSTRATIONS (commented out because they fail):
;
; 1. memref.subview with strided layout - commas in type not supported:
;    (f/func {:sym_name "get_subview"
;             :function_type (-> [memref<10x10xf32>] [memref<5x5xf32, strided<[10, 1], offset: 0>>])}
;      ...)
;
; 2. memref.expand_shape/collapse_shape with reassociation - nested arrays with commas:
;    (def expanded (m/expand_shape {:result memref<4x4xf32>
;                                   :reassociation [[0, 1]]} mem))
;
; 3. memref with affine layout map - commas in affine_map:
;    (f/func {:sym_name "column_major"
;             :function_type (-> [] ["memref<4x4xf32, affine_map<(d0, d1) -> (d1, d0)>>"])}
;      ...)
