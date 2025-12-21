; Memory operations example using memref dialect
; Simplified version - basic memref operations

(require-dialect [memref :as m] [arith :as a] [func :as f])

(module
  (do
    ; Simple function that allocates and returns an uninitialized memref
    (f/func {:sym_name "create_buffer"
             :function_type (-> [] [memref<10xi64>])}
      (region
        (block []
          (def buffer (m/alloc {:result memref<10xi64>}))
          (f/return buffer))))

    ; Function that stores a value at index 0
    (f/func {:sym_name "store_at_zero"
             :function_type (-> [memref<10xi64> i64] [])}
      (region
        (block [(: buffer memref<10xi64>) (: value i64)]
          (def idx (: 0 index))
          (m/store value buffer idx)
          (f/return))))

    ; Function that loads from index 0
    (f/func {:sym_name "load_from_zero"
             :function_type (-> [memref<10xi64>] [i64])}
      (region
        (block [(: buffer memref<10xi64>)]
          (def idx (: 0 index))
          (def value (m/load {:result i64} buffer idx))
          (f/return value))))

    ; Main function
    (f/func {:sym_name "main"
             :function_type (-> [] [i64])
             :llvm.emit_c_interface true}
      (region
        (block []
          (def buffer (m/alloc {:result memref<10xi64>}))
          (def val (: 42 i64))
          (def idx (: 0 index))
          (m/store val buffer idx)
          (def result (m/load {:result i64} buffer idx))
          (m/dealloc buffer)
          (f/return result))))))
