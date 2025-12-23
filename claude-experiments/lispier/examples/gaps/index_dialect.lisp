; WORKING: Index dialect operations
; The index dialect provides operations specific to index type
; Status: All index dialect operations work correctly

(require-dialect [func :as f] [index :as i])

(module
  (do
    ; Test 1: index.constant - create constant index
    (f/func {:sym_name "index_const"
             :function_type (-> [] [index])
             :llvm.emit_c_interface true}
      (region
        (block []
          (def idx (i/constant {:value 42}))
          (f/return idx))))

    ; Test 2: index.add - add two indices
    (f/func {:sym_name "index_add"
             :function_type (-> [index index] [index])
             :llvm.emit_c_interface true}
      (region
        (block [(: a index) (: b index)]
          (def result (i/add a b))
          (f/return result))))

    ; Test 3: index.sub - subtract indices
    (f/func {:sym_name "index_sub"
             :function_type (-> [index index] [index])
             :llvm.emit_c_interface true}
      (region
        (block [(: a index) (: b index)]
          (def result (i/sub a b))
          (f/return result))))

    ; Test 4: index.mul - multiply indices
    (f/func {:sym_name "index_mul"
             :function_type (-> [index index] [index])
             :llvm.emit_c_interface true}
      (region
        (block [(: a index) (: b index)]
          (def result (i/mul a b))
          (f/return result))))

    ; Test 5: index.divs - signed division
    (f/func {:sym_name "index_divs"
             :function_type (-> [index index] [index])
             :llvm.emit_c_interface true}
      (region
        (block [(: a index) (: b index)]
          (def result (i/divs a b))
          (f/return result))))

    ; Test 6: index.cmp - compare indices
    ; predicates: eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge
    (f/func {:sym_name "index_cmp"
             :function_type (-> [index index] [i1])
             :llvm.emit_c_interface true}
      (region
        (block [(: a index) (: b index)]
          (def result (i/cmp {:predicate "slt"} a b))
          (f/return result))))

    ; Test 7: index.casts - cast index to signed integer
    (f/func {:sym_name "index_to_i64"
             :function_type (-> [index] [i64])
             :llvm.emit_c_interface true}
      (region
        (block [(: idx index)]
          (def result (i/casts {:result i64} idx))
          (f/return result))))

    ; Test 8: index.castu - cast unsigned integer to index
    (f/func {:sym_name "i64_to_index"
             :function_type (-> [i64] [index])
             :llvm.emit_c_interface true}
      (region
        (block [(: val i64)]
          (def result (i/castu {:result index} val))
          (f/return result))))

    ; Test 9: index.sizeof - get size of type in bytes as index
    (f/func {:sym_name "sizeof_f32"
             :function_type (-> [] [index])
             :llvm.emit_c_interface true}
      (region
        (block []
          (def sz (i/sizeof {:type f32}))
          (f/return sz))))))
