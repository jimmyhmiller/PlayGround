;; Example: Direct struct field access for CVectorLayout
;;
;; This example demonstrates how to manipulate CVectorLayout structs directly
;; from MLIR code, accessing fields like length and data pointer without
;; function call overhead.
;;
;; CVectorLayout struct layout:
;; !llvm.struct<(ptr, i64, i64, i64)>
;; Fields: [data_ptr, len, capacity, elem_size]

;; Function that extracts the length from a CVectorLayout pointer
;; This demonstrates direct field access using llvm.getelementptr
(defn get_vector_length [(: %vec_layout_ptr !llvm.ptr)] i64
  ;; Create constant for field index 1 (len field)
  (constant %c1 (: 1 i32))

  ;; GEP to access the 'len' field (offset 1 in the struct)
  ;; The struct has 4 fields: [data, len, capacity, elem_size]
  ;; We want field index 1 (len)
  (op %len_field_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec_layout_ptr %c1]))

  ;; Load the i64 length value from the pointer
  (op %len (: i64) (llvm.load [%len_field_ptr]))

  ;; Return the length
  (return %len))

;; Function that gets the data pointer from a CVectorLayout
(defn get_vector_data [(: %vec_layout_ptr !llvm.ptr)] !llvm.ptr
  ;; Create constant for field index 0 (data field)
  (constant %c0 (: 0 i32))

  ;; GEP to access the 'data' field (offset 0 in the struct)
  (op %data_field_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec_layout_ptr %c0]))

  ;; Load the pointer value
  (op %data_ptr (: !llvm.ptr) (llvm.load [%data_field_ptr]))

  ;; Return the data pointer
  (return %data_ptr))

;; Function that reads the element size field
(defn get_element_size [(: %vec_layout_ptr !llvm.ptr)] i64
  ;; Create constant for field index 3 (elem_size field)
  (constant %c3 (: 3 i32))

  ;; GEP to access the 'elem_size' field (offset 3 in the struct)
  (op %elem_size_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec_layout_ptr %c3]))

  ;; Load the element size
  (op %elem_size (: i64) (llvm.load [%elem_size_ptr]))

  ;; Return it
  (return %elem_size))

;; More complex example: sum two vector lengths
;; This shows how direct struct access avoids function call overhead
(defn sum_vector_lengths [(: %vec1_ptr !llvm.ptr) (: %vec2_ptr !llvm.ptr)] i64
  ;; Create constant for len field offset
  (constant %c1 (: 1 i32))

  ;; Access len field of first vector
  (op %len1_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec1_ptr %c1]))
  (op %len1 (: i64) (llvm.load [%len1_ptr]))

  ;; Access len field of second vector
  (op %len2_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec2_ptr %c1]))
  (op %len2 (: i64) (llvm.load [%len2_ptr]))

  ;; Sum the lengths
  (op %sum (: i64) (llvm.add [%len1 %len2]))

  ;; Return the sum
  (return %sum))

;; Advanced: Manually construct a CVectorLayout on the stack
;; This shows how you could build these structs in MLIR code
(defn create_empty_vector_layout [] !llvm.ptr
  ;; Constants for field offsets
  (constant %c0 (: 0 i32))
  (constant %c1 (: 1 i32))
  (constant %c2 (: 2 i32))
  (constant %c3 (: 3 i32))
  (constant %size (: 1 i64))
  (constant %zero_i64 (: 0 i64))
  (constant %eight (: 8 i64))

  ;; Allocate space for the struct on the stack
  ;; Size: 4 fields × 8 bytes = 32 bytes (allocating 1 i64 as placeholder)
  (op %struct_ptr (: !llvm.ptr) (llvm.alloca [%size]))

  ;; Create null pointer constant
  (op %null_ptr (: !llvm.ptr) (llvm.mlir.zero []))

  ;; Initialize 'data' field to null (offset 0)
  (op %data_ptr (: !llvm.ptr)
      (llvm.getelementptr [%struct_ptr %c0]))
  (op (llvm.store [%null_ptr %data_ptr]))

  ;; Initialize 'len' field to 0 (offset 1)
  (op %len_ptr (: !llvm.ptr)
      (llvm.getelementptr [%struct_ptr %c1]))
  (op (llvm.store [%zero_i64 %len_ptr]))

  ;; Initialize 'capacity' field to 0 (offset 2)
  (op %cap_ptr (: !llvm.ptr)
      (llvm.getelementptr [%struct_ptr %c2]))
  (op (llvm.store [%zero_i64 %cap_ptr]))

  ;; Initialize 'elem_size' field to 8 (offset 3, assuming pointers)
  (op %elem_size_ptr (: !llvm.ptr)
      (llvm.getelementptr [%struct_ptr %c3]))
  (op (llvm.store [%eight %elem_size_ptr]))

  ;; Return pointer to the struct
  (return %struct_ptr))

;; Example showing how you might iterate over vector elements
;; Given a CVectorLayout pointer, get length and data for iteration
(defn get_vector_info [(: %vec_ptr !llvm.ptr)] ()
  ;; Constants
  (constant %c0 (: 0 i32))
  (constant %c1 (: 1 i32))

  ;; Get length
  (op %len_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec_ptr %c1]))
  (op %len (: i64) (llvm.load [%len_ptr]))

  ;; Get data pointer
  (op %data_field_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec_ptr %c0]))
  (op %data_ptr (: !llvm.ptr) (llvm.load [%data_field_ptr]))

  ;; TODO: Would need scf.for loop here to actually iterate
  ;; This is just a skeleton showing the struct access pattern

  (op (func.return [])))

;; Key benefits of this approach:
;;
;; 1. **Performance**: Direct memory access, no function call overhead
;; 2. **Type Safety**: MLIR type system understands the struct layout
;; 3. **Simplicity**: Fewer exported C functions needed
;; 4. **Flexibility**: Can manipulate structs directly in MLIR
;;
;; Tradeoffs:
;;
;; 1. **ABI Lock-in**: Struct layout becomes part of your stable API
;; 2. **No Encapsulation**: Internal structure is exposed
;; 3. **Mutation Risk**: Direct access bypasses immutability guarantees
