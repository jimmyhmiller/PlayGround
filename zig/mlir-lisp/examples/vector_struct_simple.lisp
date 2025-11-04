;; Simpler example: Direct struct field access for CVectorLayout
;;
;; CVectorLayout struct layout:
;; !llvm.struct<(ptr, i64, i64, i64)>
;; Fields: [data_ptr, len, capacity, elem_size]

;; Function that extracts the length from a CVectorLayout pointer
(defn get_vector_length [(: %vec_layout_ptr !llvm.ptr)] i64
  ;; Create constant for field index 1 (len field)
  (constant %c1 (: 1 i32))

  ;; GEP to access the 'len' field (offset 1 in the struct)
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

;; Function that gets the element size field
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
