;; Simpler example: Direct struct field access for CValueLayout
;;
;; CValueLayout struct layout:
;; !llvm.struct<(i8, [7 x i8], ptr, i64, i64, i64, ptr, ptr)>
;; Fields: [type_tag, padding, data_ptr, data_len, data_capacity, data_elem_size, extra_ptr1, extra_ptr2]

;; Function that extracts the type tag from a CValueLayout pointer
(defn get_value_type [(: %value_ptr !llvm.ptr)] i8
  ;; GEP to access the 'type_tag' field (offset 0 in the struct)
  ;; Using rawConstantIndices means no dynamic indices are passed
  (op %type_tag_ptr (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value_ptr]))

  ;; Load the i8 type tag value from the pointer
  (op %type_tag (: i8) (llvm.load [%type_tag_ptr]))

  ;; Return the type tag
  (return %type_tag))

;; Function that gets the data pointer from a CValueLayout
;; For atoms: points to string data
;; For collections: points to element array
(defn get_value_data_ptr [(: %value_ptr !llvm.ptr)] !llvm.ptr
  ;; GEP to access the 'data_ptr' field (offset 2 in the struct)
  (op %data_ptr_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type !llvm.ptr :rawConstantIndices array<i32: 2>} [%value_ptr]))

  ;; Load the pointer value
  (op %data_ptr (: !llvm.ptr) (llvm.load [%data_ptr_field]))

  ;; Return the data pointer
  (return %data_ptr))

;; Function that gets the data length from a CValueLayout
;; For atoms: string length in bytes
;; For collections: number of elements
(defn get_value_data_len [(: %value_ptr !llvm.ptr)] i64
  ;; GEP to access the 'data_len' field (offset 3 in the struct)
  (op %len_field_ptr (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 3>} [%value_ptr]))

  ;; Load the i64 length value
  (op %len (: i64) (llvm.load [%len_field_ptr]))

  ;; Return the length
  (return %len))

;; Function that gets extra_ptr1
;; For attr_expr: pointer to wrapped CValueLayout
;; For has_type: pointer to value CValueLayout
;; For function_type: pointer to inputs CVectorLayout
(defn get_value_extra_ptr1 [(: %value_ptr !llvm.ptr)] !llvm.ptr
  ;; GEP to access the 'extra_ptr1' field (offset 6 in the struct)
  (op %extra_ptr1_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type !llvm.ptr :rawConstantIndices array<i32: 6>} [%value_ptr]))

  ;; Load the pointer value
  (op %extra_ptr1 (: !llvm.ptr) (llvm.load [%extra_ptr1_field]))

  ;; Return the pointer
  (return %extra_ptr1))

;; Function that gets extra_ptr2
;; For has_type: pointer to type_expr CValueLayout
;; For function_type: pointer to results CVectorLayout
(defn get_value_extra_ptr2 [(: %value_ptr !llvm.ptr)] !llvm.ptr
  ;; GEP to access the 'extra_ptr2' field (offset 7 in the struct)
  (op %extra_ptr2_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type !llvm.ptr :rawConstantIndices array<i32: 7>} [%value_ptr]))

  ;; Load the pointer value
  (op %extra_ptr2 (: !llvm.ptr) (llvm.load [%extra_ptr2_field]))

  ;; Return the pointer
  (return %extra_ptr2))

;; Example: Get atom string data (for identifier, number, string, etc.)
(defn get_atom_string [(: %value_ptr !llvm.ptr)] ()
  ;; Get data pointer
  (op %data_ptr_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type !llvm.ptr :rawConstantIndices array<i32: 2>} [%value_ptr]))
  (op %string_data (: !llvm.ptr) (llvm.load [%data_ptr_field]))

  ;; Get string length
  (op %len_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 3>} [%value_ptr]))
  (op %string_len (: i64) (llvm.load [%len_field]))

  ;; Now you have both pointer and length for the string

  (op (func.return [])))

;; Main function that demonstrates the value struct access
(defn main [] i32
  ;; Create a CValueLayout on the stack and initialize it
  ;; We'll simulate an identifier with type_tag=0, data pointing to a string

  ;; Allocate space for CValueLayout (56 bytes)
  (constant %size (: 56 i64))
  (op %value_layout (: !llvm.ptr)
      (llvm.alloca {:elem_type !llvm.array<56 x i8>} [%size]))

  ;; Set type_tag to 0 (identifier)
  (constant %type_tag (: 0 i8))
  (op %type_ptr (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value_layout]))
  (op (llvm.store [%type_tag %type_ptr]))

  ;; Set data_len to 10 (pretend we have a 10-byte string)
  (constant %len (: 10 i64))
  (op %len_ptr (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 3>} [%value_layout]))
  (op (llvm.store [%len %len_ptr]))

  ;; Now call our get_value_type function to read back the type tag
  (call %retrieved_type @get_value_type %value_layout i8)

  ;; Call get_value_data_len to read back the length
  (call %retrieved_len @get_value_data_len %value_layout i64)

  ;; Convert the retrieved values to i32 for printing/returning
  (op %type_as_i32 (: i32) (llvm.zext [%retrieved_type]))
  (op %len_as_i32 (: i32) (llvm.trunc [%retrieved_len]))

  ;; Add them together just to show we're using both values
  (op %result (: i32) (llvm.add [%type_as_i32 %len_as_i32]))

  ;; Should return 0 + 10 = 10
  (return %result))
