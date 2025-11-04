;; Example: Direct struct field access for CValueLayout
;;
;; This example demonstrates how to manipulate CValueLayout structs directly
;; from MLIR code, accessing fields without function call overhead.
;;
;; CValueLayout struct layout:
;; !llvm.struct<(i8, [7 x i8], ptr, i64, i64, i64, ptr, ptr)>
;; Fields: [type_tag, padding, data_ptr, data_len, data_capacity, data_elem_size, extra_ptr1, extra_ptr2]

;; Function that extracts the type tag from a CValueLayout pointer
;; This tells you what kind of value this is (identifier, number, list, etc.)
(defn get_value_type [(: %value_ptr !llvm.ptr)] i8
  ;; GEP to access the 'type_tag' field (offset 0 in the struct)
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
  ;; Create constant for field index 2 (data_ptr field, skipping padding)

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
  ;; Create constant for field index 3 (data_len field)

  ;; GEP to access the 'data_len' field (offset 3 in the struct)
  (op %len_field_ptr (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 3>} [%value_ptr]))

  ;; Load the i64 length value
  (op %len (: i64) (llvm.load [%len_field_ptr]))

  ;; Return the length
  (return %len))

;; Function that gets the element size for collections
(defn get_value_elem_size [(: %value_ptr !llvm.ptr)] i64
  ;; Create constant for field index 5 (data_elem_size field)

  ;; GEP to access the 'data_elem_size' field (offset 5 in the struct)
  (op %elem_size_ptr (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 5>} [%value_ptr]))

  ;; Load the element size
  (op %elem_size (: i64) (llvm.load [%elem_size_ptr]))

  ;; Return it
  (return %elem_size))

;; Function that gets extra_ptr1
;; For attr_expr: pointer to wrapped CValueLayout
;; For has_type: pointer to value CValueLayout
;; For function_type: pointer to inputs CVectorLayout
(defn get_value_extra_ptr1 [(: %value_ptr !llvm.ptr)] !llvm.ptr
  ;; Create constant for field index 6 (extra_ptr1 field)

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
  ;; Create constant for field index 7 (extra_ptr2 field)

  ;; GEP to access the 'extra_ptr2' field (offset 7 in the struct)
  (op %extra_ptr2_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type !llvm.ptr :rawConstantIndices array<i32: 7>} [%value_ptr]))

  ;; Load the pointer value
  (op %extra_ptr2 (: !llvm.ptr) (llvm.load [%extra_ptr2_field]))

  ;; Return the pointer
  (return %extra_ptr2))

;; More complex example: check if a value is an identifier with a specific name
;; This shows how you can work with atom types
(defn is_identifier_named [(: %value_ptr !llvm.ptr) (: %expected_name !llvm.ptr) (: %expected_len i64)] i1
  ;; Constants for field indices

  ;; Constants for value type enum
  ;; Note: These values should match ValueType enum in your codebase
  (constant %identifier_tag (: 0 i8))

  ;; Get type tag
  (op %type_tag_ptr (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value_ptr]))
  (op %type_tag (: i8) (llvm.load [%type_tag_ptr]))

  ;; Check if it's an identifier
  (op %is_identifier (: i1) (llvm.icmp {:predicate 0} [%type_tag %identifier_tag]))

  ;; Get data length
  (op %len_ptr (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 3>} [%value_ptr]))
  (op %actual_len (: i64) (llvm.load [%len_ptr]))

  ;; Check if length matches
  (op %len_matches (: i1) (llvm.icmp {:predicate 0} [%actual_len %expected_len]))

  ;; Both conditions must be true
  (op %both_true (: i1) (llvm.and [%is_identifier %len_matches]))

  ;; TODO: Would also need to compare string contents using memcmp
  ;; For now, just return the preliminary check

  (return %both_true))

;; Example: Get collection information
;; Given a CValueLayout representing a list/vector/map, extract useful info
(defn get_collection_info [(: %value_ptr !llvm.ptr)] ()
  ;; Constants

  ;; Get element array pointer
  (op %data_ptr_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type !llvm.ptr :rawConstantIndices array<i32: 2>} [%value_ptr]))
  (op %elements (: !llvm.ptr) (llvm.load [%data_ptr_field]))

  ;; Get number of elements
  (op %len_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 3>} [%value_ptr]))
  (op %num_elements (: i64) (llvm.load [%len_field]))

  ;; Get capacity
  (op %cap_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 4>} [%value_ptr]))
  (op %capacity (: i64) (llvm.load [%cap_field]))

  ;; Get element size
  (op %elem_size_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 5>} [%value_ptr]))
  (op %elem_size (: i64) (llvm.load [%elem_size_field]))

  ;; TODO: Could iterate over elements here
  ;; This is just a skeleton showing the struct access pattern

  (op (func.return [])))

;; Example: Access typed literal (has_type) components
;; For a value with type annotation, get both the value and type_expr
(defn get_typed_literal_parts [(: %value_ptr !llvm.ptr)] ()
  ;; Constants

  ;; Get the value pointer (extra_ptr1)
  (op %value_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type !llvm.ptr :rawConstantIndices array<i32: 6>} [%value_ptr]))
  (op %inner_value (: !llvm.ptr) (llvm.load [%value_field]))

  ;; Get the type_expr pointer (extra_ptr2)
  (op %type_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type !llvm.ptr :rawConstantIndices array<i32: 7>} [%value_ptr]))
  (op %type_expr (: !llvm.ptr) (llvm.load [%type_field]))

  ;; Now you have pointers to both the value and its type annotation
  ;; Both are themselves CValueLayout pointers you can recursively access

  (op (func.return [])))

;; Advanced example: Manual construction of a simple CValueLayout
;; This shows how you could build these structs in MLIR code
(defn create_number_value_layout [] !llvm.ptr
  ;; Constants for initialization
  (constant %size (: 1 i64))
  (constant %zero_i64 (: 0 i64))

  ;; Value type tag for number (assuming it's 1 - check your enum!)
  (constant %number_tag (: 1 i8))

  ;; Allocate space for the struct on the stack
  (op %struct_ptr (: !llvm.ptr) (llvm.alloca {:elem_type i8} [%size]))

  ;; Create null pointer constant
  (op %null_ptr (: !llvm.ptr) (llvm.mlir.zero []))

  ;; Initialize 'type_tag' field to number (offset 0)
  (op %type_tag_ptr (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%struct_ptr]))
  (op (llvm.store [%number_tag %type_tag_ptr]))

  ;; Initialize 'data_ptr' field to null for now (offset 2)
  ;; In a real implementation, you'd allocate string storage here
  (op %data_ptr_field (: !llvm.ptr)
      (llvm.getelementptr {:elem_type !llvm.ptr :rawConstantIndices array<i32: 2>} [%struct_ptr]))
  (op (llvm.store [%null_ptr %data_ptr_field]))

  ;; Initialize 'data_len' field to 0 (offset 3)
  (op %len_ptr (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 3>} [%struct_ptr]))
  (op (llvm.store [%zero_i64 %len_ptr]))

  ;; Return pointer to the struct
  (return %struct_ptr))

;; Key benefits of this approach:
;;
;; 1. **Performance**: Direct memory access, no function call overhead
;; 2. **Type Safety**: MLIR type system understands the struct layout
;; 3. **Flexibility**: Can manipulate values directly in MLIR
;; 4. **Introspection**: Can check types and extract data at runtime
;;
;; Tradeoffs:
;;
;; 1. **ABI Lock-in**: Struct layout becomes part of your stable API
;; 2. **No Encapsulation**: Internal structure is exposed
;; 3. **Complexity**: Need to understand the flat struct design
;; 4. **Type Safety**: Must manually ensure you're accessing correct fields for each type

;; Main function: Demonstrates creating and manipulating CValueLayout structs
(defn main [] i32
  ;; Allocate space for three CValueLayout structs (56 bytes each)
  (constant %size (: 56 i64))

  ;; Test 1: Create and test an identifier value
  (op %value1 (: !llvm.ptr)
      (llvm.alloca {:elem_type !llvm.array<56 x i8>} [%size]))

  (constant %identifier_tag (: 0 i8))
  (constant %len1 (: 5 i64))
  (op %type_ptr1 (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value1]))
  (op (llvm.store [%identifier_tag %type_ptr1]))
  (op %len_ptr1 (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 3>} [%value1]))
  (op (llvm.store [%len1 %len_ptr1]))

  ;; Test get_value_type and get_value_data_len on value1
  (call %type1 @get_value_type %value1 i8)
  (call %len1_retrieved @get_value_data_len %value1 i64)

  ;; Test 2: Create a list value and use multiple accessors
  (op %value2 (: !llvm.ptr)
      (llvm.alloca {:elem_type !llvm.array<56 x i8>} [%size]))

  (constant %list_tag (: 3 i8))
  (constant %len2 (: 10 i64))
  (constant %elem_size (: 8 i64))
  (op %type_ptr2 (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value2]))
  (op (llvm.store [%list_tag %type_ptr2]))
  (op %len_ptr2 (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 3>} [%value2]))
  (op (llvm.store [%len2 %len_ptr2]))
  (op %elem_size_ptr2 (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 5>} [%value2]))
  (op (llvm.store [%elem_size %elem_size_ptr2]))

  ;; Test multiple helper functions on value2
  (call %type2 @get_value_type %value2 i8)
  (call %len2_retrieved @get_value_data_len %value2 i64)
  (call %elem_size_retrieved @get_value_elem_size %value2 i64)
  (call %data_ptr2 @get_value_data_ptr %value2 !llvm.ptr)

  ;; Test get_collection_info (doesn't return anything, just exercises the function)
  (call @get_collection_info %value2 ())

  ;; Test 3: Create a number value to demonstrate create_number_value_layout pattern
  (op %value3 (: !llvm.ptr)
      (llvm.alloca {:elem_type !llvm.array<56 x i8>} [%size]))

  (constant %number_tag (: 1 i8))
  (constant %len3 (: 3 i64))
  (op %type_ptr3 (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value3]))
  (op (llvm.store [%number_tag %type_ptr3]))
  (op %len_ptr3 (: !llvm.ptr)
      (llvm.getelementptr {:elem_type i64 :rawConstantIndices array<i32: 3>} [%value3]))
  (op (llvm.store [%len3 %len_ptr3]))

  (call %type3 @get_value_type %value3 i8)
  (call %len3_retrieved @get_value_data_len %value3 i64)
  (call %ptr3 @get_value_extra_ptr1 %value3 !llvm.ptr)
  (call %ptr4 @get_value_extra_ptr2 %value3 !llvm.ptr)

  ;; Calculate result to demonstrate all functions were called
  ;; Sum all the retrieved values: type1 + len1 + type2 + len2 + type3 + len3
  ;; Expected: 0 + 5 + 3 + 10 + 1 + 3 = 22
  (op %type1_i32 (: i32) (llvm.zext [%type1]))
  (op %type2_i32 (: i32) (llvm.zext [%type2]))
  (op %type3_i32 (: i32) (llvm.zext [%type3]))
  (op %len1_i32 (: i32) (llvm.trunc [%len1_retrieved]))
  (op %len2_i32 (: i32) (llvm.trunc [%len2_retrieved]))
  (op %len3_i32 (: i32) (llvm.trunc [%len3_retrieved]))

  (op %sum1 (: i32) (llvm.add [%type1_i32 %len1_i32]))
  (op %sum2 (: i32) (llvm.add [%type2_i32 %len2_i32]))
  (op %sum3 (: i32) (llvm.add [%type3_i32 %len3_i32]))
  (op %sum4 (: i32) (llvm.add [%sum1 %sum2]))
  (op %result (: i32) (llvm.add [%sum4 %sum3]))

  ;; Return the result (should be 22: 0+5+3+10+1+3)
  (return %result))
