(mlir
  (operation
    (name builtin.module)
    (regions
      (region
        (block
          (arguments [])
          (defn get_list_element [
            (: %list_value !llvm.ptr)
            (: %index i64)
          ] !llvm.ptr
            (constant %ptr_size (: 8 i64))
            (op %data_ptr_field (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%list_value]))
            (op %data_ptr (: !llvm.ptr) (llvm.load [%data_ptr_field]))
            (op %offset (: i64) (llvm.mul [%index %ptr_size]))
            (op %elem_pp (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%data_ptr %offset]))
            (op %elem_ptr (: !llvm.ptr) (llvm.load [%elem_pp]))
            (return %elem_ptr))

          (operation
            (name llvm.mlir.global)
            (attributes {:addr_space (: 0 i32) :alignment (: 1 i64) :constant true :dso_local true :linkage #llvm.linkage<internal> :sym_name @str_test :unnamed_addr true :value "test\00"}))

          (defn testFunc [(: %arg !llvm.ptr)] !llvm.ptr
            (constant %c0 (: 0 i64))
            (op %str_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_test}))
            (op %elem (: !llvm.ptr) (mlsp.get_element [%arg %c0]))
            (return %elem)))))))
