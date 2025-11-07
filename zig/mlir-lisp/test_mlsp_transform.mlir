module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.with_pdl_patterns %arg0 : !transform.any_op {
    ^bb0(%payload: !transform.any_op):

      // Pattern: mlsp.string_const → llvm.mlir.addressof
      pdl.pattern @mlsp_string_const_to_llvm : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %mlsp_op = pdl.operation "mlsp.string_const" -> (%ptr_type : !pdl.type)

        pdl.rewrite %mlsp_op {
          // Extract the global attribute and create llvm.mlir.addressof
          %addressof_op = pdl.operation "llvm.mlir.addressof" -> (%ptr_type : !pdl.type)
          pdl.replace %mlsp_op with %addressof_op
        }
      }

      // Pattern: mlsp.identifier → call to create-identifier helper
      pdl.pattern @mlsp_identifier_to_llvm : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %str_ptr = pdl.operand
        %mlsp_op = pdl.operation "mlsp.identifier"(%str_ptr : !pdl.value) -> (%ptr_type : !pdl.type)

        pdl.rewrite %mlsp_op {
          // For now: replace with llvm.call to create-identifier
          // Later: inline the full malloc + GEP + store sequence
          %zero_op = pdl.operation "llvm.mlir.zero" -> (%ptr_type : !pdl.type)
          pdl.replace %mlsp_op with %zero_op
        }
      }

      // Pattern: mlsp.get_element → inline GEP + load sequence
      pdl.pattern @mlsp_get_element_to_llvm : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %list = pdl.operand
        %index = pdl.operand
        %mlsp_op = pdl.operation "mlsp.get_element"(%list, %index : !pdl.value, !pdl.value) -> (%ptr_type : !pdl.type)

        pdl.rewrite %mlsp_op {
          // For now: replace with llvm.call to get-list-element
          // Later: inline the full GEP + load sequence
          %zero_op = pdl.operation "llvm.mlir.zero" -> (%ptr_type : !pdl.type)
          pdl.replace %mlsp_op with %zero_op
        }
      }

      // Pattern: mlsp.list → call to create-list helper
      pdl.pattern @mlsp_list_to_llvm : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %elements = pdl.operands
        %mlsp_op = pdl.operation "mlsp.list"(%elements : !pdl.range<value>) -> (%ptr_type : !pdl.type)

        pdl.rewrite %mlsp_op {
          // For now: replace with llvm.call to create-list
          // Later: inline the full array alloc + malloc + store sequence
          %zero_op = pdl.operation "llvm.mlir.zero" -> (%ptr_type : !pdl.type)
          pdl.replace %mlsp_op with %zero_op
        }
      }

      transform.sequence %payload : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %matched1 = pdl_match @mlsp_string_const_to_llvm in %arg1 : (!transform.any_op) -> !transform.any_op
        %matched2 = pdl_match @mlsp_identifier_to_llvm in %arg1 : (!transform.any_op) -> !transform.any_op
        %matched3 = pdl_match @mlsp_get_element_to_llvm in %arg1 : (!transform.any_op) -> !transform.any_op
        %matched4 = pdl_match @mlsp_list_to_llvm in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.yield
      }
    }
    transform.yield
  }
}
