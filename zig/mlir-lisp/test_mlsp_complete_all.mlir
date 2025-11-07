module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.with_pdl_patterns %arg0 : !transform.any_op {
    ^bb0(%payload: !transform.any_op):

      // Pattern: mlsp.string_const → llvm.mlir.addressof
      pdl.pattern @mlsp_string_const : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %global_attr = pdl.attribute
        %op = pdl.operation "mlsp.string_const" {"global" = %global_attr} -> (%ptr_type : !pdl.type)
        pdl.rewrite %op {
          %new_op = pdl.operation "llvm.mlir.addressof" {"global_name" = %global_attr} -> (%ptr_type : !pdl.type)
          pdl.replace %op with %new_op
        }
      }

      // Pattern: mlsp.get_element → call @get_list_element
      pdl.pattern @mlsp_get_element : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %list = pdl.operand
        %index = pdl.operand
        %op = pdl.operation "mlsp.get_element"(%list, %index : !pdl.value, !pdl.value) -> (%ptr_type : !pdl.type)
        pdl.rewrite %op {
          %callee_attr = pdl.attribute = @get_list_element
          %new_op = pdl.operation "func.call"(%list, %index : !pdl.value, !pdl.value) {"callee" = %callee_attr} -> (%ptr_type : !pdl.type)
          pdl.replace %op with %new_op
        }
      }

      // Pattern: mlsp.identifier → call @create_identifier
      pdl.pattern @mlsp_identifier : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %i64_type = pdl.type : i64
        %str_ptr = pdl.operand
        %op = pdl.operation "mlsp.identifier"(%str_ptr : !pdl.value) -> (%ptr_type : !pdl.type)
        pdl.rewrite %op {
          %zero_attr = pdl.attribute = 0 : i64
          %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%i64_type : !pdl.type)
          %zero_result = pdl.result 0 of %zero_op
          %callee_attr = pdl.attribute = @create_identifier
          %new_op = pdl.operation "func.call"(%str_ptr, %zero_result : !pdl.value, !pdl.value) {"callee" = %callee_attr} -> (%ptr_type : !pdl.type)
          pdl.replace %op with %new_op
        }
      }

      // Pattern: mlsp.list with 2 operands → call @create_list_2
      pdl.pattern @mlsp_list_2 : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %elem0 = pdl.operand
        %elem1 = pdl.operand
        %op = pdl.operation "mlsp.list"(%elem0, %elem1 : !pdl.value, !pdl.value) -> (%ptr_type : !pdl.type)
        pdl.rewrite %op {
          %callee_attr = pdl.attribute = @create_list_2
          %new_op = pdl.operation "func.call"(%elem0, %elem1 : !pdl.value, !pdl.value) {"callee" = %callee_attr} -> (%ptr_type : !pdl.type)
          pdl.replace %op with %new_op
        }
      }

      // Pattern: mlsp.list with 3 operands → call @create_list_3
      pdl.pattern @mlsp_list_3 : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %elem0 = pdl.operand
        %elem1 = pdl.operand
        %elem2 = pdl.operand
        %op = pdl.operation "mlsp.list"(%elem0, %elem1, %elem2 : !pdl.value, !pdl.value, !pdl.value) -> (%ptr_type : !pdl.type)
        pdl.rewrite %op {
          %callee_attr = pdl.attribute = @create_list_3
          %new_op = pdl.operation "func.call"(%elem0, %elem1, %elem2 : !pdl.value, !pdl.value, !pdl.value) {"callee" = %callee_attr} -> (%ptr_type : !pdl.type)
          pdl.replace %op with %new_op
        }
      }

      // Pattern: mlsp.list with 4 operands → call @create_list_4
      pdl.pattern @mlsp_list_4 : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %elem0 = pdl.operand
        %elem1 = pdl.operand
        %elem2 = pdl.operand
        %elem3 = pdl.operand
        %op = pdl.operation "mlsp.list"(%elem0, %elem1, %elem2, %elem3 : !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%ptr_type : !pdl.type)
        pdl.rewrite %op {
          %callee_attr = pdl.attribute = @create_list_4
          %new_op = pdl.operation "func.call"(%elem0, %elem1, %elem2, %elem3 : !pdl.value, !pdl.value, !pdl.value, !pdl.value) {"callee" = %callee_attr} -> (%ptr_type : !pdl.type)
          pdl.replace %op with %new_op
        }
      }

      transform.sequence %payload : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %matched1 = pdl_match @mlsp_string_const in %arg1 : (!transform.any_op) -> !transform.any_op
        %matched2 = pdl_match @mlsp_get_element in %arg1 : (!transform.any_op) -> !transform.any_op
        %matched3 = pdl_match @mlsp_identifier in %arg1 : (!transform.any_op) -> !transform.any_op
        %matched4 = pdl_match @mlsp_list_2 in %arg1 : (!transform.any_op) -> !transform.any_op
        %matched5 = pdl_match @mlsp_list_3 in %arg1 : (!transform.any_op) -> !transform.any_op
        %matched6 = pdl_match @mlsp_list_4 in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.yield
      }
    }
    transform.yield
  }
}
