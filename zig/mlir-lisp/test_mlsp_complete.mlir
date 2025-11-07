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

      // Pattern: mlsp.get_element → func.call @get_list_element
      pdl.pattern @mlsp_get_element : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %i64_type = pdl.type : i64
        %list = pdl.operand
        %index = pdl.operand
        %op = pdl.operation "mlsp.get_element"(%list, %index : !pdl.value, !pdl.value) -> (%ptr_type : !pdl.type)

        pdl.rewrite %op {
          %callee_attr = pdl.attribute = @get_list_element
          %new_op = pdl.operation "func.call"(%list, %index : !pdl.value, !pdl.value) {"callee" = %callee_attr} -> (%ptr_type : !pdl.type)
          pdl.replace %op with %new_op
        }
      }

      transform.sequence %payload : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %matched1 = pdl_match @mlsp_string_const in %arg1 : (!transform.any_op) -> !transform.any_op
        %matched2 = pdl_match @mlsp_get_element in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.yield
      }
    }
    transform.yield
  }
}
