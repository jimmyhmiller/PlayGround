// Test with transform.apply_patterns to actually apply PDL patterns
module {
  // Payload function
  func.func @test() -> !llvm.ptr {
    %0 = "mlsp.string_const"() {global = @my_string} : () -> !llvm.ptr
    return %0 : !llvm.ptr
  }

  llvm.mlir.global private constant @my_string("hello") {addr_space = 0 : i32}

  // Transform module
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      // Find all functions
      %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op

      // Apply PDL patterns to each function
      transform.foreach %funcs : !transform.any_op {
      ^bb0(%func: !transform.any_op):
        // Apply patterns greedily to the function
        transform.apply_patterns to %func {
          // Define PDL pattern inline
          transform.with_pdl_patterns %func : !transform.any_op {
          ^bb1(%payload: !transform.any_op):
            pdl.pattern @replace_string_const : benefit(1) {
              %op = pdl.operation "mlsp.string_const"
              %type = pdl.type : !llvm.ptr
              pdl.rewrite %op {
                %zero = pdl.operation "llvm.mlir.zero" -> (%type : !pdl.type)
                pdl.replace %op with %zero
              }
            }
          }
        } : !transform.any_op
      }

      transform.yield
    }
  }
}
