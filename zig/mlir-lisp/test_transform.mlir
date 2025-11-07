// Simple test to verify transform-interpreter works
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
      // Replace mlsp.string_const with llvm.mlir.zero
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%payload: !transform.any_op):
        // PDL pattern: match mlsp.string_const
        pdl.pattern @replace_string_const : benefit(1) {
          %op = pdl.operation "mlsp.string_const"
          %type = pdl.type : !llvm.ptr
          pdl.rewrite %op {
            %zero = pdl.operation "llvm.mlir.zero" -> (%type : !pdl.type)
            pdl.replace %op with %zero
          }
        }

        // Apply patterns
        transform.sequence %payload : !transform.any_op failures(suppress) {
        ^bb1(%arg1: !transform.any_op):
          transform.yield
        }
      }
      transform.yield
    }
  }
}
