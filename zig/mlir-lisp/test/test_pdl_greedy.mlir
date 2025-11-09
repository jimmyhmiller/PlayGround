// Test: Apply PDL patterns using greedy rewriter (not transform dialect)
// This module contains only PDL patterns
module {
  pdl.pattern @replace_string_const : benefit(1) {
    %op = pdl.operation "mlsp.string_const"
    %type = pdl.type : !llvm.ptr
    pdl.rewrite %op {
      %zero = pdl.operation "llvm.mlir.zero" -> (%type : !pdl.type)
      pdl.replace %op with %zero
    }
  }
}
