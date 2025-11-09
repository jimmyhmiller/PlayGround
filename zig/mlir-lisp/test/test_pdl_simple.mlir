// Test: Simple PDL pattern to replace mlsp.string_const with llvm.mlir.zero
module {
  // PDL pattern module
  pdl.pattern @replace_string_const : benefit(1) {
    // Match any mlsp.string_const operation
    %0 = pdl.operation "mlsp.string_const"
    %1 = pdl.type : !llvm.ptr
    pdl.rewrite %0 {
      // Replace with llvm.mlir.zero
      %zero = pdl.operation "llvm.mlir.zero" -> (%1 : !pdl.type)
      pdl.replace %0 with %zero
    }
  }

  // Test payload: operation to match
  func.func @test() -> !llvm.ptr {
    %0 = "mlsp.string_const"() {global = @my_string} : () -> !llvm.ptr
    return %0 : !llvm.ptr
  }

  llvm.mlir.global private constant @my_string("hello") {addr_space = 0 : i32}
}
