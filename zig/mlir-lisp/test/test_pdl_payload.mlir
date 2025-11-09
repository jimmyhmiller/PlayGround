// Payload IR to be transformed
module {
  func.func @test() -> !llvm.ptr {
    %0 = "mlsp.string_const"() {global = @my_string} : () -> !llvm.ptr
    return %0 : !llvm.ptr
  }

  llvm.mlir.global private constant @my_string("hello") {addr_space = 0 : i32}
}
