// Test file for mlsp dialect
// Load the dialect first
irdl.dialect @mlsp {
  irdl.operation @identifier {
    %0 = irdl.any
    irdl.attributes {"value" = %0}
    %1 = irdl.is !llvm.ptr
    irdl.results(result: %1)
  }

  irdl.operation @list {
    %0 = irdl.is !llvm.ptr
    irdl.operands(elements: variadic %0)
    irdl.results(result: %0)
  }

  irdl.operation @get_element {
    %0 = irdl.is !llvm.ptr
    irdl.operands(list: %0)
    %1 = irdl.is i64
    irdl.attributes {"index" = %1}
    irdl.results(result: %0)
  }

  irdl.operation @get_element_dyn {
    %0 = irdl.is !llvm.ptr
    %1 = irdl.is i64
    irdl.operands(list: %0, index: %1)
    irdl.results(result: %0)
  }
}

// Now test using the operations
module {
  llvm.mlir.global internal constant @str_test("test\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str_a("a\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str_b("b\00") {addr_space = 0 : i32}

  func.func @main() -> i64 {
    // Test mlsp.identifier
    %name = "mlsp.identifier"() {value = "test"} : () -> !llvm.ptr

    // Test mlsp.list
    %elem1 = "mlsp.identifier"() {value = "a"} : () -> !llvm.ptr
    %elem2 = "mlsp.identifier"() {value = "b"} : () -> !llvm.ptr
    %list = "mlsp.list"(%elem1, %elem2) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr

    // Test mlsp.get_element (with attribute)
    %extracted1 = "mlsp.get_element"(%list) {index = 0 : i64} : (!llvm.ptr) -> !llvm.ptr

    // Test mlsp.get_element_dyn (with heterogeneous operands)
    %idx = arith.constant 1 : i64
    %extracted2 = "mlsp.get_element_dyn"(%list, %idx) : (!llvm.ptr, i64) -> !llvm.ptr

    %result = arith.constant 42 : i64
    return %result : i64
  }
}
