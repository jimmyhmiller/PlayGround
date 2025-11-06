"builtin.module"() ({
  "irdl.dialect"() <{sym_name = "mlsp"}> ({
    "irdl.operation"() <{sym_name = "identifier"}> ({
      %13 = "irdl.any"() : () -> !irdl.attribute
      "irdl.attributes"(%13) <{attributeValueNames = ["value"]}> : (!irdl.attribute) -> ()
      %14 = "irdl.is"() <{expected = !llvm.ptr}> : () -> !irdl.attribute
      "irdl.results"(%14) <{names = ["result"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "list"}> ({
      %12 = "irdl.is"() <{expected = !llvm.ptr}> : () -> !irdl.attribute
      "irdl.operands"(%12) <{names = ["elements"], variadicity = #irdl<variadicity_array[ variadic]>}> : (!irdl.attribute) -> ()
      "irdl.results"(%12) <{names = ["result"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "get_element"}> ({
      %10 = "irdl.is"() <{expected = !llvm.ptr}> : () -> !irdl.attribute
      "irdl.operands"(%10) <{names = ["list"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
      %11 = "irdl.is"() <{expected = i64}> : () -> !irdl.attribute
      "irdl.attributes"(%11) <{attributeValueNames = ["index"]}> : (!irdl.attribute) -> ()
      "irdl.results"(%10) <{names = ["result"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "get_element_dyn"}> ({
      %8 = "irdl.is"() <{expected = !llvm.ptr}> : () -> !irdl.attribute
      %9 = "irdl.is"() <{expected = i64}> : () -> !irdl.attribute
      "irdl.operands"(%8, %9) <{names = ["list", "index"], variadicity = #irdl<variadicity_array[ single,  single]>}> : (!irdl.attribute, !irdl.attribute) -> ()
      "irdl.results"(%8) <{names = ["result"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "llvm.mlir.global"() <{addr_space = 0 : i32, constant, global_type = !llvm.array<5 x i8>, linkage = #llvm.linkage<internal>, sym_name = "str_test", unnamed_addr = 0 : i64, value = "test\00", visibility_ = 0 : i64}> ({
    }) : () -> ()
    "llvm.mlir.global"() <{addr_space = 0 : i32, constant, global_type = !llvm.array<2 x i8>, linkage = #llvm.linkage<internal>, sym_name = "str_a", unnamed_addr = 0 : i64, value = "a\00", visibility_ = 0 : i64}> ({
    }) : () -> ()
    "llvm.mlir.global"() <{addr_space = 0 : i32, constant, global_type = !llvm.array<2 x i8>, linkage = #llvm.linkage<internal>, sym_name = "str_b", unnamed_addr = 0 : i64, value = "b\00", visibility_ = 0 : i64}> ({
    }) : () -> ()
    "func.func"() <{function_type = () -> i64, sym_name = "main"}> ({
      %0 = "mlsp.identifier"() {value = "test"} : () -> !llvm.ptr
      %1 = "mlsp.identifier"() {value = "a"} : () -> !llvm.ptr
      %2 = "mlsp.identifier"() {value = "b"} : () -> !llvm.ptr
      %3 = "mlsp.list"(%1, %2) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
      %4 = "mlsp.get_element"(%3) {index = 0 : i64} : (!llvm.ptr) -> !llvm.ptr
      %5 = "arith.constant"() <{value = 1 : i64}> : () -> i64
      %6 = "mlsp.get_element_dyn"(%3, %5) : (!llvm.ptr, i64) -> !llvm.ptr
      %7 = "arith.constant"() <{value = 42 : i64}> : () -> i64
      "func.return"(%7) : (i64) -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

