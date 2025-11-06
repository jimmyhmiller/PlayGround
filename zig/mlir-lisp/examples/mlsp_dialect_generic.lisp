"builtin.module"() ({
  "irdl.dialect"() <{sym_name = "mlsp"}> ({
    "irdl.operation"() <{sym_name = "identifier"}> ({
      %6 = "irdl.any"() : () -> !irdl.attribute
      "irdl.attributes"(%6) <{attributeValueNames = ["value"]}> : (!irdl.attribute) -> ()
      %7 = "irdl.is"() <{expected = !llvm.ptr}> : () -> !irdl.attribute
      "irdl.results"(%7) <{names = ["result"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "list"}> ({
      %5 = "irdl.is"() <{expected = !llvm.ptr}> : () -> !irdl.attribute
      "irdl.operands"(%5) <{names = ["elements"], variadicity = #irdl<variadicity_array[ variadic]>}> : (!irdl.attribute) -> ()
      "irdl.results"(%5) <{names = ["result"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "get_element"}> ({
      %3 = "irdl.is"() <{expected = !llvm.ptr}> : () -> !irdl.attribute
      "irdl.operands"(%3) <{names = ["list"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
      %4 = "irdl.any"() : () -> !irdl.attribute
      "irdl.attributes"(%4) <{attributeValueNames = ["index"]}> : (!irdl.attribute) -> ()
      "irdl.results"(%3) <{names = ["result"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "get_element_dyn"}> ({
      %1 = "irdl.is"() <{expected = !llvm.ptr}> : () -> !irdl.attribute
      %2 = "irdl.is"() <{expected = i64}> : () -> !irdl.attribute
      "irdl.operands"(%1, %2) <{names = ["list", "index"], variadicity = #irdl<variadicity_array[ single,  single]>}> : (!irdl.attribute, !irdl.attribute) -> ()
      "irdl.results"(%1) <{names = ["result"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "build_operation"}> ({
      %0 = "irdl.is"() <{expected = !llvm.ptr}> : () -> !irdl.attribute
      "irdl.operands"(%0, %0, %0) <{names = ["name", "result_types", "operands"], variadicity = #irdl<variadicity_array[ single,  single,  single]>}> : (!irdl.attribute, !irdl.attribute, !irdl.attribute) -> ()
      "irdl.results"(%0) <{names = ["result"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

