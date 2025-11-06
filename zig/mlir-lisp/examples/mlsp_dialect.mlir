// MLSP Dialect Definition in MLIR Syntax
// Testing if direct MLIR syntax works better than our lisp representation

irdl.dialect @mlsp {

  // mlsp.identifier - Create identifier atom
  // Syntax: %val = mlsp.identifier "name"
  irdl.operation @identifier {
    %any_attr = irdl.any
    irdl.attributes {
      "value" = %any_attr
    }

    %ptr_type = irdl.is !llvm.ptr
    irdl.results(result: %ptr_type)
  }

  // mlsp.list - Create list from elements
  // Syntax: %list = mlsp.list(%val1, %val2, %val3)
  irdl.operation @list {
    %ptr_type = irdl.is !llvm.ptr
    irdl.operands(elements: variadic %ptr_type)
    irdl.results(result: %ptr_type)
  }

  // mlsp.get_element - Extract list element by index
  // Syntax: %elem = mlsp.get_element %list {index = 0}
  irdl.operation @get_element {
    // Operand: list pointer
    %ptr_type = irdl.is !llvm.ptr
    irdl.operands(list: %ptr_type)

    // Attribute: index (accept any attribute)
    %any_attr = irdl.any
    irdl.attributes {
      "index" = %any_attr
    }

    // Result: element pointer
    irdl.results(result: %ptr_type)
  }

  // mlsp.get_element_dyn - Extract list element by dynamic index
  // Syntax: %elem = mlsp.get_element_dyn %list, %index
  // Testing heterogeneous operands
  irdl.operation @get_element_dyn {
    %ptr_type = irdl.is !llvm.ptr
    %i64_type = irdl.is i64
    irdl.operands(list: %ptr_type, index: %i64_type)
    irdl.results(result: %ptr_type)
  }

  // mlsp.build_operation - High-level operation builder
  // Syntax: %op = mlsp.build_operation %name, %result_types, %operands
  irdl.operation @build_operation {
    %ptr_type = irdl.is !llvm.ptr
    irdl.operands(name: %ptr_type, result_types: %ptr_type, operands: %ptr_type)
    irdl.results(result: %ptr_type)
  }
}
