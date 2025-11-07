(mlir
  (operation
    (name builtin.module)
    (attributes {:transform.with_named_sequence true})
    (regions
      (region
        (block
          (arguments [])
          (operation
            (name transform.named_sequence)
            (attributes {:function_type (!function (inputs !transform.any_op) (results)) :sym_name @__transform_main})
            (regions
              (region
                (block [^bb0]
                  (arguments [(: %arg0 !transform.any_op)])
                  (operation
                    (name transform.with_pdl_patterns)
                    (operand-uses %arg0)
                    (regions
                      (region
                        (block [^bb0]
                          (arguments [(: %arg1 !transform.any_op)])
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_string_const})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%42])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.attribute)
                                    (result-bindings [%43])
                                    (result-types !pdl.attribute))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%44])
                                    (result-types !pdl.operation)
                                    (operand-uses %43 %42)
                                    (attributes {:attributeValueNames ["global"] :opName "mlsp.string_const" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %44)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%45])
                                            (result-types !pdl.operation)
                                            (operand-uses %43 %42)
                                            (attributes {:attributeValueNames ["global_name"] :opName "llvm.mlir.addressof" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %44 %45)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_get_element})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%36])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%37])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%38])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%39])
                                    (result-types !pdl.operation)
                                    (operand-uses %37 %38 %36)
                                    (attributes {:attributeValueNames [] :opName "mlsp.get_element" :operandSegmentSizes array<i32: 2, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %39)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%40])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @get_list_element}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%41])
                                            (result-types !pdl.operation)
                                            (operand-uses %37 %38 %40 %36)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 2, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %39 %41)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_identifier})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%27])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%28])
                                    (result-types !pdl.type)
                                    (attributes {:constantType i64}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%29])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%30])
                                    (result-types !pdl.operation)
                                    (operand-uses %29 %27)
                                    (attributes {:attributeValueNames [] :opName "mlsp.identifier" :operandSegmentSizes array<i32: 1, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %30)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%31])
                                            (result-types !pdl.attribute)
                                            (attributes {:value (: 0 i64)}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%32])
                                            (result-types !pdl.operation)
                                            (operand-uses %31 %28)
                                            (attributes {:attributeValueNames ["value"] :opName "arith.constant" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                          (operation
                                            (name pdl.result)
                                            (result-bindings [%33])
                                            (result-types !pdl.value)
                                            (operand-uses %32)
                                            (attributes {:index (: 0 i32)}))
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%34])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @create_identifier}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%35])
                                            (result-types !pdl.operation)
                                            (operand-uses %29 %33 %34 %27)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 2, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %30 %35)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_list_2})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%21])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%22])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%23])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%24])
                                    (result-types !pdl.operation)
                                    (operand-uses %22 %23 %21)
                                    (attributes {:attributeValueNames [] :opName "mlsp.list" :operandSegmentSizes array<i32: 2, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %24)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%25])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @create_list_2}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%26])
                                            (result-types !pdl.operation)
                                            (operand-uses %22 %23 %25 %21)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 2, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %24 %26)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_list_3})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%14])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%15])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%16])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%17])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%18])
                                    (result-types !pdl.operation)
                                    (operand-uses %15 %16 %17 %14)
                                    (attributes {:attributeValueNames [] :opName "mlsp.list" :operandSegmentSizes array<i32: 3, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %18)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%19])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @create_list_3}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%20])
                                            (result-types !pdl.operation)
                                            (operand-uses %15 %16 %17 %19 %14)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 3, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %18 %20)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_list_4})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%6])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%7])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%8])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%9])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%10])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%11])
                                    (result-types !pdl.operation)
                                    (operand-uses %7 %8 %9 %10 %6)
                                    (attributes {:attributeValueNames [] :opName "mlsp.list" :operandSegmentSizes array<i32: 4, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %11)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%12])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @create_list_4}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%13])
                                            (result-types !pdl.operation)
                                            (operand-uses %7 %8 %9 %10 %12 %6)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 4, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %11 %13)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name transform.sequence)
                            (operand-uses %arg1)
                            (attributes {:failure_propagation_mode (: 1 i32) :operandSegmentSizes array<i32: 1, 0>})
                            (regions
                              (region
                                (block [^bb0]
                                  (arguments [(: %arg2 !transform.any_op)])
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%0])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_string_const}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%1])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_get_element}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%2])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_identifier}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%3])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_list_2}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%4])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_list_3}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%5])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_list_4}))
                                  (operation
                                    (name transform.yield))))))))))
                  (operation
                    (name transform.yield)))))))))))
