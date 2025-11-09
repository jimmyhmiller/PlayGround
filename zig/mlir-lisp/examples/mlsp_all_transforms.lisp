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
                    (operands %arg0)
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
                                    (result-bindings [%18])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.attribute)
                                    (result-bindings [%19])
                                    (result-types !pdl.attribute))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%20])
                                    (result-types !pdl.operation)
                                    (operands %19 %18)
                                    (attributes {:attributeValueNames ["global"] :opName "mlsp.string_const" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operands %20)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%21])
                                            (result-types !pdl.operation)
                                            (operands %19 %18)
                                            (attributes {:attributeValueNames ["global_name"] :opName "llvm.mlir.addressof" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operands %20 %21)
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
                                    (result-bindings [%12])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%13])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%14])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%15])
                                    (result-types !pdl.operation)
                                    (operands %13 %14 %12)
                                    (attributes {:attributeValueNames [] :opName "mlsp.get_element" :operandSegmentSizes array<i32: 2, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operands %15)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%16])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @get_list_element}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%17])
                                            (result-types !pdl.operation)
                                            (operands %13 %14 %16 %12)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 2, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operands %15 %17)
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
                                    (result-bindings [%3])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%4])
                                    (result-types !pdl.type)
                                    (attributes {:constantType i64}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%5])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%6])
                                    (result-types !pdl.operation)
                                    (operands %5 %3)
                                    (attributes {:attributeValueNames [] :opName "mlsp.identifier" :operandSegmentSizes array<i32: 1, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operands %6)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%7])
                                            (result-types !pdl.attribute)
                                            (attributes {:value (: 0 i64)}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%8])
                                            (result-types !pdl.operation)
                                            (operands %7 %4)
                                            (attributes {:attributeValueNames ["value"] :opName "arith.constant" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                          (operation
                                            (name pdl.result)
                                            (result-bindings [%9])
                                            (result-types !pdl.value)
                                            (operands %8)
                                            (attributes {:index (: 0 i32)}))
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%10])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @create_identifier}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%11])
                                            (result-types !pdl.operation)
                                            (operands %5 %9 %10 %3)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 2, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operands %6 %11)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name transform.sequence)
                            (operands %arg1)
                            (attributes {:failure_propagation_mode (: 1 i32) :operandSegmentSizes array<i32: 1, 0>})
                            (regions
                              (region
                                (block [^bb0]
                                  (arguments [(: %arg2 !transform.any_op)])
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%0])
                                    (result-types !transform.any_op)
                                    (operands %arg2)
                                    (attributes {:pattern_name @mlsp_string_const}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%1])
                                    (result-types !transform.any_op)
                                    (operands %arg2)
                                    (attributes {:pattern_name @mlsp_get_element}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%2])
                                    (result-types !transform.any_op)
                                    (operands %arg2)
                                    (attributes {:pattern_name @mlsp_identifier}))
                                  (operation
                                    (name transform.yield))))))))))
                  (operation
                    (name transform.yield)))))))))))
