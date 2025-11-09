(mlir
  (operation
    (name builtin.module)
    (regions
      (region
        (block
          (arguments [])
          (operation
            (name builtin.module)
            (regions
              (region
                (block
                  (arguments [])
                  (operation
                    (name irdl.dialect)
                    (attributes {:sym_name @mymath})
                    (regions
                      (region
                        (block
                          (arguments [])
                          (operation
                            (name irdl.operation)
                            (attributes {:sym_name @add})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name irdl.is)
                                    (result-bindings [%9])
                                    (result-types !irdl.attribute)
                                    (attributes {:expected i8}))
                                  (operation
                                    (name irdl.is)
                                    (result-bindings [%10])
                                    (result-types !irdl.attribute)
                                    (attributes {:expected i16}))
                                  (operation
                                    (name irdl.is)
                                    (result-bindings [%11])
                                    (result-types !irdl.attribute)
                                    (attributes {:expected i32}))
                                  (operation
                                    (name irdl.is)
                                    (result-bindings [%12])
                                    (result-types !irdl.attribute)
                                    (attributes {:expected i64}))
                                  (operation
                                    (name irdl.is)
                                    (result-bindings [%13])
                                    (result-types !irdl.attribute)
                                    (attributes {:expected f32}))
                                  (operation
                                    (name irdl.is)
                                    (result-bindings [%14])
                                    (result-types !irdl.attribute)
                                    (attributes {:expected f64}))
                                  (operation
                                    (name irdl.any_of)
                                    (result-bindings [%15])
                                    (result-types !irdl.attribute)
                                    (operands %9 %10 %11 %12 %13 %14))
                                  (operation
                                    (name irdl.operands)
                                    (operands %15 %15)
                                    (attributes {:names ["lhs" "rhs"] :variadicity #irdl<variadicity_array[ single,  single]>}))
                                  (operation
                                    (name irdl.results)
                                    (operands %15)
                                    (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>}))))))))))))))
          (operation
            (name builtin.module)
            (regions
              (region
                (block
                  (arguments [])
                  (operation
                    (name transform.with_pdl_patterns)
                    (regions
                      (region
                        (block [^bb0]
                          (arguments [(: %arg0 !transform.any_op)])
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mymath_to_arith})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%4])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%5])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%6])
                                    (result-types !pdl.type))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%7])
                                    (result-types !pdl.operation)
                                    (operands %4 %5 %6)
                                    (attributes {:attributeValueNames [] :opName "mymath.add" :operandSegmentSizes array<i32: 2, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operands %7)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%8])
                                            (result-types !pdl.operation)
                                            (operands %4 %5 %6)
                                            (attributes {:attributeValueNames [] :opName "arith.addi" :operandSegmentSizes array<i32: 2, 0, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operands %7 %8)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name transform.sequence)
                            (operands %arg0)
                            (attributes {:failure_propagation_mode (: 1 i32) :operandSegmentSizes array<i32: 1, 0>})
                            (regions
                              (region
                                (block [^bb0]
                                  (arguments [(: %arg1 !transform.any_op)])
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%3])
                                    (result-types !transform.any_op)
                                    (operands %arg1)
                                    (attributes {:pattern_name @mymath_to_arith}))
                                  (operation
                                    (name transform.yield))))))))))))))
          (operation
            (name func.func)
            (attributes {:function_type (!function (inputs) (results i32)) :sym_name @main})
            (regions
              (region
                (block
                  (arguments [])
                  (operation
                    (name arith.constant)
                    (result-bindings [%0])
                    (result-types i32)
                    (attributes {:value (: 10 i32)}))
                  (operation
                    (name arith.constant)
                    (result-bindings [%1])
                    (result-types i32)
                    (attributes {:value (: 32 i32)}))
                  (operation
                    (name mymath.add)
                    (result-bindings [%2])
                    (result-types i32)
                    (operands %0 %1))
                  (operation
                    (name func.return)
                    (operands %2)))))))))))
