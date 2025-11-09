;; IRDL: Define mymath dialect
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
                  (result-bindings [%0])
                  (result-types !irdl.attribute)
                  (attributes {:expected i8}))
                (operation
                  (name irdl.is)
                  (result-bindings [%1])
                  (result-types !irdl.attribute)
                  (attributes {:expected i16}))
                (operation
                  (name irdl.is)
                  (result-bindings [%2])
                  (result-types !irdl.attribute)
                  (attributes {:expected i32}))
                (operation
                  (name irdl.is)
                  (result-bindings [%3])
                  (result-types !irdl.attribute)
                  (attributes {:expected i64}))
                (operation
                  (name irdl.is)
                  (result-bindings [%4])
                  (result-types !irdl.attribute)
                  (attributes {:expected f32}))
                (operation
                  (name irdl.is)
                  (result-bindings [%5])
                  (result-types !irdl.attribute)
                  (attributes {:expected f64}))
                (operation
                  (name irdl.any_of)
                  (operands %0 %1 %2 %3 %4 %5)
                  (result-bindings [%arith_type])
                  (result-types !irdl.attribute))
                (operation
                  (name irdl.operands)
                  (operands %arith_type %arith_type)
                  (attributes {:names ["lhs" "rhs"] :variadicity #irdl<variadicity_array[ single,  single]>}))
                (operation
                  (name irdl.results)
                  (operands %arith_type)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>}))))))))))

;; Transform: mymath.add → arith.addi
(operation
  (name builtin.module)
  (attributes {:transform.with_named_sequence unit})
  (regions
    (region
      (block
        (arguments [])
        (operation
          (name transform.named_sequence)
          (attributes {:sym_name @__transform_main :function_type (!function (inputs !transform.any_op) (results))})
          (regions
            (region
              (block [^entry]
                (arguments [(: %arg0 !transform.any_op)])
                (operation
                  (name transform.with_pdl_patterns)
                  (operands %arg0)
                  (regions
                    (region
                      (block
                        (arguments [(: %payload !transform.any_op)])
                        (operation
                          (name pdl.pattern)
                          (attributes {:benefit (: 1 i16) :sym_name @mymath_to_arith})
                          (regions
                            (region
                              (block
                                (arguments [])
                                (operation
                                  (name pdl.operand)
                                  (result-bindings [%lhs])
                                  (result-types !pdl.value))
                                (operation
                                  (name pdl.operand)
                                  (result-bindings [%rhs])
                                  (result-types !pdl.value))
                                (operation
                                  (name pdl.type)
                                  (result-bindings [%result_type])
                                  (result-types !pdl.type))
                                (operation
                                  (name pdl.operation)
                                  (operands %lhs %rhs %result_type)
                                  (result-bindings [%mymath_op])
                                  (result-types !pdl.operation)
                                  (attributes {:attributeValueNames [] :opName "mymath.add" :operandSegmentSizes array<i32: 2, 0, 1>}))
                                (operation
                                  (name pdl.rewrite)
                                  (operands %mymath_op)
                                  (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                  (regions
                                    (region
                                      (block
                                        (arguments [])
                                        (operation
                                          (name pdl.operation)
                                          (operands %lhs %rhs %result_type)
                                          (result-bindings [%arith_op])
                                          (result-types !pdl.operation)
                                          (attributes {:attributeValueNames [] :opName "arith.addi" :operandSegmentSizes array<i32: 2, 0, 1>}))
                                        (operation
                                          (name pdl.replace)
                                          (operands %mymath_op %arith_op)
                                          (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                        (operation
                          (name transform.sequence)
                          (operands %payload)
                          (attributes {:failure_propagation_mode (: 1 i32) :operandSegmentSizes array<i32: 1, 0>})
                          (regions
                            (region
                              (block [^bb1]
                                (arguments [(: %arg1 !transform.any_op)])
                                (operation
                                  (name transform.pdl_match)
                                  (operands %arg1)
                                  (result-bindings [%matched])
                                  (result-types !transform.any_op)
                                  (attributes {:pattern_name @mymath_to_arith}))
                                (operation
                                  (name transform.yield))))))))))
                (operation
                  (name transform.yield))))))))))

;; Application code
(operation
  (name func.func)
  (attributes {:sym_name @main :function_type (!function (inputs) (results i32))})
  (regions
    (region
      (block [^entry]
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%c10])
          (result-types i32)
          (attributes {:value (: 10 i32)}))
        (operation
          (name arith.constant)
          (result-bindings [%c32])
          (result-types i32)
          (attributes {:value (: 32 i32)}))
        (operation
          (name mymath.add)
          (operands %c10 %c32)
          (result-bindings [%result])
          (result-types i32))
        (operation
          (name func.return)
          (operands %result))))))
