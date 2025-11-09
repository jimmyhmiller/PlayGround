(mlir
  (operation
    (name builtin.module)
    (regions
      (region
        (block
          (arguments [])
          (operation
            (name llvm.func)
            (attributes {:CConv #llvm.cconv<ccc> :function_type (!function !llvm.func<ptr (i64)>) :linkage #llvm.linkage<external> :sym_name @malloc :unnamed_addr (: 0 i64) :visibility_ (: 0 i64)})
            (regions
              (region)))
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
                                            (result-bindings [%1])
                                            (result-types !pdl.type)
                                            (attributes {:constantType !llvm.ptr}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%2])
                                            (result-types !pdl.operation)
                                            (operands %1)
                                            (attributes {:attributeValueNames [] :opName "mlsp.string_const" :operandSegmentSizes array<i32: 0, 0, 1>}))
                                          (operation
                                            (name pdl.rewrite)
                                            (operands %2)
                                            (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                            (regions
                                              (region
                                                (block
                                                  (arguments [])
                                                  (operation
                                                    (name pdl.operation)
                                                    (result-bindings [%3])
                                                    (result-types !pdl.operation)
                                                    (operands %1)
                                                    (attributes {:attributeValueNames [] :opName "llvm.mlir.addressof" :operandSegmentSizes array<i32: 0, 0, 1>}))
                                                  (operation
                                                    (name pdl.replace)
                                                    (operands %2 %3)
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
                                            (name transform.yield))))))))))
                          (operation
                            (name transform.yield)))))))))))))))
