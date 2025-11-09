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
                    (name func.func)
                    (attributes {:function_type (!function (inputs) (results i32)) :sym_name @test})
                    (regions
                      (region
                        (block
                          (arguments [])
                          (operation
                            (name custom.foo)
                            (result-bindings [%4])
                            (result-types i32))
                          (operation
                            (name func.return)
                            (operands %4))))))))))
          (operation
            (name builtin.module)
            (attributes {:sym_name @patterns})
            (regions
              (region
                (block
                  (arguments [])
                  (operation
                    (name pdl.pattern)
                    (attributes {:benefit (: 1 i16) :sym_name @replace_foo})
                    (regions
                      (region
                        (block
                          (arguments [])
                          (operation
                            (name pdl.type)
                            (result-bindings [%0])
                            (result-types !pdl.type)
                            (attributes {:constantType i32}))
                          (operation
                            (name pdl.operation)
                            (result-bindings [%1])
                            (result-types !pdl.operation)
                            (operands %0)
                            (attributes {:attributeValueNames [] :opName "custom.foo" :operandSegmentSizes array<i32: 0, 0, 1>}))
                          (operation
                            (name pdl.rewrite)
                            (operands %1)
                            (attributes {:operandSegmentSizes array<i32: 1, 0>})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.attribute)
                                    (result-bindings [%2])
                                    (result-types !pdl.attribute)
                                    (attributes {:value (: 42 i32)}))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%3])
                                    (result-types !pdl.operation)
                                    (operands %2 %0)
                                    (attributes {:attributeValueNames ["value"] :opName "arith.constant" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                  (operation
                                    (name pdl.replace)
                                    (operands %1 %3)
                                    (attributes {:operandSegmentSizes array<i32: 1, 1, 0>})))))))))))))))))))
