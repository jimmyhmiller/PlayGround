;; Minimal test to verify transforms work

(operation
  (name builtin.module)
  (attributes {:transform.with_named_sequence unit})
  (regions
    (region
      (block
        (operation
          (name transform.named_sequence)
          (attributes {:sym_name @__transform_main})
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
                        ;; Pattern: match any mlsp.string_const
                        (operation
                          (name pdl.pattern)
                          (attributes {:benefit (: 1 i16) :sym_name @test_pattern})
                          (regions
                            (region
                              (block
                                (operation
                                  (name pdl.type)
                                  (result-bindings [%t])
                                  (result-types !pdl.type)
                                  (attributes {:constantType !llvm.ptr}))
                                (operation
                                  (name pdl.operation)
                                  (result-bindings [%op])
                                  (result-types !pdl.operation)
                                  (operands %t)
                                  (attributes {:opName "mlsp.string_const"}
                                             :operandSegmentSizes array<i32: 0, 0, 1>))
                                (operation
                                  (name pdl.rewrite)
                                  (operands %op)
                                  (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                  (regions
                                    (region
                                      (block
                                        (operation
                                          (name pdl.operation)
                                          (result-bindings [%new])
                                          (result-types !pdl.operation)
                                          (operands %t)
                                          (attributes {:opName "llvm.mlir.zero"}
                                                     :operandSegmentSizes array<i32: 0, 0, 1>))
                                        (operation
                                          (name pdl.replace)
                                          (operands %op %new)
                                          (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))

                        ;; Apply the pattern
                        (operation
                          (name transform.sequence)
                          (operands %arg1)
                          (attributes {:failure_propagation_mode (: 1 i32)}
                                     :operandSegmentSizes array<i32: 1, 0>)
                          (regions
                            (region
                              (block [^bb0]
                                (arguments [(: %arg2 !transform.any_op)])
                                (operation
                                  (name transform.pdl_match)
                                  (operands %arg2)
                                  (result-bindings [%m])
                                  (result-types !transform.any_op)
                                  (attributes {:pattern_name @test_pattern}))
                                (operation
                                  (name transform.yield))))))))))
                (operation
                  (name transform.yield))))))))))

(llvm.mlir.global {:sym_name @test_str
                   :value "test\\00"
                   :global_type !llvm.array<5 x i8>
                   :linkage #llvm.linkage<internal>
                   :constant true})

(defn test [] !llvm.ptr
  (op %0 (: !llvm.ptr) (mlsp.string_const {:global @test_str}))
  (return %0))
