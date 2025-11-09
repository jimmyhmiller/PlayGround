;; Minimal PDL Transform Example
;; Based on melior structure: custom.magic → arith.constant 42

;; IRDL: Define custom.magic operation
(operation
  (name irdl.dialect)
  (attributes {:sym_name @custom})
  (regions
    (region
      (block
        (operation
          (name irdl.operation)
          (attributes {:sym_name @magic})
          (regions
            (region
              (block
                (op %i32 (: !irdl.attribute) (irdl.is {:expected i32} []))
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operands %i32))))))))))

;; Transform: Use transform.with_pdl_patterns (like melior)
(operation
  (name builtin.module)
  (attributes {:transform.with_named_sequence unit})
  (regions
    (region
      (block
        (operation
          (name transform.named_sequence)
          (attributes {:sym_name @__transform_main :function_type (!function (inputs !transform.any_op) (results))})
          (regions
            (region
              (block [^entry]
                (arguments [(: %arg0 !transform.any_op)])
                ;; Apply the PDL patterns
                (operation
                  (name transform.apply_patterns.transform.with_pdl_patterns)
                  (operands %arg0)
                  (result-bindings [%result])
                  (result-types !transform.any_op)
                  (regions
                    (region
                      (block
                        (operation
                          (name pdl.pattern)
                          (attributes {:benefit (: 1 i16) :sym_name @replace_magic})
                          (regions
                            (region
                              (block
                                (operation
                                  (name pdl.type)
                                  (attributes {:constantType i32})
                                  (result-bindings [%type])
                                  (result-types !pdl.type))
                                (operation
                                  (name pdl.operation)
                                  (attributes {:opName "custom.magic" :operandSegmentSizes array<i32: 0, 0, 1>})
                                  (operands %type)
                                  (result-bindings [%op])
                                  (result-types !pdl.operation))
                                (operation
                                  (name pdl.rewrite)
                                  (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                  (operands %op)
                                  (regions
                                    (region
                                      (block
                                        (operation
                                          (name pdl.attribute)
                                          (attributes {:value (: 42 i32)})
                                          (result-bindings [%attr])
                                          (result-types !pdl.attribute))
                                        (operation
                                          (name pdl.operation)
                                          (attributes {:attributeValueNames ["value"] :opName "arith.constant" :operandSegmentSizes array<i32: 0, 1, 1>})
                                          (operands %attr %type)
                                          (result-bindings [%new])
                                          (result-types !pdl.operation))
                                        (operation
                                          (name pdl.replace)
                                          (attributes {:operandSegmentSizes array<i32: 1, 1, 0>})
                                          (operands %op %new))))))))))))))
                (operation
                  (name transform.yield))))))))))

;; Application code
(operation
  (name func.func)
  (attributes {:sym_name @main :function_type (!function (inputs) (results i64))})
  (regions
    (region
      (block [^entry]
        (arguments [])
        (operation
          (name custom.magic)
          (result-bindings [%magic])
          (result-types i32))
        (operation
          (name arith.extsi)
          (result-bindings [%result])
          (result-types i64)
          (operands %magic))
        (operation
          (name func.return)
          (operands %result))))))
