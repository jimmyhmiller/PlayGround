;; Simpler PDL example matching melior structure
;; custom.magic → arith.constant 42

;; IRDL: Define custom dialect
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

;; Transform: PDL pattern (matching melior structure)
(operation
  (name builtin.module)
  (regions
    (region
      (block
        (operation
          (name transform.with_pdl_patterns)
          (regions
            (region
              (block [^entry]
                (arguments [(: %root !transform.any_op)])
                (operation
                  (name pdl.pattern)
                  (attributes {:benefit (: 1 i16) :sym_name @replace_magic})
                  (regions
                    (region
                      (block
                        ;; Match: custom.magic () -> i32
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
                        ;; Rewrite: arith.constant {value = 42 : i32}
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
                                  (operands %op %new))))))))))
                ;; Transform sequence to apply the pattern
                (operation
                  (name transform.sequence)
                  (attributes {:failures #transform<failure_propagation_mode propagate>})
                  (operands %root)
                  (regions
                    (region
                      (block [^bb1]
                        (arguments [(: %arg1 !transform.any_op)])
                        (operation
                          (name transform.pdl_match)
                          (attributes {:pattern_name @replace_magic})
                          (operands %arg1)
                          (result-bindings [%matched])
                          (result-types !transform.any_op))
                        (operation
                          (name transform.yield))))))))))))))

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
