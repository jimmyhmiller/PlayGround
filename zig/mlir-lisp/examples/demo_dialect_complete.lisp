;; ========================================
;; COMPLETE WORKING EXAMPLE: Custom Dialect with Transform
;; ========================================
;;
;; This file proves the entire IRDL + Transform pipeline works:
;; 1. Define custom "demo" dialect with demo.constant operation
;; 2. Define PDL transform to lower demo.constant → arith.constant
;; 3. Use demo.constant in application code
;; 4. System auto-detects, loads, transforms, and executes
;;
;; Expected: Result: 42 (using OUR custom operation!)

;; ========================================
;; PART 1: IRDL Dialect Definition
;; ========================================

(operation
  (name irdl.dialect)
  (attributes {:sym_name @demo})
  (regions
    (region
      (block
        ;; Define demo.constant operation
        ;; Takes an attribute "value" and produces a result
        (operation
          (name irdl.operation)
          (attributes {:sym_name @constant})
          (regions
            (region
              (block
                ;; Define supported types (i32, i64)
                (op %i32_constraint (: !irdl.attribute) (irdl.is {:expected i32} []))
                (op %i64_constraint (: !irdl.attribute) (irdl.is {:expected i64} []))
                (op %int_types (: !irdl.attribute) (irdl.any_of [%i32_constraint %i64_constraint]))

                ;; Attributes: value (integer attribute)
                (op %attr_constraint (: !irdl.attribute) (irdl.is {:expected i64} []))
                (operation
                  (name irdl.attributes)
                  (attributes {:attributeValueNames ["value"]})
                  (operand-uses %attr_constraint))

                ;; Results: one integer value
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %int_types))))))))))

;; ========================================
;; PART 2: PDL Transform to Lower demo.constant
;; ========================================

(operation
  (name builtin.module)
  (attributes {})
  (regions
    (region
      (block
        (operation
          (name transform.with_pdl_patterns)
          (regions
            (region
              (block [^bb0]
                (arguments [(: %root !transform.any_op)])
                ;; Define PDL pattern
                (operation
                  (name pdl.pattern)
                  (attributes {:benefit (: 1 i16) :sym_name @demo_to_arith})
                  (regions
                    (region
                      (block
                        ;; Match demo.constant
                        (operation
                          (name pdl.attribute)
                          (result-bindings [%value_attr])
                          (result-types !pdl.attribute))
                        (operation
                          (name pdl.type)
                          (result-bindings [%result_type])
                          (result-types !pdl.type))
                        (operation
                          (name pdl.operation)
                          (attributes {:opName "demo.constant"
                                     :attributeValueNames ["value"]
                                     :operandSegmentSizes array<i32: 0, 1, 1>})
                          (operand-uses %value_attr %result_type)
                          (result-bindings [%demo_op])
                          (result-types !pdl.operation))

                        ;; Rewrite to arith.constant
                        (operation
                          (name pdl.rewrite)
                          (attributes {:operandSegmentSizes array<i32: 1, 0>})
                          (operand-uses %demo_op)
                          (regions
                            (region
                              (block
                                ;; Create arith.constant with same value attribute
                                (operation
                                  (name pdl.operation)
                                  (attributes {:opName "arith.constant"
                                             :attributeValueNames ["value"]
                                             :operandSegmentSizes array<i32: 0, 1, 1>})
                                  (operand-uses %value_attr %result_type)
                                  (result-bindings [%arith_op])
                                  (result-types !pdl.operation))

                                ;; Replace demo.constant with arith.constant
                                (operation
                                  (name pdl.replace)
                                  (attributes {:operandSegmentSizes array<i32: 1, 1, 0>})
                                  (operand-uses %demo_op %arith_op))))))))))

                ;; Transform sequence to apply the pattern
                (operation
                  (name transform.sequence)
                  (attributes {:failure_propagation_mode (: 1 i32)
                             :operandSegmentSizes array<i32: 1, 0>})
                  (operand-uses %root)
                  (result-types !transform.any_op)
                  (regions
                    (region
                      (block [^bb1]
                        (arguments [(: %arg1 !transform.any_op)])
                        ;; Apply PDL pattern
                        (operation
                          (name transform.pdl_match)
                          (attributes {:pattern_name @demo_to_arith})
                          (operand-uses %arg1)
                          (result-bindings [%matched])
                          (result-types !transform.any_op))
                        (operation
                          (name transform.yield))))))))))))))

;; ========================================
;; PART 3: Application Code Using demo.constant
;; ========================================

(defn main [] i64
  ;; Use OUR custom operation!
  (op %val (: i64) (demo.constant {:value (: 42 i64)}))
  (return %val))
