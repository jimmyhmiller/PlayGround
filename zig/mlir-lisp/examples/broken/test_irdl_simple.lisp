;; Simple IRDL Dialect Test
;; This file demonstrates:
;; 1. Defining a custom dialect using IRDL
;; 2. Using operations from that dialect
;; 3. The system auto-detecting and loading it

;; Step 1: Define a simple "demo" dialect with one operation
;; demo.constant - creates a constant i32 value
(operation
  (name irdl.dialect)
  (attributes {:sym_name @demo})
  (regions
    (region
      (block
        ;; Define demo.constant operation
        (operation
          (name irdl.operation)
          (attributes {:sym_name @constant})
          (regions
            (region
              (block
                ;; Result type: i32
                (op %i32_type (: !irdl.attribute) (irdl.is {:expected i32} []))

                ;; Attributes: value (any attribute - accepts integer attributes)
                (op %attr_type (: !irdl.attribute) (irdl.any []))
                (operation
                  (name irdl.attributes)
                  (attributes {:attributeValueNames ["value"]})
                  (operand-uses %attr_type))

                ;; Results: one i32 value
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %i32_type))))))))))

;; Step 2: Use the demo.constant operation
(defn main [] i64
  ;; Use our custom operation
  (op %val (: i32) (demo.constant {:value (: 42 i32)} []))

  ;; Convert to i64 for return
  (op %result (: i64) (arith.extsi {:fastmath #arith.fastmath<none>} [%val]))

  (return %result))
