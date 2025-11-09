;; Demonstration of op macro with optional operands vector

;; Case 1: Attributes and operands (both present)
(op %result (: i32) (arith.constant {value: 42} [%input]))

;; Case 2: Attributes only (no operands vector)
(op %const (: i32) (arith.constant {value: 42}))

;; Case 3: Operands only (no attributes)
(op %sum (: i32) (arith.addi [%a %b]))

;; Case 4: Neither attributes nor operands
(op %zero (: i32) (arith.constant))

;; Case 5: With operands and regions
(op %result (: i32)
    (scf.if [%cond]
            (region
              (block []
                (op (scf.yield [%true_val]))))
            (region
              (block []
                (op (scf.yield [%false_val]))))))

;; Case 6: With regions but no operands (just regions as arguments)
(op %result (: i32)
    (test.op
      (region
        (block []
          (op (test.yield [%val]))))))
