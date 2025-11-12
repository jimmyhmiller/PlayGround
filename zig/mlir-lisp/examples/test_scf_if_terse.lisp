;; Test terse scf.if syntax
;; This should compile and demonstrate implicit yield insertion

(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @test_scf_if_terse
      :function_type (!function (inputs i1 i32) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [
            (: %cond i1)
            (: %val i32)
          ])

          ;; Test 1: Simple terse scf.if with value IDs
          (declare result1 (:
            (scf.if %cond
              (region %val)           ;; Then: implicit (scf.yield %val)
              (region                 ;; Else: implicit (scf.yield %c0)
                (declare c0 (arith.constant {:value (: 0 i32)}))
                %c0))
            i32))

          ;; Test 2: Terse scf.if with operations in regions
          (declare result2 (:
            (scf.if %cond
              (region
                (declare doubled (arith.muli %val %val))
                %doubled)             ;; Implicit (scf.yield %doubled)
              (region
                (declare c1 (arith.constant {:value (: 1 i32)}))
                (declare incremented (arith.addi %val %c1))
                %incremented))        ;; Implicit (scf.yield %incremented)
            i32))

          ;; Return the second result
          (operation
            (name func.return)
            (operands %result2)))))))