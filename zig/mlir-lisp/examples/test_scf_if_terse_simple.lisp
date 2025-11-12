;; Simplest possible terse scf.if test
;; Explicit yields, explicit types

(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @test_simple
      :function_type (!function (inputs i1 i32) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [
            (: %cond i1)
            (: %n i32)
          ])

          ;; Terse scf.if with explicit type and explicit yields
          (operation
            (name scf.if)
            (result-bindings [%result])
            (result-types i32)
            (operands %cond)
            (regions
              (region
                (scf.yield %n))
              (region
                (declare c0 (arith.constant {:value (: 0 i32)}))
                (scf.yield %c0))))

          (operation
            (name func.return)
            (operands %result)))))))
